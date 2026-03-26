#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <omp.h>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "utils/percentile_stats.h"
#include "utils/tsl/robin_set.h"
#include <unordered_map>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "index.h"
#include "utils/timer.h"
#include "utils.h"
#include "utils/lock_table.h"
#include "utils/prune_neighbors.h"


namespace pipeann {
  // Initialize an index with metric m, load the data of type T with filename
  // (bin). The index will be dynamically resized as needed.
  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, const size_t dim) : _dist_metric(m), _dim(dim) {
    constexpr uint64_t kLockTableEntries = 131072;  // ~1MB lock table.
    this->_locks = new pipeann::LockTable(kLockTableEntries);
    LOG(INFO) << "Getting distance function for metric: " << get_metric_str(m);
    this->_distance = get_distance_function<T>(m);
    _width = 0;
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    delete this->_distance;
    delete this->_locks;
  }

  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_tags(std::string tags_file) {
    std::vector<TagT> tag_data(_nd);
    for (uint32_t i = 0; i < _nd; i++) {
      if (_location_to_tag.find(i) != _location_to_tag.end()) {
        tag_data[i] = _location_to_tag[i];
      }
    }
    return save_bin<TagT>(tags_file, tag_data.data(), _nd, 1);
  }

  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_data(std::string data_file) {
    return save_bin<T>(data_file, _data.data(), _nd, _dim);
  }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_graph(std::string graph_file) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    out.seekp(0, out.beg);
    uint64_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    unsigned ep_u32 = _ep;
    out.write((char *) &ep_u32, sizeof(unsigned));
    uint64_t num_frozen_pts = 0;  // For backward compatibility
    out.write((char *) &num_frozen_pts, sizeof(uint64_t));
    for (unsigned i = 0; i < _nd; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = std::max(max_degree, (uint32_t) _final_graph[i].size());
      index_size += (uint64_t) (sizeof(unsigned) * (GK + 1));
    }
    out.seekp(0, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(uint32_t));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_delete_list(const std::string &filename, uint64_t file_offset) {
    if (_delete_set.size() == 0) {
      return 0;
    }
    std::unique_ptr<uint32_t[]> delete_list = std::make_unique<uint32_t[]>(_delete_set.size());
    uint32_t i = 0;
    for (auto &del : _delete_set) {
      delete_list[i++] = del;
    }
    return save_bin<uint32_t>(filename, delete_list.get(), _delete_set.size(), 1, file_offset);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _change_lock.lock();

    std::string graph_file = std::string(filename);
    std::string tags_file = std::string(filename) + ".tags";
    std::string data_file = std::string(filename) + ".data";
    std::string delete_list_file = std::string(filename) + ".del";

    // Because the save_* functions use append mode, ensure that
    // the files are deleted before save. Ideally, we should check
    // the error code for delete_file, but will ignore now because
    // delete should succeed if save will succeed.
    delete_file(graph_file);
    save_graph(graph_file);
    delete_file(data_file);
    save_data(data_file);
    delete_file(tags_file);
    save_tags(tags_file);
    delete_file(delete_list_file);
    save_delete_list(delete_list_file);

    _change_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    LOG(INFO) << "Time taken for save: " << timespan.count() << "s.";
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tags(const std::string tag_filename, uint64_t file_frozen_pts, size_t offset) {
    if (!file_exists(tag_filename)) {
      LOG(ERROR) << "Tag file provided does not exist!";
      crash();
    }

    size_t file_dim, file_num_points;
    TagT *tag_data;
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points, file_dim, offset);

    if (file_dim != 1) {
      LOG(ERROR) << "ERROR: Loading " << file_dim << " dimensions for tags," << "but tag file must have 1 dimension.";
      crash();
    }

    // Use file_frozen_pts for backward compatibility with old static index.
    size_t num_data_points = file_frozen_pts > 0 ? file_num_points - 1 : file_num_points;
    for (uint32_t i = 0; i < (uint32_t) num_data_points; i++) {
      TagT tag = *(tag_data + i);
      if (_delete_set.find(i) == _delete_set.end()) {
        _location_to_tag[i] = tag;
        _tag_to_location[tag] = (uint32_t) i;
      }
    }
    LOG(INFO) << "Tags loaded.";
    delete[] tag_data;
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_data(std::string filename, size_t offset) {
    LOG(INFO) << "Loading data from " << filename << " offset " << offset;
    if (!file_exists(filename)) {
      LOG(ERROR) << "ERROR: data file " << filename << " does not exist.";
      crash();
    }

    size_t file_dim, file_num_points;
    pipeann::load_bin<T>(filename, _data, file_num_points, file_dim, offset);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim) {
      LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension," << "but file has " << file_dim
                 << " dimension.";
      crash();
    }

    _final_graph.resize(file_num_points);
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_delete_set(const std::string &filename, size_t offset) {
    std::unique_ptr<uint32_t[]> delete_list;
    uint64_t npts, ndim;
    load_bin<uint32_t>(filename, delete_list, npts, ndim, offset);
    assert(ndim == 1);
    for (size_t i = 0; i < npts; i++) {
      _delete_set.insert(delete_list[i]);
    }
    return npts;
  }

  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename) {
    _change_lock.lock();

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;
    uint64_t file_frozen_pts = 0;

    std::string data_file = std::string(filename) + ".data";
    std::string tags_file = std::string(filename) + ".tags";
    std::string delete_set_file = std::string(filename) + ".del";
    std::string graph_file = std::string(filename);
    data_file_num_pts = load_data(data_file);
    if (file_exists(delete_set_file)) {
      load_delete_set(delete_set_file);
    }
    // Load graph first to get file_frozen_pts for backward compatibility.
    graph_num_pts = load_graph(graph_file, data_file_num_pts, file_frozen_pts);
    tags_file_num_pts = load_tags(tags_file, file_frozen_pts);

    if (data_file_num_pts != graph_num_pts || data_file_num_pts != tags_file_num_pts) {
      LOG(ERROR) << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
                 << graph_num_pts << " from graph, and " << tags_file_num_pts << " tags, with file_frozen_pts being "
                 << file_frozen_pts << ".";
      crash();
    }

    // Use file_frozen_pts for _nd calculation (backward compatibility with old static index).
    _nd = data_file_num_pts - file_frozen_pts;
    _empty_slots.clear();
    for (uint32_t i = _nd; i < max_points(); i++) {
      _empty_slots.insert(i);
    }

    // For old static index with frozen point, we now just use _ep as loaded from file.
    // The _ep already points to a valid entry point.

    LOG(INFO) << "_nd: " << _nd << " _ep: " << _ep << " size(_location_to_tag): " << _location_to_tag.size()
              << " size(_tag_to_location):" << _tag_to_location.size() << " Max points: " << max_points();
    _change_lock.unlock();
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_graph(std::string filename, size_t expected_num_points, uint64_t &out_file_frozen_pts,
                                    size_t offset) {
    std::ifstream in(filename, std::ios::binary);
    in.seekg(offset, in.beg);
    size_t expected_file_size;
    in.read((char *) &expected_file_size, sizeof(uint64_t));
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    in.read((char *) &out_file_frozen_pts, sizeof(uint64_t));

    // Support loading old static index (out_file_frozen_pts == 0) for backward compatibility.
    // We just use _ep as is from the file.
    LOG(INFO) << "Loading vamana index " << filename << " (file_frozen_pts=" << out_file_frozen_pts << ")...";

    // Sanity check. In case the user gave us fewer points as max_points than
    // the number
    // of points in the dataset, resize the _final_graph to the larger size.
    if (max_points() < expected_num_points) {
      LOG(INFO) << "Number of points in data: " << expected_num_points
                << " is more than max_points argument: " << _final_graph.size()
                << " Setting max points to: " << expected_num_points;
      resize(expected_num_points);
    }

    size_t bytes_read = 24;
    size_t cc = 0;
    unsigned nodes = 0;
    while (bytes_read != expected_file_size) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (k == 0) {
        LOG(ERROR) << "ERROR: Point found with no out-neighbors, point#" << nodes;
      }
      //      if (in.eof())
      //        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      tmp.reserve(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      _final_graph[nodes - 1].swap(tmp);
      bytes_read += sizeof(uint32_t) * ((uint64_t) k + 1);
      if (nodes % 10000000 == 0)
        LOG(INFO) << "Loaded " << nodes / 1000000 << "M nodes...";
    }

    LOG(INFO) << "done. Index has " << nodes << " nodes and " << cc << " out-edges, _ep is set to " << _ep;
    return nodes;
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
    // allocate and init centroid
    std::vector<float> center(_dim, 0.0f);

    for (size_t i = 0; i < _nd; i++)
      for (size_t j = 0; j < _dim; j++)
        center[j] += (float) _data[i * _dim + j];

    for (auto &c : center)
      c /= (float) _nd;

    // compute all to one distance, updating the atomic variables should not be the bottleneck.
    constexpr uint64_t kDistNum = 256;
    struct alignas(64) AtomicDistance {
      unsigned idx = 0;
      float dist = std::numeric_limits<float>::max();
      std::mutex lk;

      void update(unsigned i, float d) {
        std::lock_guard<std::mutex> guard(lk);
        if (d < dist) {
          dist = d;
          idx = i;
        }
      }
    };
    AtomicDistance atomic_dists[kDistNum];

#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t) _nd; i++) {
      // extract point and distance reference
      float dist = 0;
      const T *cur_vec = _data.data() + (i * _dim);
      for (size_t j = 0; j < _dim; j++) {
        dist += (center[j] - (float) cur_vec[j]) * (center[j] - (float) cur_vec[j]);
      }
      atomic_dists[(i / 65536) % kDistNum].update(i, dist);
    }

    unsigned min_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (unsigned i = 0; i < kDistNum; i++) {
      if (atomic_dists[i].dist < min_dist) {
        min_idx = atomic_dists[i].idx;
        min_dist = atomic_dists[i].dist;
      }
    }
    return min_idx;
  }

  /* iterate_to_fixed_point():
   * node_coords : point whose neighbors to be found.
   * init_ids : ids of initial search list.
   * Lsize : size of list.
   * beam_width: beam_width when performing indexing
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * expanded_nodes_ids : will contain all the nodes that are expanded during
   * search.
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(const T *node_coords, const unsigned Lsize,
                                                                       const std::vector<unsigned> &init_ids,
                                                                       std::vector<Neighbor> &expanded_nodes_info,
                                                                       tsl::robin_set<unsigned> &expanded_nodes_ids,
                                                                       std::vector<Neighbor> &best_L_nodes,
                                                                       QueryStats *stats) {
    best_L_nodes.resize(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);

    unsigned l = 0;
    Neighbor nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      assert(id < max_points());
      nn = Neighbor(id, _distance->compare(_data.data() + _dim * (size_t) id, node_coords, _dim), true);
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        best_L_nodes[l++] = nn;
      }
      if (l == Lsize)
        break;
    }

    Timer query_timer, io_timer, cpu_timer;

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        io_timer.reset();
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        expanded_nodes_info.emplace_back(best_L_nodes[k]);
        expanded_nodes_ids.insert(n);
        std::vector<unsigned> des;

        {
          // pipeann::SparseReadLockGuard<uint64_t> guard(&_locks, n);
          pipeann::LockGuard guard(_locks->rdlock(n));
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= max_points()) {
              LOG(ERROR) << "Wrong id found: " << _final_graph[n][m];
              crash();
            }
            des.emplace_back(_final_graph[n][m]);
          }
        }
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();  // read vec
        }

        cpu_timer.reset();

        for (unsigned m = 0; m < des.size(); ++m) {
          unsigned id = des[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            // io_timer.reset();
            if ((m + 1) < des.size()) {
              auto nextn = des[m + 1];
              pipeann::prefetch_vector((const char *) _data.data() + _dim * (size_t) nextn, sizeof(T) * _dim);
            }
            cmps++;

            float dist = _distance->compare(node_coords, _data.data() + _dim * (size_t) id, (unsigned) _dim);

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }
        if (stats != nullptr) {
          stats->cpu_us += cpu_timer.elapsed();  // compute + read nbr
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else
        k++;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(const size_t node_id, const unsigned Lindex,
                                          std::vector<Neighbor> &expanded_nodes_info) {
    const T *node_coords = _data.data() + _dim * node_id;
    std::vector<unsigned> init_ids{_ep};
    std::vector<Neighbor> best_L_nodes;
    tsl::robin_set<unsigned> expanded_nodes_ids;
    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info, expanded_nodes_ids, best_L_nodes);
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                                    const IndexBuildParameters &params) {
    assert(n >= 0 && n < _nd);

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < max_points());
      /* des_pool contains the neighbors of the neighbors of n */
      auto &des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool prune_needed = false;
      {
        pipeann::LockGuard guard(_locks->wrlock(des));
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (uint64_t) (SLACK_FACTOR * params.R)) {
            des_pool.emplace_back(n);
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        std::vector<Neighbor> pool;
        pool.reserve(copy_of_neighbors.size());

        for (auto cur_nbr : copy_of_neighbors) {
          if (cur_nbr != des) {
            float dist =
                _distance->compare(_data.data() + _dim * (size_t) des, _data.data() + _dim * (size_t) cur_nbr, _dim);
            pool.emplace_back(Neighbor(cur_nbr, dist, true));
          }
        }
        std::vector<unsigned> new_out_neighbors;
        pipeann::prune_neighbors(pool, new_out_neighbors, params, _dist_metric, [this](uint32_t a, uint32_t b) {
          return _distance->compare(_data.data() + _dim * a, _data.data() + _dim * b, _dim);
        });
        {
          // pipeann::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
          pipeann::LockGuard guard(_locks->wrlock(des));
          _final_graph[des].assign(new_out_neighbors.begin(), new_out_neighbors.end());
        }
      }
    }
  }

  // one-pass graph building.
  template<typename T, typename TagT>
  void Index<T, TagT>::link(IndexBuildParameters &params) {
    unsigned num_threads = params.num_threads;
    unsigned L = params.L;  // Search list size
    params.print();

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    int64_t n_vecs_to_visit = _nd;
    _ep = calculate_entry_point();

    std::vector<unsigned> init_ids;
    init_ids.emplace_back(_ep);

    pipeann::Timer link_timer;
#pragma omp parallel for schedule(dynamic)
    for (int64_t node = 0; node < n_vecs_to_visit; node++) {
      // search.
      std::vector<Neighbor> pool;
      pool.reserve(2 * L);
      get_expanded_nodes(node, L, pool);
      // remove the node itself from pool.
      pool.erase(std::remove_if(pool.begin(), pool.end(), [node](const Neighbor &n) { return n.id == node; }),
                 pool.end());

      // prune neighbors.
      std::vector<unsigned> pruned_list;
      pipeann::prune_neighbors(pool, pruned_list, params, _dist_metric, [this](uint32_t a, uint32_t b) {
        return _distance->compare(_data.data() + _dim * a, _data.data() + _dim * b, _dim);
      });

      {
        pipeann::LockGuard guard(_locks->wrlock(node));
        _final_graph[node].assign(pruned_list.begin(), pruned_list.end());
      }

      inter_insert(node, pruned_list, params);

      if (node % 100000 == 0) {
        std::cerr << "\r" << (100.0 * node) / (n_vecs_to_visit) << "% of index build completed.";
      }
    }

    final_prune(params);
    if (_nd > 0) {
      LOG(INFO) << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000) << "s";
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::final_prune(IndexBuildParameters &params) {
    int64_t n_vecs_to_visit = _nd;
    if (_nd > 0) {
      LOG(INFO) << "Starting final cleanup..";
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (int64_t node_ctr = 0; node_ctr < n_vecs_to_visit; node_ctr++) {
      auto node = node_ctr;
      if (_final_graph[node].size() > params.R) {
        std::vector<Neighbor> pool;
        std::vector<unsigned> new_out_neighbors;

        for (auto cur_nbr : _final_graph[node]) {
          if (cur_nbr != node) {
            float dist =
                _distance->compare(_data.data() + _dim * (size_t) node, _data.data() + _dim * (size_t) cur_nbr, _dim);
            pool.emplace_back(Neighbor(cur_nbr, dist, true));
          }
        }
        pipeann::prune_neighbors(pool, new_out_neighbors, params, _dist_metric, [this](uint32_t a, uint32_t b) {
          return _distance->compare(_data.data() + _dim * a, _data.data() + _dim * b, _dim);
        });

        _final_graph[node].clear();
        for (auto id : new_out_neighbors)
          _final_graph[node].emplace_back(id);
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename, const size_t num_points_to_load, IndexBuildParameters &params,
                             const std::vector<TagT> &tags, bool normalize_cosine) {
    if (!file_exists(filename)) {
      LOG(ERROR) << "Data file " << filename << " does not exist!!! Exiting....";
      crash();
    }

    LOG(INFO) << "Building index with normalize_cosine: " << normalize_cosine;

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      LOG(INFO) << "Starting with an empty index.";
      _nd = 0;
    } else {
      pipeann::load_bin<T>(filename, _data, file_num_points, file_dim);

      if (num_points_to_load > file_num_points) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points but file has "
                   << file_num_points << " points.";
        crash();
      }
      if (file_dim != _dim) {
        LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension," << "but file has " << file_dim
                   << " dimension.";
        crash();
      }

      _final_graph.resize(file_num_points);

      if (normalize_cosine && _dist_metric == Metric::COSINE) {
        for (size_t i = 0; i < file_num_points; i++) {
          pipeann::normalize_data(_data.data() + i * _dim, _data.data() + i * _dim, _dim);
        }
      }

      LOG(INFO) << "Loading only first " << num_points_to_load << " from file.. ";
      _nd = num_points_to_load;

      if (tags.size() != num_points_to_load) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points from file,"
                   << "but tags vector is of size " << tags.size() << ".";
        crash();
      }
      for (size_t i = 0; i < tags.size(); ++i) {
        _tag_to_location[tags[i]] = (unsigned) i;
        _location_to_tag[(unsigned) i] = tags[i];
      }
    }

    if (params.use_pipnn) {
      pipnn_link(params);
    } else {
      link(params);
    }

    size_t max_deg = 0, min_deg = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max_deg = std::max(max_deg, pool.size());
      min_deg = std::min(min_deg, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (_nd > 0) {
      LOG(INFO) << "Index built with degree: max:" << max_deg << " avg:" << (float) total / (float) (_nd)
                << " min:" << min_deg << " count(deg<2):" << cnt;
    }
    _width = std::max(_width, (unsigned) max_deg);

    // Initialize empty slots for future insertions
    for (uint32_t i = _nd; i < max_points(); i++) {
      _empty_slots.insert(i);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename, const size_t num_points_to_load, IndexBuildParameters &params,
                             const char *tag_filename, bool normalize_cosine) {
    std::vector<TagT> tags{};

    if (tag_filename == nullptr) {
      // Generate default tags
      tags.resize(num_points_to_load);
      std::iota(tags.begin(), tags.end(), TagT{0});
    } else {
      if (!file_exists(tag_filename)) {
        LOG(ERROR) << "Tag file " << tag_filename << " does not exist. Exiting...";
        crash();
      }
      LOG(INFO) << "Loading tags from " << tag_filename << " for vamana index build";
      size_t npts, ndim;
      pipeann::load_bin(tag_filename, tags, npts, ndim);
      if (npts != num_points_to_load) {
        std::stringstream sstream;
        sstream << "Loaded " << npts << " tags instead of expected number: " << num_points_to_load;
        LOG(ERROR) << sstream.str();
        crash();
      }
    }

    build(filename, num_points_to_load, params, tags, normalize_cosine);
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const size_t K, const unsigned L,
                                                       unsigned *indices, float *distances, QueryStats *stats) {
    std::vector<unsigned> init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor> best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *aligned_query;
    size_t allocSize = _dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _dim * sizeof(T));
    if (_dist_metric == pipeann::Metric::COSINE) {
      pipeann::normalize_data(aligned_query, query, _dim);
    } else {
      memcpy(aligned_query, query, _dim * sizeof(T));
    }
    auto retval = iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids,
                                         best_L_nodes, stats);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      if (it.id < max_points()) {
        indices[pos] = it.id;
        if (distances != nullptr)
          distances[pos] = it.distance;
        pos++;
      }
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const size_t K, const unsigned L, TagT *tags,
                                          float *distances) {
    uint32_t *indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t pos = 0;
    for (int i = 0; i < (int) L; ++i) {
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        if (distances != nullptr)
          distances[pos] = dist_interim[i];
        pos++;
        if (pos == K)
          break;
      }
    }
    delete[] indices;
    delete[] dist_interim;
    return pos;
  }

  template<typename T, typename TagT>
  uint32_t Index<T, TagT>::search_with_tags_fast(const T *normalized_query, const unsigned Lsize, TagT *tags,
                                                 float *dists) {
    std::vector<Neighbor> best_L_nodes(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }

    unsigned l = 0;
    Neighbor nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    auto id = _ep;
    nn = Neighbor(id, _distance->compare(_data.data() + _dim * (size_t) id, normalized_query, _dim), true);
    inserted_into_pool.insert(id);
    best_L_nodes[l++] = nn;

    unsigned k = 0, cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;

        auto &cur_v = _final_graph[n];
        for (unsigned m = 0; m < cur_v.size(); ++m) {
          unsigned id = cur_v[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            if ((m + 1) < cur_v.size()) {
              auto nextn = cur_v[m + 1];
              pipeann::prefetch_vector((const char *) _data.data() + _dim * (size_t) nextn, sizeof(T) * _dim);
            }

            float dist = _distance->compare(normalized_query, _data.data() + _dim * (size_t) id, (unsigned) _dim);
            cmps++;

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else {
        k++;
      }
    }
    for (uint32_t i = 0; i < Lsize; ++i) {
      tags[i] = _location_to_tag[best_L_nodes[i].id];
      dists[i] = best_L_nodes[i].distance;
    }
    return cmps;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    return _nd;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  // Consolidate deleted points: update neighbor lists and compact data.
  // Similar to merge_deletes in SSDIndex but for in-memory index.
  template<typename T, typename TagT>
  void Index<T, TagT>::consolidate(IndexBuildParameters &params) {
    if (_delete_set.empty()) {
      return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Consolidating " << _delete_set.size() << " deleted points, _nd: " << _nd
              << ", max_points: " << max_points();

    // Step 1: Build old_id -> new_id mapping for non-deleted points.
    // new_id is assigned in order, so new_id <= old_id always holds.
    std::vector<uint32_t> id_map(max_points(), kInvalidID);
    uint32_t new_nd = 0;
    for (uint32_t old_id = 0; old_id < max_points(); ++old_id) {
      if (_location_to_tag.find(old_id) != _location_to_tag.end()) {
        id_map[old_id] = new_nd++;
      }
    }
    LOG(INFO) << "After consolidation, new_nd: " << new_nd;

    // Save old ep vector for finding new ep later
    std::vector<T> ep_vector(_dim);
    memcpy(ep_vector.data(), _data.data() + _dim * _ep, _dim * sizeof(T));

    // Step 2: Update neighbor lists (parallel) then compact (sequential).
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t old_id = 0; old_id < (int64_t) max_points(); ++old_id) {
      if (id_map[old_id] == kInvalidID)
        continue;  // Skip deleted points

      // Update neighbors: replace deleted neighbors with their neighbors
      tsl::robin_set<uint32_t> new_nbrs_set;
      for (auto nbr : _final_graph[old_id]) {
        if (id_map[nbr] == kInvalidID) {
          // Neighbor is deleted, add its non-deleted neighbors
          for (auto nbr_of_nbr : _final_graph[nbr]) {
            if (id_map[nbr_of_nbr] != kInvalidID && nbr_of_nbr != (uint32_t) old_id) {
              new_nbrs_set.insert(nbr_of_nbr);
            }
          }
        } else {
          new_nbrs_set.insert(nbr);
        }
      }

      std::vector<uint32_t> new_nbrs(new_nbrs_set.begin(), new_nbrs_set.end());

      // Prune if too many neighbors (use old IDs since data is still at old positions)
      if (new_nbrs.size() > params.R) {
        std::vector<Neighbor> pool;
        pool.reserve(new_nbrs.size());
        for (auto nbr : new_nbrs) {
          float dist = _distance->compare(_data.data() + _dim * old_id, _data.data() + _dim * nbr, _dim);
          pool.emplace_back(nbr, dist, true);
        }
        pipeann::prune_neighbors(pool, new_nbrs, params, _dist_metric, [this](uint32_t a, uint32_t b) {
          return _distance->compare(_data.data() + _dim * a, _data.data() + _dim * b, _dim);
        });
      }

      // Remap neighbor IDs to new IDs
      for (auto &nbr : new_nbrs) {
        nbr = id_map[nbr];
      }

      _final_graph[old_id] = std::move(new_nbrs);
    }

    // Phase 2: Compact graph and data (sequential, since new_id <= old_id).
    for (uint32_t old_id = 0; old_id < max_points(); ++old_id) {
      uint32_t new_id = id_map[old_id];
      if (new_id == kInvalidID || new_id == old_id) {
        continue;
      }

      _final_graph[new_id] = std::move(_final_graph[old_id]);
      memcpy(_data.data() + _dim * new_id, _data.data() + _dim * old_id, _dim * sizeof(T));
    }

    // Clear graph entries beyond new_nd
    for (uint32_t i = new_nd; i < max_points(); ++i) {
      _final_graph[i].clear();
    }

    // Step 3: Update tag mappings and state.
    std::unordered_map<TagT, unsigned> new_tag_to_location;
    std::unordered_map<unsigned, TagT> new_location_to_tag;
    for (auto &[tag, old_loc] : _tag_to_location) {
      uint32_t new_loc = id_map[old_loc];
      if (new_loc != kInvalidID) {
        new_tag_to_location[tag] = new_loc;
        new_location_to_tag[new_loc] = tag;
      }
    }
    _tag_to_location = std::move(new_tag_to_location);
    _location_to_tag = std::move(new_location_to_tag);

    _nd = new_nd;
    _delete_set.clear();
    _empty_slots.clear();
    for (uint32_t i = _nd; i < max_points(); ++i) {
      _empty_slots.insert(i);
    }

    // Step 4: After everything finishes, find new entry point.
    float min_dist = std::numeric_limits<float>::max();
    this->search(ep_vector.data(), 1, 10, &_ep, &min_dist);

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Consolidation completed in "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  uint32_t Index<T, TagT>::reserve_location() {
    std::lock_guard<std::mutex> guard(_change_lock);
    if (_nd >= max_points()) {
      return kInvalidID;
    }

    assert(!_empty_slots.empty());
    assert(_empty_slots.size() + _nd == max_points());

    auto iter = _empty_slots.begin();
    unsigned location = *iter;
    _empty_slots.erase(iter);
    _delete_set.erase(location);

    ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::resize(size_t new_max_points) {
    new_max_points = std::max(new_max_points, 50000ul);  // at least 50000 points.
    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() == 0);  // should not resize if there are empty slots.

    LOG(INFO) << "Resize from " << max_points() << " to " << new_max_points;
    _data.resize(new_max_points * _dim);
    _final_graph.resize(new_max_points);

    for (size_t i = _nd; i < max_points(); i++) {
      _empty_slots.insert(i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const IndexBuildParameters &params, const TagT tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    // If tag already exists, mark old location for deletion
    {
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      if (_tag_to_location.find(tag) != _tag_to_location.end()) {
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    auto location = reserve_location();
    while (location == kInvalidID) {
      lock.unlock();
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);
      if (_nd >= max_points()) {
        auto new_max_points = (size_t) (max_points() * INDEX_GROWTH_FACTOR);
        resize(new_max_points);
      }
      growth_lock.unlock();
      lock.lock();
      location = reserve_location();
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data.data() + _dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _dim);
    if (_dist_metric == pipeann::Metric::COSINE) {
      pipeann::normalize_data(offset_data, point, _dim);
    } else {
      memcpy((void *) offset_data, point, sizeof(T) * _dim);
    }

    std::vector<Neighbor> pool;
    get_expanded_nodes(location, params.L, pool);
    // remove itself from pool.
    pool.erase(std::remove_if(pool.begin(), pool.end(), [location](const Neighbor &n) { return n.id == location; }),
               pool.end());

    std::vector<unsigned> pruned_list;
    pipeann::prune_neighbors(pool, pruned_list, params, _dist_metric, [this](uint32_t a, uint32_t b) {
      return _distance->compare(_data.data() + _dim * a, _data.data() + _dim * b, _dim);
    });
    assert(_final_graph.size() == max_points());

    _final_graph[location].clear();
    _final_graph[location].reserve((uint64_t) (params.R * SLACK_FACTOR * 1.05));

    if (pruned_list.empty()) {
      LOG(INFO) << "Thread: " << std::this_thread::get_id() << " Tag: " << tag
                << " pruned_list.size(): " << pruned_list.size();
    }

    assert(!pruned_list.empty());
    {
      // pipeann::SparseWriteLockGuard<uint64_t> guard(&_locks, location);
      pipeann::LockGuard guard(_locks->wrlock(location));
      for (auto link : pruned_list) {
        _final_graph[location].emplace_back(link);
      }
    }

    assert(_final_graph[location].size() <= params.R);
    inter_insert(location, pruned_list, params);
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);

    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      return -1;
    }
    assert(_tag_to_location[tag] < max_points());

    _delete_set.insert(_tag_to_location[tag]);
    _location_to_tag.erase(_tag_to_location[tag]);
    _tag_to_location.erase(tag);

    return 0;
  }

  /*  Internals of the library */
  // EXPORTS
  template class Index<float, uint32_t>;
  template class Index<int8_t, uint32_t>;
  template class Index<uint8_t, uint32_t>;
}  // namespace pipeann
