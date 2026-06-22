#include <algorithm>
#include <bitset>
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
  Index<T, TagT>::Index(Metric m, const size_t dim, uint64_t max_points) : _dist_metric(m), _dim(dim) {
    constexpr uint64_t kLockTableEntries = 131072;  // ~1MB lock table.
    this->_locks = new pipeann::LockTable(kLockTableEntries);
    LOG(INFO) << "Getting distance function for metric: " << get_metric_str(m);
    this->_distance = get_distance_function<T>(m);
    range = 0;
    if (max_points > 0) {
      resize(max_points);
    }
  }

  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, const SSDIndexMetadata<T> &meta) : Index(m, meta.data_dim) {
    _ep = meta.entry_point;
    range = meta.range;
    R_ood = meta.R_ood;
    R_base = range - R_ood;
    _nd = meta.npoints;
    resize(meta.npoints);
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    delete this->_distance;
    delete this->_locks;
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
        ++hops;
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
                             const std::vector<TagT> &tags, bool normalize_cosine, const std::string &train_query_path,
                             uint32_t R_ood, uint32_t L_ood) {
    if (filename == nullptr || !file_exists(filename)) {
      LOG(ERROR) << "Data file " << filename << " does not exist. Exiting....";
      return;
    }

    LOG(INFO) << "Building index with normalize_cosine: " << normalize_cosine;

    size_t file_num_points, file_dim;
    pipeann::load_bin<T>(filename, _data, file_num_points, file_dim);

    if (num_points_to_load > file_num_points) {
      LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points but file has "
                 << file_num_points << " points.";
      crash();
    }
    if (file_dim != _dim) {
      LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension,"
                 << "but file has " << file_dim << " dimension.";
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

    R_ood = std::min(R_ood, params.R);
    if (train_query_path.empty()) {
      R_ood = 0;
    }

    this->R_base = params.R - R_ood;
    this->R_ood = R_ood;

    // Base build at reduced degree so refine gets R_ood slots.
    IndexBuildParameters base_params = params;
    base_params.R = R_base;

    if (params.use_pipnn) {
      pipnn_link(base_params);
    } else {
      link(base_params);
    }

    if (R_ood > 0) {
      ngfix_refine(train_query_path, L_ood);
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
    range = std::max(range, (uint16_t) max_deg);

    // Initialize empty slots for future insertions
    for (uint32_t i = _nd; i < max_points(); i++) {
      _empty_slots.insert(i);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename, const size_t num_points_to_load, IndexBuildParameters &params,
                             const char *tag_filename, bool normalize_cosine, const std::string &train_query_path,
                             uint32_t R_ood, uint32_t L_ood) {
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

    build(filename, num_points_to_load, params, tags, normalize_cosine, train_query_path, R_ood, L_ood);
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const size_t K, const unsigned L,
                                                       unsigned *indices, float *distances, QueryStats *stats,
                                                       const std::vector<uint32_t> *cand_ids) {
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
    std::pair<uint32_t, uint32_t> retval{0, 0};

    if (cand_ids != nullptr) {
      best_L_nodes.reserve(cand_ids->size());
      for (auto loc : *cand_ids) {
        best_L_nodes.emplace_back(loc, _distance->compare(aligned_query, _data.data() + (size_t) loc * _dim, _dim),
                                  true);
      }
      std::sort(best_L_nodes.begin(), best_L_nodes.end());
      retval.second = (uint32_t) best_L_nodes.size();
    } else {
      retval = iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids,
                                      best_L_nodes, stats);
    }

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
    for (size_t i = pos; i < K; i++) {
      indices[i] = std::numeric_limits<unsigned>::max();
      if (distances != nullptr) {
        distances[i] = std::numeric_limits<float>::max();
      }
    }

    if (stats != nullptr) {
      stats->n_hops = retval.first;
      stats->n_cmps = retval.second;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const size_t K, const unsigned L, TagT *tags,
                                          float *distances, const std::vector<uint32_t> *cand_ids) {
    uint32_t *indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim, nullptr, cand_ids);

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
  void Index<T, TagT>::save_tags(const std::string &tags_file) const {
    std::remove(tags_file.c_str());
    if (_location_to_tag.empty()) {
      return;
    }
    std::vector<TagT> tag_data(_nd);
    for (uint64_t i = 0; i < _nd; i++) {
      auto it = _location_to_tag.find((unsigned) i);
      if (it != _location_to_tag.end()) {
        tag_data[i] = it->second;
      }
    }
    pipeann::save_bin<TagT>(tags_file, tag_data.data(), _nd, 1);
    LOG(INFO) << "Memory index tags saved to " << tags_file;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::load_tags(const std::string &tags_file) {
    _location_to_tag.clear();
    _tag_to_location.clear();
    if (file_exists(tags_file)) {
      size_t tag_num = 0, tag_dim = 0;
      std::vector<TagT> tag_v;
      pipeann::load_bin<TagT>(tags_file, tag_v, tag_num, tag_dim);
      for (size_t i = 0; i < tag_num; i++) {
        _location_to_tag[(unsigned) i] = tag_v[i];
        _tag_to_location[tag_v[i]] = (unsigned) i;
      }
    } else {
      for (uint64_t i = 0; i < _nd; i++) {
        _location_to_tag[(unsigned) i] = (TagT) i;
        _tag_to_location[(TagT) i] = (unsigned) i;
      }
    }
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const IndexBuildParameters &params, const TagT tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    // Dynamic in-memory inserts do not go through build(), so keep the
    // persisted max out-degree in sync with the insertion configuration.
    range = std::max<uint16_t>(range, static_cast<uint16_t>(params.R));

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
    return static_cast<int>(location);
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

  // ---- NGFix refine ------------------------------------------------------
  // Query-driven OOD edge augmentation. See SIGMOD'26 NGFix paper.
  // Constants mirror ngfixlib/graph/hnsw_ngfix.h.
  namespace {
    static constexpr size_t NGFIX_MAX_NQ = 200;
    static constexpr size_t NGFIX_MAX_S = 200;
    static constexpr uint16_t NGFIX_EH_INF = std::numeric_limits<uint16_t>::max();
    static constexpr float NGFIX_INF_RATIO = 0.5f;
  }  // namespace

  template<typename T, typename TagT>
  void Index<T, TagT>::add_ngfix_neighbor(unsigned u, unsigned v, uint16_t eh) {
    if (u == v)
      return;
    pipeann::LockGuard guard(_locks->wrlock(u));
    auto &pool = _final_graph[u];
    // dedupe against all existing (base + refine) edges
    for (auto x : pool) {
      if (x == v)
        return;
    }
    uint16_t cnt = _ngfix_count[u];
    if (cnt < R_ood) {
      pool.push_back(v);
      _ngfix_ehs[u][cnt] = eh;
      _ngfix_count[u] = cnt + 1;
      return;
    }
    // refine region is full → EH-based eviction
    uint32_t inf_cnt = 0, min_idx = 0;
    uint16_t min_eh = NGFIX_EH_INF;
    for (uint32_t i = 0; i < R_ood; i++) {
      uint16_t e = _ngfix_ehs[u][i];
      if (e < min_eh) {
        min_eh = e;
        min_idx = i;
      }
      if (e == NGFIX_EH_INF)
        inf_cnt++;
    }
    if (eh == NGFIX_EH_INF && inf_cnt >= (uint32_t) (NGFIX_INF_RATIO * R_ood))
      return;
    if (eh <= min_eh)
      return;
    _ngfix_ehs[u][min_idx] = eh;
    pool[R_base + min_idx] = v;
  }

  // CalculateHardness: Floyd-Warshall over induced subgraph of top-S gt.
  // H[i][j] = smallest h such that gt[i] reaches gt[j] via gt[0..h]; EH_INF if unreachable.
  template<typename T, typename TagT>
  void Index<T, TagT>::ngfix_calc_hardness(const unsigned *gt, size_t Nq, size_t S,
                                           std::vector<std::vector<uint16_t>> &H) {
    H.assign(Nq, std::vector<uint16_t>(Nq, NGFIX_EH_INF));
    std::unordered_map<unsigned, uint16_t> p2rank;
    p2rank.reserve(S);
    for (size_t i = 0; i < S; i++)
      p2rank[gt[i]] = (uint16_t) i;

    std::vector<std::bitset<NGFIX_MAX_S>> f(S);
    for (size_t h = 0; h < S; h++) {
      f[h][h] = 1;
      if (h < Nq)
        H[h][h] = (uint16_t) h;
    }

    // Induced subgraph on gt[0..S): iterate each u = gt[i] once, take its current neighbors.
    for (size_t i = 0; i < S; i++) {
      unsigned u = gt[i];
      std::vector<unsigned> nbrs;
      {
        pipeann::LockGuard guard(_locks->rdlock(u));
        nbrs = _final_graph[u];
      }
      for (unsigned v : nbrs) {
        auto it = p2rank.find(v);
        if (it == p2rank.end())
          continue;
        size_t j = it->second;
        f[i][j] = 1;
        if (i < Nq && j < Nq)
          H[i][j] = (uint16_t) std::max(i, j);
      }
    }

    // Floyd-Warshall with bitset
    for (size_t h = 0; h < S; h++) {
      for (size_t i = 0; i < S; i++) {
        if (!f[i][h])
          continue;
        auto last = f[i];
        f[i] |= f[h];
        last ^= f[i];
        if (i < Nq && last.any()) {
          for (size_t j = 0; j < Nq; j++) {
            if (last[j])
              H[i][j] = (uint16_t) std::max(std::max(i, j), h);
          }
        }
      }
    }
  }

  // getDefectsFixingEdges: sort unreachable (i,j) pairs by dist, greedily add minimum set.
  template<typename T, typename TagT>
  void Index<T, TagT>::ngfix_pass(const T *query, const unsigned *gt, size_t Nq, size_t Kh) {
    if (Nq > NGFIX_MAX_NQ)
      return;
    std::vector<std::vector<uint16_t>> H;
    size_t S = std::min(NGFIX_MAX_S, 2 * Nq);
    ngfix_calc_hardness(gt, Nq, S, H);

    std::vector<std::bitset<NGFIX_MAX_NQ>> f(Nq);
    for (size_t i = 0; i < Nq; i++)
      for (size_t j = 0; j < Nq; j++)
        if (H[i][j] <= Kh)
          f[i][j] = 1;

    // Collect candidate edges (i,j) where f[i][j] == 0, sorted by distance.
    struct Cand {
      float d;
      uint16_t i, j;
    };
    std::vector<Cand> vs;
    vs.reserve(Nq * Nq);
    for (size_t i = 0; i < Nq; i++) {
      for (size_t j = 0; j < Nq; j++) {
        if (f[i][j])
          continue;
        float d = _distance->compare(_data.data() + _dim * (size_t) gt[i], _data.data() + _dim * (size_t) gt[j], _dim);
        vs.push_back({d, (uint16_t) i, (uint16_t) j});
      }
    }
    std::sort(vs.begin(), vs.end(), [](const Cand &a, const Cand &b) { return a.d < b.d; });

    std::unordered_map<unsigned, std::vector<std::pair<unsigned, uint16_t>>> new_edges;
    for (const auto &c : vs) {
      if (f[c.i][c.j])
        continue;
      unsigned u = gt[c.i];
      unsigned v = gt[c.j];
      new_edges[u].push_back({v, H[c.i][c.j]});
      f[c.i][c.j] = 1;
      for (size_t k = 0; k < Nq; k++) {
        if (f[k][c.i])
          f[k] |= f[c.j];
      }
    }

    for (auto &kv : new_edges) {
      for (auto &ve : kv.second)
        add_ngfix_neighbor(kv.first, ve.first, ve.second);
    }
  }

  // RFix: single round. If ANN can't reach gt[Nq-1], add heuristic-pruned closer points as refine edges.
  template<typename T, typename TagT>
  void Index<T, TagT>::rfix_pass(const T *query, const unsigned *gt, size_t Nq, uint32_t L_ood) {
    std::vector<unsigned> ids(L_ood);
    std::vector<float> dists(L_ood);
    search(query, 1, Nq, ids.data(), dists.data());
    float d2 = _distance->compare(_data.data() + _dim * (size_t) gt[Nq - 1], query, _dim);
    if (dists[0] <= d2)
      return;

    unsigned ANN = ids[0];
    search(query, L_ood, L_ood, ids.data(), dists.data());
    std::vector<unsigned> res;
    for (size_t i = 0; i < L_ood; i++) {
      if (dists[i] < d2)
        res.push_back(ids[i]);
    }

    std::vector<Neighbor> pool;
    pool.reserve(res.size());
    for (unsigned v : res) {
      float d = _distance->compare(_data.data() + _dim * (size_t) ANN, _data.data() + _dim * (size_t) v, _dim);
      pool.emplace_back(Neighbor(v, d, true));
    }

    IndexBuildParameters rfix_params;
    rfix_params.R = 6;  // default in NGFix.
    rfix_params.C = pool.size();
    rfix_params.alpha = 1.0f;
    rfix_params.saturate_graph = false;

    std::vector<unsigned> picked;
    pipeann::prune_neighbors(pool, picked, rfix_params, _dist_metric, [this](uint32_t a, uint32_t b) {
      return _distance->compare(_data.data() + _dim * a, _data.data() + _dim * b, _dim);
    });

    // RFix edges use EH=MAX_S+1 (just above ordinary hardness range, below EH_INF).
    const uint16_t rfix_eh = (uint16_t) (NGFIX_MAX_S + 1);
    for (unsigned p : picked)
      add_ngfix_neighbor(ANN, p, rfix_eh);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::ngfix_refine(const std::string &train_query_path, uint32_t L_ood) {
    // Too small L_ood won't find enough neighbors for refine.
    L_ood = std::max<uint32_t>(L_ood, NGFIX_MAX_S);
    
    // Load training queries.
    std::vector<T> train_data;
    size_t train_n = 0, train_dim = 0;
    pipeann::load_bin<T>(train_query_path, train_data, train_n, train_dim);
    if (train_dim != _dim) {
      LOG(ERROR) << "Train query dim " << train_dim << " != index dim " << _dim;
      crash();
    }
    LOG(INFO) << "NGFix refine: " << train_n << " train queries, R_base=" << R_base << " R_ood=" << R_ood
              << " L_ood=" << L_ood;

    // Allocate sidecars.
    _ngfix_count.assign(_nd, 0);
    _ngfix_ehs.assign(_nd, std::vector<uint16_t>(R_ood, 0));

    pipeann::Timer refine_timer;
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < train_n; i++) {
      const T *q = train_data.data() + i * _dim;
      // AKNN ground truth (top-500 via beam L_ood, NGFix default)
      std::vector<unsigned> ids(std::min<size_t>(500, L_ood));
      search(q, ids.size(), L_ood, ids.data());
      // Coarse + fine NGFix passes, then RFix.
      // use this->R_base and this->R_ood.
      ngfix_pass(q, ids.data(), /*Nq=*/100, /*Kh=*/100);
      ngfix_pass(q, ids.data(), /*Nq=*/10, /*Kh=*/10);
      rfix_pass(q, ids.data(), /*Nq=*/10, L_ood);

      if (i % 100000 == 0) {
        LOG(INFO) << "NGFix refine: processed " << i << "/" << train_n;
      }
    }
    LOG(INFO) << "NGFix refine done in " << (refine_timer.elapsed() / 1e6) << "s";
    // Sidecars no longer needed.
    _ngfix_count.clear();
    _ngfix_count.shrink_to_fit();
    _ngfix_ehs.clear();
    _ngfix_ehs.shrink_to_fit();
  }

  /*  Internals of the library */
  // EXPORTS
  template class Index<float, uint32_t>;
  template class Index<int8_t, uint32_t>;
  template class Index<uint8_t, uint32_t>;
}  // namespace pipeann
