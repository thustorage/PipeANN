#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <filesystem>
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
#include "candidate_pool.h"
#include "utils/lock_table.h"
#include "utils/prune_neighbors.h"

namespace pipeann {

  template<typename T, typename TagT>
  Index<T, TagT>::Index(std::unique_ptr<AbstractNeighbor<T>> vectors, uint64_t max_points)
      : _vectors(std::move(vectors)) {
    if (!_vectors)
      throw std::invalid_argument("Index requires a Neighbor");
    _dist_metric = _vectors->metric;
    _dim = _vectors->dim();
    constexpr uint64_t kLockTableEntries = 131072;  // ~1MB lock table.
    this->_locks = new pipeann::LockTable(kLockTableEntries);
    LOG(INFO) << "Getting distance function for metric: " << get_metric_str(_dist_metric);
    _nd = _vectors->npoints;
    _ep = 0;
    if (max_points > this->max_points()) {
      resize(max_points);
    } else {
      _final_graph.resize(this->max_points());
    }
  }

  template<typename T, typename TagT>
  Index<T, TagT>::Index(const SSDIndexMetadata<T> &meta, std::unique_ptr<AbstractNeighbor<T>> vectors)
      : Index(std::move(vectors)) {
    _ep = meta.entry_point;
    range = meta.range;
    R_ood = meta.R_ood;
    R_base = range - R_ood;
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    delete this->_locks;
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
    // Fixed seed keeps graph builds reproducible without scanning the vectors.
    std::mt19937 rng(42);
    return std::uniform_int_distribution<unsigned>(0, _nd - 1)(rng);
  }

  /* iterate_to_fixed_point():
   * query : RAW vector or RaBitQ8 code, prepared by the caller.
   * ep : entry point of the search.
   * Lsize : size of list.
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(const uint8_t *query, const unsigned Lsize,
                                                                       const unsigned ep,
                                                                       std::vector<Neighbor> &expanded_nodes_info,
                                                                       std::vector<Neighbor> &best_L_nodes,
                                                                       QueryStats *stats) {
    expanded_nodes_info.reserve(10 * Lsize);
    // Search/insert callers prevent storage growth; builds allocate up front.
    const uint8_t *vector_data = _vectors->data();
    const size_t vector_bytes = _vectors->data_size;

    // Reuse one table per worker; clear retains its allocation between traversals.
    FlatVisitedSet inserted_into_pool;

    CandidatePool pool(Lsize, Lsize + 1);
    inserted_into_pool.insert(ep);
    float entry_distance;
    _vectors->compute_dists_sym(query, &ep, 1, &entry_distance);
    pool.insert(Neighbor(ep, entry_distance));

    Timer query_timer, io_timer, cpu_timer;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    std::vector<unsigned> des;
    std::vector<float> distances;
    Neighbor *cur;
    while ((cur = pool.next_unvisited()) != nullptr) {
      ++hops;
      if (stats != nullptr)
        io_timer.reset();
      auto n = cur->id;
      expanded_nodes_info.emplace_back(*cur);
      des.clear();

      {
        // pipeann::SparseReadLockGuard<uint64_t> guard(&_locks, n);
        pipeann::LockGuard guard(_locks->rdlock(n));
        for (unsigned m = 0; m < _final_graph[n].size(); m++) {
          if (_final_graph[n][m] >= max_points()) {
            LOG(ERROR) << "Wrong id found: " << _final_graph[n][m];
            crash();
          }
          const auto id = _final_graph[n][m];
          if (inserted_into_pool.insert(id)) {
            des.emplace_back(id);
            prefetch_vector(reinterpret_cast<const char *>(vector_data + size_t(id) * vector_bytes), vector_bytes);
          }
        }
      }
      if (stats != nullptr) {
        stats->io_us += io_timer.elapsed();  // collect unseen neighbors
      }

      if (stats != nullptr)
        cpu_timer.reset();

      distances.resize(des.size());
      _vectors->compute_dists_sym(query, des.data(), des.size(), distances.data());
      cmps += des.size();
      // The visited set guarantees unique IDs within and across batches.
      pool.insert_batch(des.data(), distances.data(), des.size());
      if (stats != nullptr) {
        stats->cpu_us += cpu_timer.elapsed();  // compute + read nbr
      }
    }

    best_L_nodes.resize(std::min<size_t>(Lsize, pool.size()));
    for (size_t i = 0; i < best_L_nodes.size(); ++i) {
      best_L_nodes[i] = pool[i];
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(const size_t node_id, const unsigned Lindex,
                                          std::vector<Neighbor> &expanded_nodes_info) {
    std::vector<Neighbor> best_L_nodes;
    // Build has fixed storage; insert_point holds _update_lock against resize.
    const uint8_t *node_coords = _vectors->at(node_id);
    iterate_to_fixed_point(node_coords, Lindex, _ep, expanded_nodes_info, best_L_nodes);
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

        std::vector<float> distances(copy_of_neighbors.size());
        _vectors->compute_dists(des, copy_of_neighbors.data(), copy_of_neighbors.size(), distances.data());
        for (size_t i = 0; i < copy_of_neighbors.size(); ++i) {
          if (copy_of_neighbors[i] != des)
            pool.emplace_back(copy_of_neighbors[i], distances[i]);
        }
        std::vector<unsigned> new_out_neighbors;
        pipeann::prune_neighbors(
            pool, new_out_neighbors, params, _dist_metric,
            [this](uint32_t a, const uint32_t *ids, size_t n, float *out) { _vectors->compute_dists(a, ids, n, out); });
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
      pipeann::prune_neighbors(
          pool, pruned_list, params, _dist_metric,
          [this](uint32_t a, const uint32_t *ids, size_t n, float *out) { _vectors->compute_dists(a, ids, n, out); });

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

        const auto &neighbors = _final_graph[node];
        pool.reserve(neighbors.size());
        std::vector<float> distances(neighbors.size());
        _vectors->compute_dists(node, neighbors.data(), neighbors.size(), distances.data());
        for (size_t i = 0; i < neighbors.size(); ++i) {
          if (neighbors[i] != node)
            pool.emplace_back(neighbors[i], distances[i]);
        }
        pipeann::prune_neighbors(
            pool, new_out_neighbors, params, _dist_metric,
            [this](uint32_t a, const uint32_t *ids, size_t n, float *out) { _vectors->compute_dists(a, ids, n, out); });

        _final_graph[node].clear();
        for (auto id : new_out_neighbors)
          _final_graph[node].emplace_back(id);
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(IndexBuildParameters &params, const std::vector<TagT> &tags,
                             const std::string &train_query_path, uint32_t R_ood, uint32_t L_ood) {
    if (params.num_threads)
      omp_set_num_threads(params.num_threads);
    LOG(INFO) << "Connecting " << _nd << " points.";
    if (tags.size() != _nd)
      throw std::invalid_argument("Tag count does not match vector file point count");
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
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(IndexBuildParameters &params, const char *tag_filename,
                             const std::string &train_query_path, uint32_t R_ood, uint32_t L_ood) {
    std::vector<TagT> tags{};

    if (tag_filename == nullptr) {
      tags.resize(_nd);
      std::iota(tags.begin(), tags.end(), TagT{0});
    } else {
      if (!file_exists(tag_filename)) {
        LOG(ERROR) << "Tag file " << tag_filename << " does not exist. Exiting...";
        crash();
      }
      LOG(INFO) << "Loading tags from " << tag_filename << " for vamana index build";
      size_t npts, ndim;
      pipeann::load_bin(tag_filename, tags, npts, ndim);
      if (ndim != 1)
        throw std::invalid_argument("Tag file must have one column");
    }

    build(params, tags, train_query_path, R_ood, L_ood);
  }

  template<typename T, typename TagT>
  std::vector<uint8_t> Index<T, TagT>::prepare_record(const T *query) {
    std::vector<uint8_t> record(_vectors->data_size);
    if (_dist_metric == COSINE) {
      std::vector<T> normalized(_dim);
      normalize_data(normalized.data(), query, _dim);
      _vectors->encode(normalized.data(), record.data());
    } else {
      _vectors->encode(query, record.data());
    }
    return record;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const size_t K, const unsigned L,
                                                       unsigned *indices, float *distances, QueryStats *stats,
                                                       const std::vector<uint32_t> *cand_ids) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    if (K == 0 || L == 0) return {0, 0};
    const auto prepared = prepare_record(query);
    std::vector<Neighbor> results;
    const auto work = search_prepared(prepared.data(), L, results, stats, cand_ids);
    const size_t count = std::min(K, results.size());
    for (size_t i = 0; i < count; ++i) {
      indices[i] = results[i].id;
      if (distances) distances[i] = results[i].distance;
    }
    std::fill(indices + count, indices + K, kInvalidID);
    if (distances)
      std::fill(distances + count, distances + K, std::numeric_limits<float>::max());
    return work;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search_prepared(const uint8_t *prepared_query, unsigned L,
                                                            std::vector<Neighbor> &results, QueryStats *stats,
                                                            const std::vector<uint32_t> *cand_ids) {
    results.clear();
    std::pair<uint32_t, uint32_t> work{0, 0};
    if (cand_ids != nullptr) {
      results.reserve(cand_ids->size());
      std::vector<float> distances(cand_ids->size());
      _vectors->compute_dists_sym(prepared_query, cand_ids->data(), cand_ids->size(), distances.data());
      for (size_t i = 0; i < cand_ids->size(); ++i)
        results.emplace_back((*cand_ids)[i], distances[i]);
      std::sort(results.begin(), results.end());
      work.second = static_cast<uint32_t>(results.size());
    } else {
      std::vector<Neighbor> expanded;
      work = iterate_to_fixed_point(prepared_query, L, _ep, expanded, results, stats);
    }
    if (stats) {
      stats->n_hops = work.first;
      stats->n_cmps = work.second;
    }
    return work;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const size_t K, const unsigned L, TagT *tags,
                                        float *distances, const std::vector<uint32_t> *cand_ids) {
    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    if (K == 0 || L == 0) return 0;
    const auto prepared = prepare_record(query);
    std::vector<Neighbor> results;
    search_prepared(prepared.data(), L, results, nullptr, cand_ids);

    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t count = 0;
    for (const auto &result : results) {
      auto it = _location_to_tag.find(result.id);
      if (it == _location_to_tag.end()) continue;
      tags[count] = it->second;
      if (distances) distances[count] = result.distance;
      if (++count == K) break;
    }
    return count;
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
        std::vector<float> distances(new_nbrs.size());
        _vectors->compute_dists(old_id, new_nbrs.data(), new_nbrs.size(), distances.data());
        for (size_t i = 0; i < new_nbrs.size(); ++i)
          pool.emplace_back(new_nbrs[i], distances[i]);
        pipeann::prune_neighbors(
            pool, new_nbrs, params, _dist_metric,
            [this](uint32_t a, const uint32_t *ids, size_t n, float *out) { _vectors->compute_dists(a, ids, n, out); });
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
      memmove(_vectors->at(new_id), _vectors->at(old_id), _vectors->data_size);
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
    _vectors->npoints = _nd;
    _delete_set.clear();

    // Preserve a surviving entry point; otherwise select a remaining node.
    _ep = id_map[_ep] != kInvalidID ? id_map[_ep] : calculate_entry_point();

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Consolidation completed in "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";
  }

  template<typename T, typename TagT>
  uint32_t Index<T, TagT>::reserve_location() {
    std::lock_guard<std::mutex> guard(_change_lock);
    if (_nd >= max_points()) {
      return kInvalidID;
    }

    const uint32_t location = static_cast<uint32_t>(_nd);
    _vectors->npoints = ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::resize(size_t new_max_points) {
    new_max_points = std::max(new_max_points, 50000ul);  // at least 50000 points.
    auto start = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "Resize from " << max_points() << " to " << new_max_points;
    _vectors->reserve(new_max_points);
    _final_graph.resize(new_max_points);

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

    const auto record = prepare_record(point);
    memcpy(_vectors->at(location), record.data(), _vectors->data_size);

    std::vector<Neighbor> pool;
    get_expanded_nodes(location, params.L, pool);
    // remove itself from pool.
    pool.erase(std::remove_if(pool.begin(), pool.end(), [location](const Neighbor &n) { return n.id == location; }),
               pool.end());

    std::vector<unsigned> pruned_list;
    pipeann::prune_neighbors(
        pool, pruned_list, params, _dist_metric,
        [this](uint32_t a, const uint32_t *ids, size_t n, float *out) { _vectors->compute_dists(a, ids, n, out); });
    assert(_final_graph.size() == max_points());

    _final_graph[location].clear();
    _final_graph[location].reserve((uint64_t) (params.R * SLACK_FACTOR * 1.05));

    if (pruned_list.empty()) {
      LOG(INFO) << "Thread: " << std::this_thread::get_id() << " Tag: " << tag
                << " pruned_list.size(): " << pruned_list.size();
    }

    assert(_nd == 1 || !pruned_list.empty());
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
  void Index<T, TagT>::ngfix_pass(const unsigned *gt, size_t Nq, size_t Kh) {
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
    for (size_t i = 0; i < Nq; ++i) {
      for (size_t j = 0; j < Nq; ++j) {
        if (f[i][j]) continue;
        float distance;
        _vectors->compute_dists(gt[i], gt + j, 1, &distance);
        vs.push_back({distance, static_cast<uint16_t>(i), static_cast<uint16_t>(j)});
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
  void Index<T, TagT>::rfix_pass(const uint8_t *prepared_query, const unsigned *gt, size_t Nq, uint32_t L_ood) {
    std::vector<Neighbor> results;
    search_prepared(prepared_query, Nq, results);
    if (results.empty()) return;
    float d2;
    _vectors->compute_dists_sym(prepared_query, gt + Nq - 1, 1, &d2);
    if (results[0].distance <= d2) return;

    const unsigned ANN = results[0].id;
    search_prepared(prepared_query, L_ood, results);
    std::vector<unsigned> res;
    for (const auto &result : results) {
      if (result.distance < d2) res.push_back(result.id);
    }

    std::vector<Neighbor> pool;
    pool.reserve(res.size());
    std::vector<float> distances(res.size());
    _vectors->compute_dists(ANN, res.data(), res.size(), distances.data());
    for (size_t i = 0; i < res.size(); ++i)
      pool.emplace_back(res[i], distances[i]);

    IndexBuildParameters rfix_params;
    rfix_params.R = 6;  // default in NGFix.
    rfix_params.C = pool.size();
    rfix_params.alpha = 1.0f;
    rfix_params.saturate_graph = false;

    std::vector<unsigned> picked;
    pipeann::prune_neighbors(
        pool, picked, rfix_params, _dist_metric,
        [this](uint32_t a, const uint32_t *ids, size_t n, float *out) { _vectors->compute_dists(a, ids, n, out); });

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
      std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
      const auto query = prepare_record(train_data.data() + i * _dim);
      // AKNN ground truth (top-500 via beam L_ood, NGFix default)
      std::vector<Neighbor> results;
      search_prepared(query.data(), L_ood, results);
      std::vector<unsigned> ids;
      for (size_t j = 0; j < std::min<size_t>(500, results.size()); ++j)
        ids.push_back(results[j].id);
      // Hardness needs up to 2 * Nq neighbors; small graphs may return fewer.
      const size_t coarse_nq = std::min<size_t>(100, ids.size() / 2);
      const size_t fine_nq = std::min<size_t>(10, ids.size() / 2);
      ngfix_pass(ids.data(), coarse_nq, coarse_nq);
      ngfix_pass(ids.data(), fine_nq, fine_nq);
      if (!ids.empty()) rfix_pass(query.data(), ids.data(), std::min<size_t>(10, ids.size()), L_ood);

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
