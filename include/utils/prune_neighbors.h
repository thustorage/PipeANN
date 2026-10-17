#ifndef PRUNE_NEIGHBORS_H_
#define PRUNE_NEIGHBORS_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>
#include "utils.h"
#include "utils/tsl/robin_set.h"

namespace pipeann {
  // Use alpha-RNG triangle inequality to prune neighbors.
  // Assume that dij (node -> nbr1) < dik (node -> nbr2).
  // Vamana: check if dik (node -> nbr2) / djk (nbr1 -> nbr2) > alpha.
  // If so, we can prune nbr2.
  inline float get_occlude_factor(Metric metric, float dik, float djk) {
    if (metric == Metric::L2 || metric == Metric::COSINE) {
      // The two distances are always non-negative.
      return (djk == 0) ? std::numeric_limits<float>::max() : dik / djk;
    } else if (metric == Metric::INNER_PRODUCT) {
      // The two distances' signs maybe different.
      if (dik > djk) {  // dik is the longest edge, pruning may be needed.
        if (djk > 0) {  // both positive.
          return dik / djk;
        } else if (dik < 0) {  // both negative. reverse.
          return djk / dik;
        } else {  // positive - negative, return a value that can prune.
          return std::numeric_limits<float>::max();
        }
      }
    }
    return 0.0f;
  }

  // ============================================================================
  // Unified prune_neighbors implementation
  // ============================================================================
  // pool: candidates to the query point (unsorted, length may larger than maxc).
  // pruned_list: output, IDs of the pruned neighbors.
  // params: build parameters (R, C, alpha, saturate_graph).
  // metric: distance metric.
  // compute_distances: one-to-many callback: void(uint32_t, const uint32_t *, size_t, float *)
  template<typename DistBatchFunc>
  inline void prune_neighbors(std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list,
                              const IndexBuildParameters &params, Metric metric, DistBatchFunc &&compute_distances) {
    if (pool.empty())
      return;

    uint32_t range = params.R;
    uint32_t maxc = params.C;
    float alpha = params.alpha;
    bool saturate_graph = params.saturate_graph;

    // Sort and truncate pool.
    std::sort(pool.begin(), pool.end());
    if (pool.size() > maxc) {
      pool.resize(maxc);
    }

    std::set<Neighbor> result_set;  // deduplication and keep distance sorted.
    std::vector<float> occlude_factor(pool.size(), 0);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result_set.size() < range) {
      uint32_t start = 0;
      while (result_set.size() < range && start < pool.size()) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result_set.insert(p);
        // Dynamic programming: if p (current) is included,
        // then D(t, p0) / D(t, p) should be updated.
        std::vector<uint32_t> ids, positions;
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; ++t) {
          if (occlude_factor[t] <= alpha) {
            ids.push_back(pool[t].id);
            positions.push_back(t);
          }
        }
        std::vector<float> distances(ids.size());
        if (!ids.empty()) compute_distances(p.id, ids.data(), ids.size(), distances.data());
        for (size_t i = 0; i < ids.size(); ++i) {
          auto t = positions[i];
          occlude_factor[t] = std::max(occlude_factor[t], get_occlude_factor(metric, pool[t].distance, distances[i]));
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }

    pruned_list.clear();
    pruned_list.reserve(range);
    for (auto &x : result_set) {
      pruned_list.emplace_back(x.id);
    }

    // Saturate graph: fill up to range with nearest unselected points.
    if (saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end()) {
          pruned_list.emplace_back(pool[i].id);
        }
      }
    }
  }

  // ============================================================================
  // Unified delta_prune_neighbors implementation
  // ============================================================================
  // Delta prune neighbors: when inserting a new point into a full neighbor list.
  // nhood: current neighbors + target_id (size = R + 1), will be modified in place to output result (size = R).
  // NOTE: The last element of nhood should be target_id!
  // center_id: the center node whose neighbors are being pruned.
  // target_id: ID of the newly inserted point.
  // params: build parameters (R, alpha).
  // metric: distance metric.
  // compute_distances: callable to compute distances from center_id to multiple IDs.
  //   void(uint32_t center_id, const uint32_t *ids, uint32_t n, float *dists_out)
  template<typename DistBatchFunc>
  inline void delta_prune_neighbors(std::vector<uint32_t> &nhood, uint32_t center_id, uint32_t target_id,
                                    const IndexBuildParameters &params, Metric metric,
                                    DistBatchFunc &&compute_distances) {
    struct TriangleNeighbor {
      uint32_t id;
      float tgt_dis;   // distance to target
      float distance;  // distance to center
      inline bool operator<(const TriangleNeighbor &other) const {
        return (distance < other.distance) || (distance == other.distance && id < other.id);
      }
    };

    uint32_t range = params.R;
    float alpha = params.alpha;

    if (unlikely(nhood.size() != range + 1)) {
      LOG(ERROR) << "nhood size " << nhood.size() << " != R + 1 (" << range + 1 << ")";
    }

    // Compute distances from center and target to all neighbors.
    std::vector<float> center_dists(nhood.size()), target_dists(nhood.size());
    compute_distances(center_id, nhood.data(), nhood.size(), center_dists.data());
    compute_distances(target_id, nhood.data(), nhood.size(), target_dists.data());

    assert(nhood.back() == target_id);
    float target_center_dist = center_dists.back();

    // Build pool with both distances and sort by distance to center.
    std::vector<TriangleNeighbor> pool(nhood.size());
    for (uint32_t i = 0; i < nhood.size(); i++) {
      pool[i] = {nhood[i], target_dists[i], center_dists[i]};
    }
    std::sort(pool.begin(), pool.end());

    uint32_t to_evict = kInvalidID;
    uint32_t tgt_idx = kInvalidID;

    // Fast path: try to find a point to evict using triangle inequality with target.
    // From farthest to nearest.
    float cur_alpha = alpha;
    while (cur_alpha >= (1.0f - 1e-5f) && to_evict == kInvalidID) {
      for (int i = (int) pool.size() - 1; i >= 0; --i) {
        if (pool[i].id == target_id) {
          tgt_idx = i;
          continue;
        }
        // Check if target occludes pool[i] or vice versa.
        if (pool[i].distance > target_center_dist) {
          // pool[i] -> center is the longest edge.
          if (get_occlude_factor(metric, pool[i].distance, pool[i].tgt_dis) > cur_alpha) {
            to_evict = (uint32_t) i;
            break;
          }
        } else {
          // target -> center is the longest edge.
          if (get_occlude_factor(metric, target_center_dist, pool[i].tgt_dis) > cur_alpha) {
            to_evict = tgt_idx;
            break;
          }
        }
      }
      cur_alpha /= 1.2f;
    }

    auto finish = [&]() {
      nhood.clear();
      nhood.reserve(range);
      for (uint32_t i = 0; i < pool.size(); i++) {
        if (i != to_evict) {
          nhood.emplace_back(pool[i].id);
        }
      }
    };

    if (to_evict != kInvalidID) {
      finish();
      return;
    }

    // Fast path failed. The target is high quality.
    // Seek another low quality point to evict using full alpha-RNG check.
    // Copy sorted IDs for batch distance computation.
    std::vector<uint32_t> ids(pool.size());
    for (uint32_t i = 0; i < pool.size(); i++) {
      ids[i] = pool[i].id;
    }

    for (uint32_t start = 0; start < pool.size() - 1; ++start) {
      if (start == tgt_idx) {
        continue;
      }
      // Batch compute distances from pool[start] to all points after it.
      compute_distances(ids[start], ids.data() + start + 1, pool.size() - start - 1, center_dists.data() + start + 1);
      for (uint32_t t = start + 1; t < pool.size(); t++) {
        if (get_occlude_factor(metric, pool[t].distance, center_dists[t]) > alpha) {
          to_evict = t;
          finish();
          return;
        }
      }
    }

    // All points satisfy alpha-RNG, evict the farthest.
    to_evict = pool.size() - 1;
    finish();
  }

  // ============================================================================
  // Sample 2-hop neighbors of `center`, shuffled, up to `nnbrs_dense` results.
  // ============================================================================
  // `get_nbrs(id)` returns `id`'s 1-hop neighbor list as a (uint32_t *, uint32_t) pair.
  // The pointer only needs to stay valid until the next call to get_nbrs: the
  // helper copies `center`'s 1-hop into an internal thread-local buffer before
  // expanding further, so callers can reuse one scratch region across calls.
  template<typename GetNbrs>
  inline uint32_t sample_two_hop_nbrs(uint32_t center, uint32_t nnbrs_dense, uint32_t *buffer,
                                      GetNbrs &&get_nbrs) {
    if (nnbrs_dense == 0)
      return 0;

    // Thread-local RNG shared by the 2-hop sampler below.
    thread_local std::mt19937 rng{std::random_device{}()};
    thread_local tsl::robin_set<uint32_t> seen;
    thread_local std::vector<uint32_t> cands;
    seen.clear();
    cands.clear();

    auto [one_hop_start, one_hop_n] = get_nbrs(center);
    seen.insert(center);
    for (uint32_t i = 0; i < one_hop_n; i++) {
      seen.insert(one_hop_start[i]);
    }

    for (uint32_t i = 0; i < one_hop_n; i++) {
      auto [two_hop_start, two_hop_n] = get_nbrs(one_hop_start[i]);
      for (uint32_t j = 0; j < two_hop_n; j++) {
        uint32_t id = two_hop_start[j];
        if (seen.insert(id).second) {
          cands.push_back(id);
        }
      }
    }

    std::shuffle(cands.begin(), cands.end(), rng);
    uint32_t n_take = std::min<uint32_t>(nnbrs_dense, cands.size());
    for (uint32_t i = 0; i < n_take; i++) {
      buffer[i] = cands[i];
    }
    return n_take;
  }
}  // namespace pipeann

#endif  // PRUNE_NEIGHBORS_H_
