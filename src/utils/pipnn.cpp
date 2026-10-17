#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "utils/prune_neighbors.h"
#include <omp.h>

#include "utils/pipnn.h"
#include "index.h"
#include "utils/timer.h"

namespace pipeann {

  // --- Index pipnn methods ---

  template<typename T, typename TagT>
  void Index<T, TagT>::pipnn_link(IndexBuildParameters &params) {
    unsigned num_threads = params.num_threads;
    unsigned L = params.L;
    params.print();

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    int64_t total_node = _nd;
    std::vector<uint32_t> node_ids;
    for (int i = 0; i < total_node; i++) {
      node_ids.push_back(i);
    }

    LOG(INFO) << "Using PiPNN builder, Threads: " << omp_get_max_threads();
    auto &pc = params.pipnn_config;
    // Keep at most twice the retained capacity per source before distance pruning.
    const size_t capacity = std::max<size_t>(params.R, std::max(1, pc.climit));
    std::vector<std::vector<Neighbor>> candidates(total_node);
    auto prune = [&](uint32_t source, size_t range) {
      auto &pool = candidates[source];
      auto prune_params = params;
      prune_params.R = range;
      prune_params.C = std::max<size_t>(params.C, pool.size());
      std::vector<uint32_t> kept;
      prune_neighbors(pool, kept, prune_params, _dist_metric,
                      [&](uint32_t a, const uint32_t *ids, size_t n, float *out) {
        _vectors->compute_dists(a, ids, n, out);
      });
      std::vector<Neighbor> retained;
      retained.reserve(kept.size());
      for (auto id : kept) {
        auto it = std::find_if(pool.begin(), pool.end(), [id](const Neighbor &n) { return n.id == id; });
        retained.push_back(*it);
      }
      pool.swap(retained);
    };
    auto update = [&](const std::vector<WeightedEdge> &edges) {
      for (const auto &edge : edges) {
        if (edge.u == edge.v) continue;
        auto &pool = candidates[edge.u];
        auto it = std::find_if(pool.begin(), pool.end(), [&](const Neighbor &n) { return n.id == edge.v; });
        if (it != pool.end()) { it->distance = std::min(it->distance, edge.dist); continue; }
        pool.emplace_back(edge.v, edge.dist);
        if (pool.size() >= 2 * capacity) prune(edge.u, capacity);
      }
    };

    pipeann::Timer link_timer;

    _ep = calculate_entry_point();

    LOG(INFO) << "Start Partition";

    auto blocks = pipnn_partition(node_ids, params);

    LOG(INFO) << "Finish Partition, " << ((double) link_timer.elapsed() / (double) 1000000) << "s";
    LOG(INFO) << "Start Leaf Build";

    const size_t GROUP_SIZE = 1000;
    int num_threads_used = omp_get_max_threads();
    int num_buckets = num_threads_used;
    size_t bucket_size = (_nd + num_buckets - 1) / num_buckets;

    double total_build_time = 0.0;
    double total_update_time = 0.0;

    for (size_t g_start = 0; g_start < blocks.size(); g_start += GROUP_SIZE) {
      size_t g_end = std::min(g_start + GROUP_SIZE, blocks.size());

      std::vector<std::vector<WeightedEdge>> buckets(num_threads_used * num_buckets);

      auto t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
      {
        int tid = omp_get_thread_num();

#pragma omp for schedule(dynamic)
        for (size_t i = g_start; i < g_end; i++) {
          auto edges = pipnn_build_leaf_nodes(blocks[i], pc.k);

          for (const auto &e : edges) {
            int bucket_id = std::min((int) (e.u / bucket_size), num_buckets - 1);
            buckets[tid * num_buckets + bucket_id].push_back(e);
          }
        }
      }

      auto t2 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
      {
        int tid = omp_get_thread_num();

        for (int t = 0; t < num_threads_used; t++) {
          update(buckets[t * num_buckets + tid]);
        }
      }

      auto t3 = std::chrono::high_resolution_clock::now();

      double build_time = std::chrono::duration<double>(t2 - t1).count();
      double update_time = std::chrono::duration<double>(t3 - t2).count();
      total_build_time += build_time;
      total_update_time += update_time;
    }

    LOG(INFO) << "Finish Leaf build, total build " << total_build_time << "s, total update " << total_update_time
              << "s";

#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t u = 0; u < total_node; ++u) {
      if (candidates[u].size() > params.R) prune(u, params.R);
      auto &neighbors = _final_graph[u];
      neighbors.clear();
      for (const auto &candidate : candidates[u]) neighbors.push_back(candidate.id);
    }
    if (_nd > 0) {
      LOG(INFO) << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000) << "s";
    }
  }

  template<typename T, typename TagT>
  std::vector<std::vector<uint32_t>> Index<T, TagT>::pipnn_partition(const std::vector<uint32_t> &node_ids,
                                                                     IndexBuildParameters &params) {
    using Partition = std::vector<std::vector<uint32_t>>;
    auto &pc = params.pipnn_config;
    const int cmax = pc.cmax;
    const int cmin = pc.cmin;
    const float p_samp = pc.p_samp;
    const int pmax = pc.pmax;
    const auto fanout = [&pc](int depth) {
      if (depth == 1)
        return pc.first_layer_fanout;
      if (depth == 2)
        return pc.second_layer_fanout;
      else
        return 1;
    };

    int num_rng_threads = omp_get_max_threads();
    std::vector<std::mt19937> gens(num_rng_threads);
    for (int i = 0; i < num_rng_threads; i++) {
      gens[i].seed(pc.seed + i);
    }

    auto merge = [&cmin, &cmax, &gens](Partition &partitions) {
      std::vector<int> small_indices;
      for (int i = 0; i < (int) partitions.size(); i++) {
        if ((int) partitions[i].size() < cmin) {
          small_indices.push_back(i);
        }
      }

      auto &rng = gens[omp_get_thread_num()];
      std::shuffle(small_indices.begin(), small_indices.end(), rng);

      for (int idx : small_indices) {
        if (partitions[idx].empty())
          continue;

        std::vector<int> candidates;
        for (int i = 0; i < (int) partitions.size(); i++) {
          if (i != idx && !partitions[i].empty() && partitions[i].size() < (size_t) cmax) {
            candidates.push_back(i);
          }
        }
        std::shuffle(candidates.begin(), candidates.end(), rng);

        for (int j : candidates) {
          if (partitions[idx].size() + partitions[j].size() <= (size_t) cmax) {
            partitions[j].insert(partitions[j].end(), partitions[idx].begin(), partitions[idx].end());
            partitions[idx].clear();
            break;
          }
        }
      }

      partitions.erase(std::remove_if(partitions.begin(), partitions.end(), [](const auto &p) { return p.empty(); }),
                       partitions.end());
    };

    // Shared function: assign nodes to nearest centers
    auto assign_nodes_to_centers = [&](const std::vector<uint32_t> &nodes, const std::vector<uint32_t> &centers,
                                       int local_fanout, bool use_parallel) -> Partition {
      int num_centers = centers.size();
      Partition partitions(num_centers);

      auto assign_batch = [&](size_t start, size_t end, Partition &out) {
        const size_t size = end - start;
        std::vector<float> distances(size * num_centers);
        for (size_t i = 0; i < size; ++i)
          _vectors->compute_dists(nodes[start + i], centers.data(), num_centers,
                                  distances.data() + i * num_centers);

        std::vector<std::pair<float, int>> ranked(num_centers);
        const int n_select = std::min(local_fanout, num_centers);
        for (size_t i = 0; i < size; ++i) {
          for (int j = 0; j < num_centers; ++j)
            ranked[j] = {distances[i * num_centers + j], j};
          std::partial_sort(ranked.begin(), ranked.begin() + n_select, ranked.end());
          for (int k = 0; k < n_select; ++k)
            out[ranked[k].second].push_back(nodes[start + i]);
        }
      };

      if (use_parallel && nodes.size() > 10000) {
        const int num_threads = omp_get_max_threads();
        const size_t batch_size = 10000;
        const size_t num_batches = (nodes.size() + batch_size - 1) / batch_size;
        std::vector<Partition> local_partitions(num_threads, Partition(num_centers));
#pragma omp parallel for schedule(dynamic)
        for (size_t b = 0; b < num_batches; ++b) {
          const size_t start = b * batch_size;
          assign_batch(start, std::min(start + batch_size, nodes.size()), local_partitions[omp_get_thread_num()]);
        }
        for (const auto &local : local_partitions)
          for (int p = 0; p < num_centers; ++p)
            partitions[p].insert(partitions[p].end(), local[p].begin(), local[p].end());
      } else {
        const size_t batch_size = 2048;
        for (size_t start = 0; start < nodes.size(); start += batch_size)
          assign_batch(start, std::min(start + batch_size, nodes.size()), partitions);
      }

      return partitions;
    };

    // Sequential recursive function
    std::function<Partition(const std::vector<uint32_t> &, int)> recurse_serial;
    recurse_serial = [&](const std::vector<uint32_t> &nodes, int depth) -> Partition {
      if ((int) nodes.size() <= cmax) {
        return {nodes};
      }

      int num_centers = std::min((int) (nodes.size() * p_samp), pmax);
      int local_fanout = fanout(depth);

      std::vector<uint32_t> centers(num_centers);
      std::sample(nodes.begin(), nodes.end(), centers.begin(), num_centers, gens[omp_get_thread_num()]);

      auto partitions = assign_nodes_to_centers(nodes, centers, local_fanout, false);

      merge(partitions);

      Partition result;
      for (auto &partition : partitions) {
        if (partition.empty())
          continue;

        auto sub_partitions = recurse_serial(partition, depth + 1);
        result.insert(result.end(), sub_partitions.begin(), sub_partitions.end());
      }

      return result;
    };

    // First level: parallel processing
    auto partition_first_level = [&](const std::vector<uint32_t> &nodes) -> Partition {
      int num_centers = std::min((int) (nodes.size() * p_samp), pmax);
      int local_fanout = fanout(1);

      std::vector<uint32_t> centers(num_centers);
      std::sample(nodes.begin(), nodes.end(), centers.begin(), num_centers, gens[omp_get_thread_num()]);

      auto partitions = assign_nodes_to_centers(nodes, centers, local_fanout, true);

      merge(partitions);

      LOG(INFO) << "Finish First Level";

      std::vector<Partition> sub_results(partitions.size());

#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < partitions.size(); i++) {
        if (!partitions[i].empty()) {
          sub_results[i] = recurse_serial(partitions[i], 2);
        }
      }

      Partition result;
      for (auto &sub : sub_results) {
        result.insert(result.end(), sub.begin(), sub.end());
      }

      return result;
    };

    return partition_first_level(node_ids);
  }

  template<typename T, typename TagT>
  std::vector<WeightedEdge> Index<T, TagT>::pipnn_build_leaf_nodes(const std::vector<uint32_t> &nodes, int k) {
    std::vector<WeightedEdge> edges;

    size_t n = nodes.size();
    if (n <= 1)
      return edges;

    std::vector<float> distances(n * n);
    for (size_t i = 0; i < n; ++i)
      _vectors->compute_dists(nodes[i], nodes.data(), n, distances.data() + i * n);

    for (size_t i = 0; i < n; i++) {
      std::vector<std::pair<float, size_t>> dists;
      dists.reserve(n - 1);

      for (size_t j = 0; j < n; j++) {
        if (i != j) {
          dists.emplace_back(distances[i * n + j], j);
        }
      }

      size_t num_neighbors = std::min((size_t) k, dists.size());
      std::partial_sort(dists.begin(), dists.begin() + num_neighbors, dists.end());

      for (size_t ki = 0; ki < num_neighbors; ki++) {
        uint32_t neighbor = nodes[dists[ki].second];
        float d = dists[ki].first;
        edges.push_back({nodes[i], neighbor, d});
        edges.push_back({neighbor, nodes[i], d});
      }
    }

    return edges;
  }

  // Explicit template instantiations for Index pipnn methods
  template class Index<float, uint32_t>;
  template class Index<int8_t, uint32_t>;
  template class Index<uint8_t, uint32_t>;
}  // namespace pipeann
