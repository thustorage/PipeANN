#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cblas.h>
#include <omp.h>

#include "pipnn.h"
#include "index.h"
#include "utils/index_build_utils.h"
#include "utils/timer.h"

namespace pipeann {

  // --- HashPruner implementation ---

  template<typename T>
  HashPruner::HashPruner(int64_t num_nodes, int64_t dims, int num_hyperplanes, int climit, const T *raw_data,
                         uint32_t seed)
      : _norm_vectors(nullptr), _sketch(nullptr), _reservoirs(num_nodes), _num_hyperplanes(num_hyperplanes),
        _dims(dims), _num_nodes(num_nodes), _climit(climit) {
    _norm_vectors = new float[num_hyperplanes * dims];
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < num_hyperplanes * dims; i++) {
      _norm_vectors[i] = dist(g);
    }

    // _sketch = raw_data (num_nodes x dims) x _norm_vectors^T (dims x num_hyperplanes)
    // Partition nodes into batches to limit temporary float buffer memory.
    _sketch = new float[num_nodes * num_hyperplanes];
    constexpr int64_t BATCH_SIZE = 100000;
    std::vector<float> buf;
    for (int64_t offset = 0; offset < num_nodes; offset += BATCH_SIZE) {
      int64_t batch = std::min(BATCH_SIZE, num_nodes - offset);
      const float *batch_data;
      if constexpr (std::is_same_v<T, float>) {
        batch_data = raw_data + offset * dims;
      } else {
        buf.resize(batch * dims);
        const T *src = raw_data + offset * dims;
        for (int64_t i = 0; i < batch * dims; i++) {
          buf[i] = static_cast<float>(src[i]);
        }
        batch_data = buf.data();
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, num_hyperplanes, dims, 1.0f, batch_data, dims,
                  _norm_vectors, dims, 0.0f, _sketch + offset * num_hyperplanes, num_hyperplanes);
    }
  }

  template HashPruner::HashPruner(int64_t, int64_t, int, int, const float *, uint32_t);
  template HashPruner::HashPruner(int64_t, int64_t, int, int, const int8_t *, uint32_t);
  template HashPruner::HashPruner(int64_t, int64_t, int, int, const uint8_t *, uint32_t);

  HashPruner::~HashPruner() {
    delete[] _norm_vectors;
    delete[] _sketch;
  }

  void HashPruner::update(const std::vector<WeightedEdge> &edges) {
    for (const auto &edge : edges) {
      uint32_t u = edge.u;
      uint32_t v = edge.v;
      float dist = edge.dist;

      uint16_t hash = 0;
      for (int h = 0; h < _num_hyperplanes; h++) {
        float diff = _sketch[u * _num_hyperplanes + h] - _sketch[v * _num_hyperplanes + h];
        hash = (hash << 1) | (diff >= 0 ? 1 : 0);
      }

      auto &M = _reservoirs[u];

      auto it = std::lower_bound(M.begin(), M.end(), HashPruneSlot{hash, 0, 0});
      if (it != M.end() && it->hash == hash) {
        if (dist < it->dist) {
          it->id = v;
          it->dist = dist;
        }
      } else {
        if (M.size() < _climit) {
          M.insert(it, {hash, v, dist});
        } else {
          auto farthest = std::max_element(
              M.begin(), M.end(), [](const HashPruneSlot &a, const HashPruneSlot &b) { return a.dist < b.dist; });
          if (dist < farthest->dist) {
            M.erase(farthest);
            it = std::lower_bound(M.begin(), M.end(), HashPruneSlot{hash, 0, 0});
            M.insert(it, {hash, v, dist});
          }
        }
      }
    }
  }

  void HashPruner::to_graph(std::vector<std::vector<unsigned>> &final_graph) {
    final_graph.resize(_num_nodes);
    for (int64_t u = 0; u < _num_nodes; u++) {
      for (const auto &slot : _reservoirs[u]) {
        final_graph[u].push_back(slot.id);
      }
    }
  }

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

    LOG(INFO) << "Using PiPNN builder";
    LOG(INFO) << "Threads:" << omp_get_max_threads();
    auto &pc = params.pipnn_config;
    HashPruner pruner(total_node, _dim, pc.num_hyperplanes, pc.climit, _data.data(), pc.seed);

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
          pruner.update(buckets[t * num_buckets + tid]);
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

    pruner.to_graph(_final_graph);

    final_prune(params);
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

      if (use_parallel && nodes.size() > 10000) {
        // Parallel version
        std::vector<T> center_data(num_centers * _dim);
        for (int i = 0; i < num_centers; i++) {
          std::copy_n(_data.data() + _dim * (size_t) centers[i], _dim, center_data.data() + i * _dim);
        }

        int num_threads = omp_get_max_threads();
        const size_t BATCH_SIZE = 10000;
        size_t num_batches = (nodes.size() + BATCH_SIZE - 1) / BATCH_SIZE;

        std::vector<Partition> local_partitions(num_threads, Partition(num_centers));

#pragma omp parallel for schedule(dynamic)
        for (size_t b = 0; b < num_batches; b++) {
          size_t batch_start = b * BATCH_SIZE;
          size_t batch_end = std::min(batch_start + BATCH_SIZE, nodes.size());
          size_t batch_size = batch_end - batch_start;

          std::vector<T> node_data_batch(batch_size * _dim);
          for (size_t i = 0; i < batch_size; i++) {
            std::copy_n(_data.data() + _dim * (size_t) nodes[batch_start + i], _dim, node_data_batch.data() + i * _dim);
          }

          std::vector<float> dist_batch(batch_size * num_centers);
          _distance->bulk_compare(node_data_batch.data(), batch_size, center_data.data(), num_centers, _dim,
                                  dist_batch.data());

          int tid = omp_get_thread_num();
          auto &my_partitions = local_partitions[tid];

          for (size_t i = 0; i < batch_size; i++) {
            std::vector<std::pair<float, int>> dists(num_centers);
            for (int j = 0; j < num_centers; j++) {
              dists[j] = {dist_batch[i * num_centers + j], j};
            }

            int n_select = std::min(local_fanout, num_centers);
            std::partial_sort(dists.begin(), dists.begin() + n_select, dists.end());

            for (int k = 0; k < n_select; k++) {
              my_partitions[dists[k].second].push_back(nodes[batch_start + i]);
            }
          }
        }

        for (int t = 0; t < num_threads; t++) {
          for (int p = 0; p < num_centers; p++) {
            partitions[p].insert(partitions[p].end(), local_partitions[t][p].begin(), local_partitions[t][p].end());
          }
        }
      } else {
        // Sequential version (batched bulk_compare)
        std::vector<T> center_data(num_centers * _dim);
        for (int i = 0; i < num_centers; i++) {
          std::copy_n(_data.data() + _dim * (size_t) centers[i], _dim, center_data.data() + i * _dim);
        }

        const size_t BATCH_SIZE = 2048;
        for (size_t batch_start = 0; batch_start < nodes.size(); batch_start += BATCH_SIZE) {
          size_t batch_end = std::min(batch_start + BATCH_SIZE, nodes.size());
          size_t batch_size = batch_end - batch_start;

          std::vector<T> node_data_batch(batch_size * _dim);
          for (size_t i = 0; i < batch_size; i++) {
            std::copy_n(_data.data() + _dim * (size_t) nodes[batch_start + i], _dim, node_data_batch.data() + i * _dim);
          }

          std::vector<float> dist_batch(batch_size * num_centers);
          _distance->bulk_compare(node_data_batch.data(), batch_size, center_data.data(), num_centers, _dim,
                                  dist_batch.data());

          for (size_t i = 0; i < batch_size; i++) {
            std::vector<std::pair<float, int>> dists(num_centers);
            for (int j = 0; j < num_centers; j++) {
              dists[j] = {dist_batch[i * num_centers + j], j};
            }

            int n_select = std::min(local_fanout, num_centers);
            std::partial_sort(dists.begin(), dists.begin() + n_select, dists.end());

            for (int k = 0; k < n_select; k++) {
              partitions[dists[k].second].push_back(nodes[batch_start + i]);
            }
          }
        }
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
  std::vector<WeightedEdge> Index<T, TagT>::pipnn_build_leaf_nodes(
      const std::vector<uint32_t> &nodes, int k) {
    std::vector<WeightedEdge> edges;

    size_t n = nodes.size();
    if (n <= 1)
      return edges;

    std::vector<T> node_data(n * _dim);
    for (size_t i = 0; i < n; i++) {
      std::copy_n(_data.data() + _dim * (size_t) nodes[i], _dim, node_data.data() + i * _dim);
    }

    std::vector<float> distances(n * n);
    _distance->bulk_compare(node_data.data(), n, node_data.data(), n, _dim, distances.data());

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

  // --- build_pipnn_index ---

  template<typename T, typename TagT>
  bool build_pipnn_index(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t l1_fanout,
                         uint32_t l2_fanout, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                         pipeann::Metric _compareMetric, const char *tag_file, AbstractNeighbor<T> *nbr_handler,
                         AbstractLabel *label) {
    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    LOG(INFO) << "Starting PiPNN index build: R=" << R << " L1Fanout=" << l1_fanout << " L2Fanout=" << l2_fanout
              << " Build RAM budget: " << M << "GB T: " << num_threads << " bytes per neighbor: " << bytes_per_nbr;

    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
      normalize_data_file<T>(dataFilePath, normalized_file_path);
    }

    auto s = std::chrono::high_resolution_clock::now();
    nbr_handler->build(index_prefix_path, normalized_file_path, bytes_per_nbr);

    size_t base_num, base_dim;
    pipeann::get_bin_metadata(normalized_file_path, base_num, base_dim);

    pipeann::IndexBuildParameters paras;
    paras.set(R, 0, 750, 1.2, 0, true);
    paras.pipnn_set(2048, 256, 0.01f, 1000, l1_fanout, l2_fanout, 2, 12, 128);

    auto start = std::chrono::high_resolution_clock::now();
    auto _pIndex = std::make_unique<pipeann::Index<T>>(_compareMetric, base_dim);
    _pIndex->build(normalized_file_path.c_str(), base_num, paras, tag_file, false);
    _pIndex->save(mem_index_path.c_str());
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "PiPNN index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (tag_file == nullptr) {
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, "", disk_index_path, label);
    } else {
      std::string tag_filename = std::string(tag_file);
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, disk_index_path, label);
    }

    LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    std::remove(mem_index_path.c_str());
    std::remove((mem_index_path + ".data").c_str());
    if (normalized_file_path != dataFilePath) {
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  // Explicit template instantiations for Index pipnn methods
  template void Index<float, uint32_t>::pipnn_link(IndexBuildParameters &);
  template void Index<int8_t, uint32_t>::pipnn_link(IndexBuildParameters &);
  template void Index<uint8_t, uint32_t>::pipnn_link(IndexBuildParameters &);

  template std::vector<std::vector<uint32_t>> Index<float, uint32_t>::pipnn_partition(const std::vector<uint32_t> &, IndexBuildParameters &);
  template std::vector<std::vector<uint32_t>> Index<int8_t, uint32_t>::pipnn_partition(const std::vector<uint32_t> &, IndexBuildParameters &);
  template std::vector<std::vector<uint32_t>> Index<uint8_t, uint32_t>::pipnn_partition(const std::vector<uint32_t> &, IndexBuildParameters &);

  template std::vector<WeightedEdge> Index<float, uint32_t>::pipnn_build_leaf_nodes(const std::vector<uint32_t> &, int);
  template std::vector<WeightedEdge> Index<int8_t, uint32_t>::pipnn_build_leaf_nodes(const std::vector<uint32_t> &, int);
  template std::vector<WeightedEdge> Index<uint8_t, uint32_t>::pipnn_build_leaf_nodes(const std::vector<uint32_t> &, int);

  template bool build_pipnn_index<int8_t, uint32_t>(const char *, const char *, uint32_t, uint32_t, uint32_t, uint32_t,
                                                     uint32_t, uint32_t, pipeann::Metric, const char *,
                                                     AbstractNeighbor<int8_t> *, AbstractLabel *);
  template bool build_pipnn_index<uint8_t, uint32_t>(const char *, const char *, uint32_t, uint32_t, uint32_t, uint32_t,
                                                      uint32_t, uint32_t, pipeann::Metric, const char *,
                                                      AbstractNeighbor<uint8_t> *, AbstractLabel *);
  template bool build_pipnn_index<float, uint32_t>(const char *, const char *, uint32_t, uint32_t, uint32_t, uint32_t,
                                                    uint32_t, uint32_t, pipeann::Metric, const char *,
                                                    AbstractNeighbor<float> *, AbstractLabel *);
}  // namespace pipeann
