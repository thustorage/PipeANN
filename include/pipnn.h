#pragma once

#include <cstdint>
#include <vector>

#include "nbr/abstract_nbr.h"
#include "filter/label.h"
#include "utils.h"

namespace pipeann {
  struct WeightedEdge {
    uint32_t u, v;
    float dist;
  };

  struct HashPruneSlot {
    uint16_t hash;
    uint32_t id;
    float dist;

    bool operator<(const HashPruneSlot &other) const {
      return hash < other.hash;
    }
  };

  struct HashPruner {
    float *_norm_vectors;
    float *_sketch;
    std::vector<std::vector<HashPruneSlot>> _reservoirs;
    int _num_hyperplanes;
    int _dims;
    int64_t _num_nodes;
    size_t _climit;

    template<typename T>
    HashPruner(int64_t num_nodes, int64_t dims, int num_hyperplanes, int climit, const T *raw_data,
               uint32_t seed = 42);

    ~HashPruner();

    void update(const std::vector<WeightedEdge> &edges);
    void to_graph(std::vector<std::vector<unsigned>> &final_graph);
  };

  template<typename T, typename TagT = uint32_t>
  bool build_pipnn_index(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t l1_fanout,
                         uint32_t l2_fanout, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                         pipeann::Metric _compareMetric, const char *tag_file, AbstractNeighbor<T> *nbr_handler,
                         AbstractLabel *label);
}  // namespace pipeann
