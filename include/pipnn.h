#pragma once

#include "nbr/abstract_nbr.h"
#include "filter/label.h"
#include "utils.h"

namespace pipeann {
  template<typename T, typename TagT = uint32_t>
  bool build_pipnn_index(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t l1_fanout,
                         uint32_t l2_fanout, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                         pipeann::Metric _compareMetric, const char *tag_file, AbstractNeighbor<T> *nbr_handler,
                         AbstractLabel *label);
}  // namespace pipeann
