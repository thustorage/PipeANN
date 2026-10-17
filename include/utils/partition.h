#pragma once
#include <cassert>
#include <string>
#include <vector>
#include "distance.h"

template<typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate,
                      size_t offset = 0);

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims);

// Partition prepared RAW vectors or RBQ8 records. K-means uses bounded float
// buffers; shard records and codec metadata are copied without re-encoding.
template<typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base,
                              pipeann::Metric metric = pipeann::L2);
