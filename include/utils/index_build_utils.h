#pragma once
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

#include <unistd.h>

#include "filter/attribute.h"
#include "nbr/nbr.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"

namespace pipeann {
  // Persist to filename + "_processed.bin" (RAW L2/IP can reuse filename).
  // RBQ8 uses the normal Neighbor metadata suffixes on filename as its prefix.
  template<typename T>
  std::string prepare_build_vectors(const std::string &filename, Metric metric,
                                    VectorType node_type = VectorType::RABITQ8);

  // Merge per-shard SSD-format indices into a single SSD-format disk file.
  // {shard_prefix}{shard}{shard_suffix} points to each shard's SSD index (e.g., "prefix_subshard-0_disk.index").
  // {idmaps_prefix}{shard}{idmaps_suffix} holds each shard's local->global id mapping.
  template<typename T, typename TagT = uint32_t>
  int merge_shards(const std::string &shard_prefix, const std::string &shard_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, uint64_t nshards, uint16_t R, uint16_t R_dense, uint16_t R_ood,
                   const std::string &output_disk_file, const std::string &tag_file = "",
                   AttrWriter *attr_writer = nullptr);

  template<typename T, typename TagT = uint32_t>
  int build_merged_index(const std::string &base_file, pipeann::Metric _compareMetric, uint16_t R, uint32_t L_or_L1,
                         double ram_budget, std::string disk_index_path,
                         const char *tag_file = nullptr, uint16_t R_dense = 0, uint32_t num_threads = 0,
                         uint32_t L2 = 0, const std::string &train_query_path = "", uint16_t R_ood = 0,
                         uint32_t L_ood = 1500, AttrWriter *attr_writer = nullptr, VectorType node_type = VectorType::RAW);

  // Build a disk index in the common file format.
  // L2 == 0 uses Vamana and treats L_or_L1 as build L; otherwise it uses PiPNN with L_or_L1/L2 as L1/L2.
  template<typename T, typename TagT = uint32_t>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, uint16_t R, uint32_t L_or_L1, uint32_t M,
                        uint32_t num_threads, uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                        const char *tag_file, AbstractNeighbor<T> *nbr_handler, AttrWriter *attr_writer,
                        uint16_t R_dense = 0, uint32_t L2 = 0, const std::string &train_query_path = "",
                        uint16_t R_ood = 0, uint32_t L_ood = 1500, VectorType node_type = VectorType::RAW);
}  // namespace pipeann
