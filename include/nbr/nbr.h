#ifndef NBR_H_
#define NBR_H_

#include "nbr/abstract_nbr.h"
#include "nbr/raw_nbr.h"
#include "nbr/dummy_nbr.h"
#include "nbr/pq_nbr.h"
#include "nbr/rabitq_nbr.h"
#include <filesystem>

namespace pipeann {
  // Accept a RAW bin, a compressed record file, or a codec prefix. The metric
  // remains explicit because existing bin/codec files do not persist it.
  template<typename T>
  std::unique_ptr<AbstractNeighbor<T>> load_neighbor(const std::string &path, Metric metric = L2) {
    std::string prefix = path;
    for (const std::string suffix : {"_processed.bin", "_pq_compressed.bin"}) {
      if (path.size() >= suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        prefix.resize(path.size() - suffix.size());
        break;
      }
    }
    const bool pq = std::filesystem::exists(prefix + "_pq_pivots.bin");
    bool rbq = false;
    for (const char *suffix : RABITQ_METADATA_SUFFIXES)
      rbq |= std::filesystem::exists(prefix + suffix);
    if (pq && rbq)
      throw std::invalid_argument("Ambiguous neighbor metadata: " + prefix);

    std::unique_ptr<AbstractNeighbor<T>> neighbor;
    if (pq) {
      neighbor = std::make_unique<PQNeighbor<T>>(metric);
    } else if (rbq) {
      std::vector<uint64_t> metadata;
      size_t rows, cols;
      load_bin(prefix + "_rabitq_metadata.bin", metadata, rows, cols);
      if (rows != 1 || cols != 3 || !metadata[0] || metadata[0] > 1048576 ||
          metadata[1] != RaBitQQuantizer<T, 8>::pad_dim(metadata[0]))
        throw std::runtime_error("Invalid RaBitQ metadata: " + prefix);
      const auto dim = metadata[0], bytes = metadata[2];
      if (bytes == RaBitQQuantizer<T, 1>::vector_size(dim))
        neighbor = std::make_unique<RaBitQNeighbor<T, 1>>(metric);
      else if (bytes == RaBitQQuantizer<T, 2>::vector_size(dim))
        neighbor = std::make_unique<RaBitQNeighbor<T, 2>>(metric);
      else if (bytes == RaBitQQuantizer<T, 4>::vector_size(dim))
        neighbor = std::make_unique<RaBitQNeighbor<T, 4>>(metric);
      else if (bytes == RaBitQQuantizer<T, 8>::vector_size(dim))
        neighbor = std::make_unique<RaBitQNeighbor<T, 8>>(metric);
      else
        throw std::runtime_error("Unsupported RaBitQ record size: " + prefix);
    } else {
      size_t count, dim;
      get_bin_metadata(path, count, dim);
      if (!dim || std::filesystem::file_size(path) != 8 + count * dim * sizeof(T))
        throw std::invalid_argument("RAW vector file shape/size mismatch: " + path);
      return std::make_unique<RawNeighbor<T>>(metric, dim, path.c_str());
    }
    neighbor->load(prefix.c_str());
    return neighbor;
  }

  // Create an untrained, unloaded Neighbor of the requested type.
  template<typename T>
  inline AbstractNeighbor<T> *create_neighbor(Metric metric, const std::string &nbr_type) {
    if (nbr_type == "rabitq") {
      return new RaBitQNeighbor<T>(metric);
    } else if (nbr_type == "pq") {
      return new PQNeighbor<T>(metric);
    } else if (nbr_type == "dummy") {
      return new DummyNeighbor<T>(metric);
    } else if (nbr_type == "rabitq2") {
      return new RaBitQNeighbor<T, 2>(metric);
    } else if (nbr_type == "rabitq4") {
      return new RaBitQNeighbor<T, 4>(metric);
    } else if (nbr_type == "rabitq8") {
      return new RaBitQNeighbor<T, 8>(metric);
    }
    throw std::invalid_argument("Unknown neighbor type: " + nbr_type);
  }
}  // namespace pipeann

#endif  // NBR_H_
