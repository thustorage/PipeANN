#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include "utils/arch_compat.h"
#include "utils.h"
#include "nbr/pq_table.h"
#include "nbr/rabitq_quantizer.h"

namespace pipeann {
  template<typename T>
  class Distance {
   public:
    // RAW has no separate metadata. A nonzero expected_dim validates the loaded model.
    virtual void load_metadata(const std::string &prefix, uint64_t expected_dim = 0) {}
    virtual void save_metadata(const std::string &prefix) const {}

    // RAW queries use the stored-record representation. Encoded distances may
    // override this to produce a query-specific representation such as a LUT.
    // Input may alias out; normalization is performed by the caller beforehand.
    virtual void initialize_query(const T *query, void *out, unsigned length) const {
      encode(query, out, length);
    }

    virtual void encode(const T *point, void *out, unsigned length) const {
      if (out != point) std::memmove(out, point, length * sizeof(T));
    }

    // Reconstruct coordinates at export/output boundaries; encoded values are approximate.
    virtual void decode(const void *record, T *out, unsigned length) const {
      std::memmove(out, record, length * sizeof(T));
    }

    virtual float compare(const void *a, const void *b, unsigned length) const = 0;

    // RAW uses the same representation for queries and stored records. Encoded
    // distances override this method for code-to-code comparisons.
    virtual float compare_sym(const uint8_t *a, const void *b, unsigned length) const {
      return compare(static_cast<const void *>(a), b, length);
    }

    // Existing dense RAW matrix API (BLAS specializations override this).
    virtual void bulk_compare(const T *a, size_t n, const T *b, size_t m, unsigned length, float *out) const {
      for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) out[i * m + j] = compare(a + i * length, b + j * length, length);
    }

    // Indexed batch over fixed-size records. Layout is supplied by Neighbor.
    // Concrete implementations use compare_indexed or a specialized batch kernel.
    virtual void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                              unsigned length, size_t data_size) const = 0;
    virtual void bulk_compare_sym(const uint8_t *a, const void *b, const uint32_t *ids, size_t n, float *out,
                              unsigned length, size_t data_size) const {
      bulk_compare(static_cast<const void *>(a), b, ids, n, out, length, data_size);
    }
    virtual ~Distance() = default;
  };

  // Shared gather/prefetch loop for concrete bulk_compare overrides. Passing *this
  // deduces the concrete type D, so each distance class can reuse the loop without
  // a virtual compare call per record. An outer bulk_compare call through Distance<T>
  // may still use virtual dispatch once to select the override.
  template<class D, class QueryT>
  inline void compare_indexed(const D &distance, const QueryT *a, const void *b,
                              const uint32_t *ids, size_t n, float *out, unsigned length,
                              size_t data_size) {
    constexpr size_t kPrefetchWindow = 8;
    const auto *bytes = static_cast<const uint8_t *>(b);
    auto prefetch = [&](size_t i) { prefetch_vector(reinterpret_cast<const char *>(bytes + ids[i] * data_size), data_size); };
    for (size_t i = 0; i < std::min(kPrefetchWindow, n); ++i) prefetch(i);
    for (size_t i = 0; i < n; ++i) {
      if (kPrefetchWindow < n - i) prefetch(i + kPrefetchWindow);
      // Explicit qualification suppresses virtual dispatch; inlining is up to the compiler.
      out[i] = distance.D::compare(a, bytes + ids[i] * data_size, length);
    }
  }

  class DistanceCosineInt8 final : public Distance<int8_t> {
   public:
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      compare_indexed(*this, a, b, ids, n, out, length, data_size);
    }

    virtual float compare(const void *a, const void *b, uint32_t length) const;
    virtual void bulk_compare(const int8_t *a, size_t n, const int8_t *b, size_t m, unsigned length,
                              float *result) const override;
  };

  class DistanceCosineFloat final : public Distance<float> {
   public:
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      compare_indexed(*this, a, b, ids, n, out, length, data_size);
    }

    virtual float compare(const void *a, const void *b, uint32_t length) const;
    virtual void bulk_compare(const float *a, size_t n, const float *b, size_t m, unsigned length,
                              float *result) const override;
  };

  class DistanceCosineUInt8 final : public Distance<uint8_t> {
   public:
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      compare_indexed(*this, a, b, ids, n, out, length, data_size);
    }

    virtual float compare(const void *a, const void *b, uint32_t length) const;
    virtual void bulk_compare(const uint8_t *a, size_t n, const uint8_t *b, size_t m, unsigned length,
                              float *result) const override;
  };

  class DistanceL2Int8 final : public Distance<int8_t> {
   public:
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      compare_indexed(*this, a, b, ids, n, out, length, data_size);
    }

    virtual float compare(const void *a, const void *b, uint32_t size) const;
    virtual void bulk_compare(const int8_t *a, size_t n, const int8_t *b, size_t m, unsigned length,
                              float *result) const override;
  };

  class DistanceL2UInt8 final : public Distance<uint8_t> {
   public:
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      compare_indexed(*this, a, b, ids, n, out, length, data_size);
    }

    virtual float compare(const void *a, const void *b, uint32_t size) const;
    virtual void bulk_compare(const uint8_t *a, size_t n, const uint8_t *b, size_t m, unsigned length,
                              float *result) const override;
  };

  class DistanceL2Float final : public Distance<float> {
   public:
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      compare_indexed(*this, a, b, ids, n, out, length, data_size);
    }

    virtual float compare(const void *a, const void *b, uint32_t size) const __attribute__((hot));
    virtual void bulk_compare(const float *a, size_t n, const float *b, size_t m, unsigned length,
                              float *result) const override;
  };

  // Query comparisons consume a prepared LUT; symmetric comparisons consume codes.
  template<typename T>
  class DistancePQ final : public Distance<T> {
   public:
    FixedChunkPQTable<T> table;
    explicit DistancePQ(Metric metric) : table(metric) {}

    void load_metadata(const std::string &prefix, uint64_t expected_dim = 0) override {
      table.load_pq_centroid_bin((prefix + "_pq_pivots.bin").c_str());
      if (expected_dim && table.ndims != expected_dim)
        throw std::runtime_error("PQ metadata dimension mismatch");
    }
    void save_metadata(const std::string &prefix) const override {
      table.save_pq_pivots((prefix + "_pq_pivots.bin").c_str());
    }

    void initialize_query(const T *query, void *out, unsigned length) const override {
      if (out == query) {
        const std::vector<T> input(query, query + table.ndims);
        table.populate_chunk_distances(input.data(), static_cast<float *>(out));
      } else {
        table.populate_chunk_distances(query, static_cast<float *>(out));
      }
    }
    void encode(const T *point, void *out, unsigned length) const override {
      std::vector<float> values(table.ndims);
      for (size_t i = 0; i < table.ndims; ++i) values[i] = static_cast<float>(point[i]);
      table.deflate_vec(values.data(), static_cast<uint8_t *>(out));
    }
    void decode(const void *record, T *out, unsigned length) const override {
      const auto *codes = static_cast<const uint8_t *>(record);
      for (size_t c = 0; c < table.n_chunks; ++c)
        for (size_t d = table.chunk_offsets[c]; d < table.chunk_offsets[c + 1]; ++d) {
          float value = table.tables[codes[c] * table.ndims + d] + table.centroid[d];
          if constexpr (std::is_integral_v<T>)
            value = std::clamp(value, float(std::numeric_limits<T>::lowest()), float(std::numeric_limits<T>::max()));
          out[d] = static_cast<T>(value);
        }
    }
    // Asymmetric: a is a prepared lookup table; b contains stored codes.
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      table.compute_distances(static_cast<const float *>(a), static_cast<const uint8_t *>(b),
                               ids, n, out, data_size);
    }
    // Symmetric: a is a stored code; b contains stored codes.
    void bulk_compare_sym(const uint8_t *a, const void *b, const uint32_t *ids, size_t n, float *out,
                      unsigned length, size_t data_size) const override {
      table.compute_distances(a, static_cast<const uint8_t *>(b), ids, n, out, data_size);
    }
    float compare(const void *a, const void *b, unsigned) const override {
      return table.compute_distance(static_cast<const float *>(a), static_cast<const uint8_t *>(b));
    }
    float compare_sym(const uint8_t *a, const void *b, unsigned) const override {
      return table.compute_distance(a, static_cast<const uint8_t *>(b));
    }
  };

  template<typename T, unsigned Bits>
  class DistanceRaBitQ final : public Distance<T> {
   public:
    RaBitQQuantizer<T, Bits> quantizer;

    explicit DistanceRaBitQ(Metric metric) : quantizer(metric) {}

    void load_metadata(const std::string &prefix, uint64_t expected_dim = 0) override {
      quantizer.load(prefix);
      if (expected_dim && quantizer.dim != expected_dim)
        throw std::runtime_error("RaBitQ metadata dimension mismatch");
    }
    void save_metadata(const std::string &prefix) const override {
      quantizer.save(prefix);
    }

    void initialize_query(const T *query, void *out, unsigned length) const override {
      quantizer.initialize_query(query, out);
    }

    void encode(const T *point, void *out, unsigned length) const override {
      quantizer.encode(point, static_cast<uint8_t *>(out));
    }

    void decode(const void *record, T *out, unsigned length) const override {
      quantizer.decode(static_cast<const uint8_t *>(record), out);
    }

    // Asymmetric: a is a prepared query; b is a stored code.
    float compare(const void *a, const void *b, unsigned) const override {
      return quantizer.compute_distance(a, static_cast<const uint8_t *>(b));
    }
    // Symmetric: a and b are stored codes.
    float compare_sym(const uint8_t *a, const void *b, unsigned) const override {
      return quantizer.compute_distance(a, static_cast<const uint8_t *>(b));
    }
    using Distance<T>::bulk_compare;
    void bulk_compare(const void *a, const void *b, const uint32_t *ids, size_t n, float *out, unsigned,
                      size_t data_size) const override {
      quantizer.compute_distances(a, static_cast<const uint8_t *>(b), ids, n, out, data_size);
    }
    void bulk_compare_sym(const uint8_t *a, const void *b, const uint32_t *ids, size_t n, float *out, unsigned,
                          size_t data_size) const override {
      quantizer.compute_distances(a, static_cast<const uint8_t *>(b), ids, n, out, data_size);
    }
  };

  inline Metric get_metric(const std::string &metric_str) {
    if (metric_str == "l2") {
      return Metric::L2;
    } else if (metric_str == "cosine") {
      return Metric::COSINE;
    } else if (metric_str == "mips") {
      return Metric::INNER_PRODUCT;
    } else {
      LOG(ERROR) << "Unsupported metric: " << metric_str << ". Using L2.";
      return Metric::L2;
    }
  }

  inline std::string get_metric_str(Metric m) {
    switch (m) {
      case Metric::L2:
        return "l2";
      case Metric::COSINE:
        return "cosine";
      case Metric::INNER_PRODUCT:
        return "mips";
      default:
        return "unknown";
    }
  }
  // The distance function does not return the actual distance, but reserves the partial order.
  // For L2, it returns the squared L2 distance.
  // For IP, it returns A^2 - inner_product, where A is 1 if float, 255 if uint8_t, and 127 if int8_t.
  // For cosine, it returns A^2 * (1 - cosine(theta)). Data should be first normalized by normalize_data.
  // Note that cosine distance function is used for both inner_product and cosine metrics.
  template<typename T>
  Distance<T> *get_distance_function(Metric m);
}  // namespace pipeann
