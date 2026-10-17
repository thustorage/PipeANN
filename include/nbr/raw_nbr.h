#pragma once
#include "nbr/abstract_nbr.h"
#include "distance.h"

namespace pipeann {
  template<typename T>
  class RawNeighbor : public AbstractNeighbor<T> {
    size_t dim_;
    using L2Distance =
        std::conditional_t<std::is_same_v<T, float>, DistanceL2Float,
                           std::conditional_t<std::is_same_v<T, int8_t>, DistanceL2Int8, DistanceL2UInt8>>;
    using IPDistance =
        std::conditional_t<std::is_same_v<T, float>, DistanceCosineFloat,
                           std::conditional_t<std::is_same_v<T, int8_t>, DistanceCosineInt8, DistanceCosineUInt8>>;
    L2Distance l2_;
    IPDistance ip_;
    template<class F>
    void with_distance(F &&f) const {
      if (this->metric == L2)
        f(l2_);
      else
        f(ip_);
    }

   public:
    RawNeighbor(Metric metric, size_t dim) : AbstractNeighbor<T>(metric), dim_(dim) {
      this->data_size = dim * sizeof(T);
    }
    RawNeighbor(Metric metric, size_t dim, const char *filename) : RawNeighbor(metric, dim) {
      load(filename);
    }
    std::string get_name() const override {
      return "RawNeighbor";
    }
    uint64_t dim() const override {
      return dim_;
    }
    uint64_t query_ctx_size() override {
      return this->data_size;
    }
    void load(const char *filename) override {
      this->load_metadata(filename);
      size_t n, dim;
      load_bin<T>(filename, this->data_, n, dim);
      if (dim != dim_)
        throw std::invalid_argument("RAW vector file dimension mismatch");
      this->npoints = n;
    }
    void initialize_query(const T *query, void *out) override {
      with_distance([&](const auto &distance) { distance.initialize_query(query, out, dim_); });
    }
    void encode(const T *point, void *out) override {
      with_distance([&](const auto &distance) { distance.encode(point, out, dim_); });
    }
    void decode(const void *record, T *out) const override {
      with_distance([&](const auto &distance) { distance.decode(record, out, dim_); });
    }
    void compute_dists_sym(const uint8_t *record, const uint32_t *ids, size_t n, float *out) override {
      with_distance([&](const auto &distance) {
        distance.bulk_compare_sym(record, this->data(), ids, n, out, dim_, this->data_size);
      });
    }
    void compute_dists(const void *query, const uint32_t *ids, size_t n, float *out) override {
      with_distance([&](const auto &distance) {
        distance.bulk_compare(query, this->data(), ids, n, out, dim_, this->data_size);
      });
    }
    void compute_dists(uint32_t source, const uint32_t *ids, size_t n, float *out) override {
      with_distance([&](const auto &distance) {
        distance.bulk_compare_sym(this->at(source), this->data(), ids, n, out, dim_, this->data_size);
      });
    }
    void insert(const T *point, uint32_t loc) override {
      memcpy(this->at(loc), point, this->data_size);
    }
  };
}  // namespace pipeann
