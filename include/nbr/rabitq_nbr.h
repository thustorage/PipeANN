#pragma once
#include "nbr/abstract_nbr.h"
#include "distance.h"
#include "utils/kmeans_utils.h"
#include <fstream>
#include <shared_mutex>
#include "utils/lock_table.h"

namespace pipeann {
  template<typename T, size_t Bits = 1>
  class RaBitQNeighbor : public AbstractNeighbor<T> {
    // Reads dominate; keep reader counters on separate cache lines.
    mutable ReaderOptSharedMutex codes_mutex_;

   public:
    DistanceRaBitQ<T, Bits> distance;
    size_t training_centers = Bits == 8 ? 1 : 256;
    explicit RaBitQNeighbor(Metric metric) : AbstractNeighbor<T>(metric), distance(metric) {
      if (const char *setting = std::getenv("PIPEANN_RABITQ_CENTERS")) {
        training_centers = std::stoul(setting);
        if (!training_centers || training_centers > RaBitQQuantizer<T, Bits>::MAX_CENTROIDS)
          throw std::invalid_argument("RaBitQ centers must be in [1,65536]");
      }
    }
    std::string get_name() const override {
      return "RaBitQ" + std::to_string(Bits);
    }
    uint64_t query_ctx_size() override {
      return distance.quantizer.query_ctx_size();
    }
    uint64_t dim() const override {
      return distance.quantizer.dim;
    }
    void initialize_query(const T *query, void *out) override {
      distance.initialize_query(query, out, distance.quantizer.dim);
    }
    void encode(const T *point, void *out) override {
      distance.encode(point, out, distance.quantizer.dim);
    }
    void decode(const void *record, T *out) const override {
      distance.decode(record, out, distance.quantizer.dim);
    }
    void compute_dists_sym(const uint8_t *record, const uint32_t *ids, size_t n, float *out) override {
      std::shared_lock lock(codes_mutex_);
      distance.bulk_compare_sym(record, this->data(), ids, n, out, distance.quantizer.dim, this->data_size);
    }
    void compute_dists(uint32_t source, const uint32_t *ids, size_t count, float *out) override {
      std::shared_lock lock(codes_mutex_);
      distance.bulk_compare_sym(this->at(source), this->data(), ids, count, out, distance.quantizer.dim,
                            this->data_size);
    }
    void compute_dists(const void *query, const uint32_t *ids, size_t n, float *out) override {
      std::shared_lock lock(codes_mutex_);
      distance.bulk_compare(query, this->data(), ids, n, out, distance.quantizer.dim, this->data_size);
    }
    void save_metadata(const char *prefix) override {
      distance.save_metadata(prefix);
    }
    void load_metadata(const char *prefix) override {
      distance.load_metadata(prefix);
      this->data_size = distance.quantizer.vector_size();
    }
    void save(const char *prefix) override {
      save_metadata(prefix);
      save_bin(std::string(prefix) + "_processed.bin", this->data_.data(), this->npoints, this->data_size);
    }
    void load(const char *prefix) override {
      load_metadata(prefix);
      load_records((std::string(prefix) + "_processed.bin").c_str());
    }
    void train(const std::string &filename) override {
      size_t n, dim;
      get_bin_metadata(filename, n, dim);
      if (!n || !dim)
        throw std::invalid_argument("RaBitQ requires nonempty vectors");
      const size_t samples = std::min<size_t>(n, 32768);
      if (!training_centers || training_centers > RaBitQQuantizer<T, Bits>::MAX_CENTROIDS)
        throw std::invalid_argument("RaBitQ centers must be in [1,65536]");
      const size_t clusters = std::min(training_centers, samples);
      distance.quantizer.initialize(dim, clusters);
      this->data_size = distance.quantizer.vector_size();
      std::vector<float> training(samples * dim);
      std::vector<T> row(dim);
      std::ifstream f(filename, std::ios::binary);
      for (size_t i = 0; i < samples; ++i) {
        f.seekg(8 + (i * n / samples) * dim * sizeof(T));
        f.read((char *) row.data(), dim * sizeof(T));
        if (!f)
          throw std::runtime_error("Cannot read RaBitQ training vector");
        for (size_t j = 0; j < dim; ++j)
          training[i * dim + j] = static_cast<float>(row[j]);
      }
      std::vector<float> original_pivots(clusters * dim);
      LOG(INFO) << "Training RaBitQ" << Bits << ": " << samples << " samples, " << clusters << " centers";
      if (clusters == 1) {
        // One shared center is the mean of the complete input, not a sampled
        // k-means center. Stream the reduction without retaining raw vectors.
        std::vector<double> sums(dim, 0);
        std::vector<T> block(4096 * dim);
        f.clear();
        f.seekg(8);
        for (size_t start = 0; start < n; start += 4096) {
          const size_t count = std::min<size_t>(4096, n - start);
          f.read((char *) block.data(), count * dim * sizeof(T));
          if (!f)
            throw std::runtime_error("Cannot read single-center training data");
          for (size_t i = 0; i < count; ++i) {
            T *point = block.data() + i * dim;
            for (size_t j = 0; j < dim; ++j)
              sums[j] += double(point[j]);
          }
        }
        for (size_t j = 0; j < dim; ++j)
          original_pivots[j] = sums[j] / n;
      } else {
        kmeans::kmeanspp_selecting_pivots(training.data(), samples, dim, original_pivots.data(), clusters);
        kmeans::run_elkan(training.data(), samples, dim, original_pivots.data(), clusters, 15, nullptr, nullptr);
      }
      for (size_t c = 0; c < clusters; ++c) {
        float *p = distance.quantizer.pivots.data() + c * distance.quantizer.padded_dim;
        std::copy_n(original_pivots.data() + c * dim, dim, p);
        distance.quantizer.transform(p);
      }
      std::vector<uint32_t> assignments(samples);
      if (clusters > 1)
        kmeans::compute_closest_centers(training.data(), samples, dim, original_pivots.data(), clusters, 1,
                                        assignments.data());
      distance.quantizer.calibrate(training.data(), assignments.data(), samples);
    }
    void build(const std::string &prefix, const std::string &filename, uint32_t) override {
      if (!distance.quantizer.dim)
        train(filename);
      size_t n, dim;
      get_bin_metadata(filename, n, dim);
      if (dim != distance.quantizer.dim)
        throw std::invalid_argument("RaBitQ input dimension mismatch");
      this->npoints = n;
      this->data_size = distance.quantizer.vector_size();
      save_metadata(prefix.c_str());
      std::ifstream input(filename, std::ios::binary);
      input.seekg(8);
      std::ofstream output(prefix + "_processed.bin", std::ios::binary);
      const uint32_t header[] = {static_cast<uint32_t>(n), static_cast<uint32_t>(this->data_size)};
      output.write(reinterpret_cast<const char *>(header), sizeof(header));
      constexpr size_t block = 4096;
      std::vector<T> points(block * dim);
      std::vector<uint8_t> codes(block * this->data_size);
      for (size_t start = 0; start < n; start += block) {
        const size_t count = std::min(block, n - start);
        input.read(reinterpret_cast<char *>(points.data()), count * dim * sizeof(T));
        if (!input)
          throw std::runtime_error("Cannot read RaBitQ input vectors");
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i)
          distance.quantizer.encode(points.data() + i * dim, codes.data() + i * this->data_size);
        output.write(reinterpret_cast<const char *>(codes.data()), count * this->data_size);
      }
      if (!output)
        throw std::runtime_error("Cannot write RaBitQ encoded vectors");
    }
    void insert(const T *point, uint32_t loc) override {
      std::vector<uint8_t> code(this->data_size);
      distance.encode(point, code.data(), distance.quantizer.dim);
      std::unique_lock lock(codes_mutex_);
      const size_t required = (size_t(loc) + 1) * this->data_size;
      if (this->data_.size() < required)
        this->data_.resize(required);
      memcpy(this->at(loc), code.data(), this->data_size);
      this->npoints = std::max<uint64_t>(this->npoints, size_t(loc) + 1);
    }
    AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map, uint64_t count,
                                 uint32_t nthreads) override {
      std::shared_lock lock(codes_mutex_);
      auto result = std::make_unique<RaBitQNeighbor<T, Bits>>(this->metric);
      result->distance.quantizer = distance.quantizer;
      result->training_centers = training_centers;
      result->data_size = this->data_size;
      result->resize(count);
#pragma omp parallel for num_threads(nthreads)
      for (uint64_t i = 0; i < count; ++i)
        memcpy(result->at(i), this->at(rev_id_map.find(i)), this->data_size);
      return result.release();
    }
   private:
    // Reuse the trained quantizer when switching build inputs (e.g. shards).
    void load_records(const char *filename) {
      if (!distance.quantizer.dim)
        throw std::logic_error("RaBitQ codes require a trained quantizer");
      size_t n, data_size;
      load_bin(filename, this->data_, n, data_size);
      if (data_size != this->data_size)
        throw std::runtime_error("RaBitQ code size mismatch");
      this->npoints = n;
      for (size_t i = 0; i < n; ++i)
        if (RaBitQQuantizer<T, Bits>::header(this->data_.data() + i * data_size).pivot >= distance.quantizer.clusters)
          throw std::runtime_error("Invalid RaBitQ pivot id");
    }
  };
}  // namespace pipeann
