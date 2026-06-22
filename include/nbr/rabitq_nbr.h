#pragma once

#include "nbr/abstract_nbr.h"
#include "ssd_index_defs.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "distance.h"
#include <immintrin.h>
#include <vector>
#include "utils/partition.h"
#include "utils/kmeans_utils.h"

#ifdef USE_AVX512
#include "rabitq/utils/data_layout.hpp"
#include "rabitq/utils/defines.hpp"
#include "rabitq/rotator.hpp"
#include "rabitq/rabitq.hpp"
#include "rabitq/utils/space.hpp"
#include "rabitq/utils/warmup_space.hpp"

namespace pipeann {
  // 1-bit and multi-bit implementation of RaBitQ. The query is quantized to 4 bits.
  // Update is not supported yet, due to difficult symmetric distance computation.
  template<typename T, size_t BITS = 1>
  class RaBitQNeighbor : public AbstractNeighbor<T> {
    static_assert(BITS >= 1, "BITS must be >= 1");
    static_assert(BITS <= 8, "BITS must be <= 8");
    static constexpr uint32_t NUM_KMEANS = 15;
    static constexpr uint32_t NUM_CLUSTERS = 16;  // recommended by RaBitQ library.
    static constexpr uint32_t QUERY_BITS = 4;     // 4 bits for each query dimension.
    static constexpr size_t EX_BITS = (BITS > 1 ? BITS - 1 : 0);
    static constexpr bool HAS_EX = (EX_BITS > 0);
    static constexpr size_t META_FLOATS = HAS_EX ? 4 : 3;
    using pivot_id_t = uint8_t;

    uint64_t data_dim = 0, pad_dim = 0;
    uint64_t bin_data_size = 0, ex_data_size = 0, data_size = 0;

    enum { M_DELTA = 0, M_VL = 1, M_K1XSUMQ = 2, M_KBXSUMQ = 3 };  // offsets in query context.

    // Float only, use fhtkac rotator.
    rabitqlib::rotator_impl::FhtKacRotator *rotator_ = nullptr;
    float *rotated_pivots = nullptr;

    // Each element: [pivot_id | bin_data]
    std::vector<char> data;
    Distance<float> *distance;

    rabitqlib::MetricType rabitq_metric = rabitqlib::MetricType::METRIC_L2;
    rabitqlib::ex_ipfunc ex_ip_func_ = nullptr;

   public:
    RaBitQNeighbor<T, BITS>(pipeann::Metric metric) : AbstractNeighbor<T>(metric) {
      distance = get_distance_function<float>(metric);
      rabitq_metric = metric == INNER_PRODUCT ? rabitqlib::MetricType::METRIC_IP : rabitqlib::MetricType::METRIC_L2;
      if constexpr (HAS_EX) {
        ex_ip_func_ = rabitqlib::select_excode_ipfunc(EX_BITS);
      }
    }

    ~RaBitQNeighbor<T, BITS>() {
      if (rotator_ != nullptr) {
        delete rotator_;
        rotator_ = nullptr;
      }
      if (rotated_pivots != nullptr) {
        delete[] rotated_pivots;
        rotated_pivots = nullptr;
      }
    }

    // max size of context needed for a single query.
    uint64_t query_ctx_size() {
      if (unlikely(this->pad_dim == 0)) {
        LOG(ERROR) << "Please load neighbor first!";
        exit(-1);
      }
      /* quantized_vec | q_to_centroids | delta | vl | k1xsumq | kbxsumq | rotated_query */
      uint64_t ctx_bytes =
          this->pad_dim * QUERY_BITS / 8 + NUM_CLUSTERS * 2 * sizeof(float) + META_FLOATS * sizeof(float);
      if constexpr (HAS_EX) {
        ctx_bytes += this->pad_dim * sizeof(float);
      }
      return ctx_bytes;
    }

    std::string get_name() {
      return "RaBitQNeighbor";
    }
    // rev_id_map: new_id -> old_id.
    AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map, uint64_t new_npoints,
                                 uint32_t nthreads) {
      LOG(ERROR) << "Update is not supported by RaBitQNeighbor.";
      exit(-1);
      return this;
    }

    void initialize_query(const T *query, QueryBuffer *query_buf) {
      std::vector<float> query_float(data_dim);
      std::vector<float> rotated_query_float(pad_dim);
      for (uint64_t i = 0; i < data_dim; ++i) {
        query_float[i] = (float) query[i];
      }
      this->rotator_->rotate(query_float.data(), rotated_query_float.data());

      // quantize the vector.
      std::vector<uint16_t> quant_query(pad_dim);
      float delta_ = 0, vl_ = 0;
      rabitqlib::quant::quantize_scalar<float, uint16_t>(rotated_query_float.data(), pad_dim, QUERY_BITS,
                                                         quant_query.data(), delta_, vl_);
      uint64_t *query_bin = (uint64_t *) (query_buf->nbr_ctx_scratch);
      rabitqlib::new_transpose_bin(quant_query.data(), query_bin, pad_dim, QUERY_BITS);

      float *q_to_centroids = get_q_to_centroids(query_buf->nbr_ctx_scratch);
      if (this->metric == Metric::INNER_PRODUCT) {
        for (size_t i = 0; i < NUM_CLUSTERS; i++) {
          q_to_centroids[i] = rabitqlib::dot_product(rotated_query_float.data(), rotated_pivots + i * pad_dim, pad_dim);
          q_to_centroids[i + NUM_CLUSTERS] =
              std::sqrt(rabitqlib::euclidean_sqr(rotated_query_float.data(), rotated_pivots + i * pad_dim, pad_dim));
        }
      } else {
        for (size_t i = 0; i < NUM_CLUSTERS; ++i) {
          q_to_centroids[i] =
              std::sqrt(distance->compare(rotated_query_float.data(), rotated_pivots + i * pad_dim, pad_dim));
        }
      }
      float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
      float sumq = std::accumulate(rotated_query_float.data(), rotated_query_float.data() + pad_dim, 0.0f);

      float *meta = get_meta(query_buf->nbr_ctx_scratch);
      meta[M_DELTA] = delta_;
      meta[M_VL] = vl_;
      meta[M_K1XSUMQ] = c_1 * sumq;
      if constexpr (HAS_EX) {
        float c_b = -static_cast<float>((1 << (EX_BITS + 1)) - 1) / 2.F;
        meta[M_KBXSUMQ] = c_b * sumq;
        float *rotated_query_out = get_rotated_query(query_buf->nbr_ctx_scratch);
        std::memcpy(rotated_query_out, rotated_query_float.data(), pad_dim * sizeof(float));
      }
    }

    // Compute dists using assymetric distance computation.
    void compute_dists(QueryBuffer *query_buf, const uint32_t *ids, const uint64_t n_ids) {
      uint64_t *query_bin = (uint64_t *) (query_buf->nbr_ctx_scratch);
      float *q_to_centroids = get_q_to_centroids(query_buf->nbr_ctx_scratch);
      float *meta = get_meta(query_buf->nbr_ctx_scratch);
      float *rotated_query = HAS_EX ? get_rotated_query(query_buf->nbr_ctx_scratch) : nullptr;

      for (uint64_t i = 0; i < n_ids; ++i) {
        if ((i + 1) < n_ids) {
          uint64_t next_id = ids[i + 1];
          pipeann::prefetch_vector(data.data() + data_size * next_id, data_size);
        }
        char *item = data.data() + data_size * ids[i];

        pivot_id_t pivot_id = 0;
        memcpy(&pivot_id, item, sizeof(pivot_id_t));
        char *raw_bin_data = item + sizeof(pivot_id_t);
        memcpy(query_buf->nbr_vec_scratch, raw_bin_data, bin_data_size + ex_data_size);
        char *bin_data = (char *) query_buf->nbr_vec_scratch;  // for alignment.
        char *ex_data = HAS_EX ? (bin_data + bin_data_size) : nullptr;

        if (this->metric == Metric::INNER_PRODUCT) {
          float norm = q_to_centroids[pivot_id];
          float error = q_to_centroids[pivot_id + NUM_CLUSTERS];
          if constexpr (HAS_EX) {
            query_buf->aligned_dist_scratch[i] =
                split_single_ex_estdist(bin_data, ex_data, query_bin, meta, rotated_query, -norm, error);
          } else {
            query_buf->aligned_dist_scratch[i] = split_single_estdist(bin_data, query_bin, meta, -norm, error);
          }
        } else {
          // L2 distance
          float norm = q_to_centroids[pivot_id];
          if constexpr (HAS_EX) {
            query_buf->aligned_dist_scratch[i] =
                split_single_ex_estdist(bin_data, ex_data, query_bin, meta, rotated_query, norm * norm, norm);
          } else {
            query_buf->aligned_dist_scratch[i] = split_single_estdist(bin_data, query_bin, meta, norm * norm, norm);
          }
        }
      }
    }
    // Compute dists using PQ all-to-all.
    void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *aligned_scratch) {
      LOG(ERROR) << "Update (symmetric distance computation) is not supported by RaBitQNeighbor.";
      exit(-1);
    }
    // Load the neighbor data (e.g., PQ) from disk.
    void load(const char *index_prefix) {
      std::string meta_path = index_prefix + std::string("_rabitq_metadata.bin");
      std::string rotator_path = index_prefix + std::string("_rabitq_rotator.bin");
      std::string pivot_path = index_prefix + std::string("_rabitq_pivots.bin");
      std::string data_path = index_prefix + std::string("_rabitq_compressed.bin");

      std::vector<uint64_t> metadata;
      uint64_t npts = 0, dim = 0;
      pipeann::load_bin(meta_path, metadata, npts, dim);
      if (metadata.size() != 4) {
        LOG(ERROR) << "Metadata file is corrupted: " << meta_path;
        exit(-1);
      }

      this->npoints = metadata[0];
      this->data_dim = metadata[1];
      this->pad_dim = metadata[2];
      this->data_size = metadata[3];
      this->ex_data_size = HAS_EX ? rabitqlib::ExDataMap<float>::data_bytes(this->pad_dim, EX_BITS) : 0;
      this->bin_data_size = this->data_size - sizeof(pivot_id_t) - this->ex_data_size;
      if (this->bin_data_size == 0 || this->data_size <= sizeof(pivot_id_t)) {
        LOG(ERROR) << "Invalid data size in metadata.";
        exit(-1);
      }

      this->initialize_rotator(this->data_dim);
      std::ifstream rotator_if(rotator_path, std::ios::binary);
      rotator_->load(rotator_if);
      rotator_if.close();

      pipeann::load_bin(pivot_path, rotated_pivots, npts, dim);
      if (dim != this->pad_dim) {
        LOG(ERROR) << "Pivot dimension does not match: " << dim << " vs. " << this->pad_dim;
        exit(-1);
      }

      pipeann::load_bin(data_path, this->data, npts, dim);
      if (this->data_size != dim) {
        LOG(ERROR) << "Data size does not match: " << this->data_size << " vs. " << metadata[3];
        exit(-1);
      }
      if (this->npoints != npts) {
        LOG(ERROR) << "Number of points does not match: " << npts << " vs. " << metadata[0];
        exit(-1);
      }
      this->npoints = npts;
    }

    // Save the neighbor data (e.g., PQ) to disk.
    void save(const char *index_prefix) {
      std::string rotator_path = index_prefix + std::string("_rabitq_rotator.bin");
      std::string pivot_path = index_prefix + std::string("_rabitq_pivots.bin");
      std::string data_path = index_prefix + std::string("_rabitq_compressed.bin");
      std::string meta_path = index_prefix + std::string("_rabitq_metadata.bin");

      std::ofstream rotator_of(rotator_path, std::ios::binary);
      rotator_->save(rotator_of);
      rotator_of.close();
      pipeann::save_bin(pivot_path, rotated_pivots, NUM_CLUSTERS, this->pad_dim);
      pipeann::save_bin(data_path, this->data.data(), this->npoints, this->data_size);

      // save metadata.
      std::vector<uint64_t> metadata = {this->npoints, this->data_dim, this->pad_dim, this->data_size};
      pipeann::save_bin(meta_path, metadata.data(), 1, metadata.size());
    }

    // Call load after build to load the neighbors.
    void build(const std::string &index_prefix, const std::string &data_bin, uint32_t bytes_per_nbr) {
      // 1. initialize metadata.
      size_t num_points, dim;
      pipeann::get_bin_metadata(data_bin, num_points, dim);
      this->npoints = num_points;
      this->initialize_rotator(dim);  // initialize rotator and pad_dim.

      // quantize_split_single uses float, drop f_error as we do not use it.
      bin_data_size = rabitqlib::BinDataMap<float>::data_bytes(this->pad_dim) - sizeof(float);
      ex_data_size = HAS_EX ? rabitqlib::ExDataMap<float>::data_bytes(this->pad_dim, EX_BITS) : 0;
      bytes_per_nbr = bin_data_size + ex_data_size + sizeof(pivot_id_t);
      LOG(INFO) << "Bytes per neighbor: " << bytes_per_nbr << ", use " << BITS << "-bit RaBitQ.";
      this->data_size = bytes_per_nbr;

      // 2. get pivots.
      size_t train_size, train_dim;
      float *train_data;  // maximum: 256000 * dim * data_size, 1GB for 1024-dim float vector.

      auto start = std::chrono::high_resolution_clock::now();
      double p_val = this->get_sample_p();
      // generates random sample and sets it to train_data and updates train_size
      gen_random_slice<T>(data_bin, p_val, train_data, train_size, train_dim);

      LOG(INFO) << "Running k-means clustering for " << train_size << " samples, to get pivots for quantization...";
      auto pivots = generate_pivots(train_data, train_size, train_dim, NUM_CLUSTERS, NUM_KMEANS);
      delete[] train_data;
      train_data = nullptr;

      // rotate pivots.
      pipeann::alloc_aligned((void **) &rotated_pivots, NUM_CLUSTERS * this->pad_dim * sizeof(float), 64);
      for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
        this->rotator_->rotate(pivots.data() + (i * train_dim), rotated_pivots + (i * this->pad_dim));
      }

      // 3. Get cluster ID for each vector, and then quantize it.
      data.resize(this->npoints * this->data_size);
      LOG(INFO) << "Quantizing " << this->npoints << " vectors...";

      // quantize vectors.
      uint64_t read_blk_size = 64 * 1024 * 1024;
      std::ifstream base_reader(data_bin, std::ios::binary);
      base_reader.seekg(sizeof(uint32_t) * 2);  // Skip header

      size_t BLOCK_SIZE = std::min((size_t) 16384, num_points);  // hard-coded max block size.

      std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(BLOCK_SIZE * dim);
      std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(BLOCK_SIZE * dim);
      std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(BLOCK_SIZE);

      size_t num_blocks = DIV_ROUND_UP(num_points, BLOCK_SIZE);

      for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * BLOCK_SIZE;
        size_t end_id = std::min((block + 1) * BLOCK_SIZE, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *) (block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
        pipeann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

        kmeans::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots.data(), NUM_CLUSTERS, 1,
                                        closest_center.get());

#pragma omp parallel for
        for (int64_t j = 0; j < (int64_t) cur_blk_size; j++) {
          uint32_t id = start_id + j;
          uint32_t cluster_id = closest_center[j];
          quantize_single(block_data_float.get() + j * dim, cluster_id, this->data.data() + id * this->data_size);
        }
      }

      // save rotator, pivots, and quantized vectors.
      this->save(index_prefix.c_str());
    }

    void insert(T *point, uint32_t loc) {
      LOG(ERROR) << "Update is not supported by RaBitQNeighbor.";
      exit(-1);
    }

    void clear() override {
      data.clear();
      data.shrink_to_fit();
      if (rotated_pivots != nullptr) {
        delete[] rotated_pivots;
        rotated_pivots = nullptr;
      }
      this->npoints = 0;
    }

   private:
    void initialize_rotator(uint64_t data_dim) {
      this->data_dim = data_dim;
      this->pad_dim = ROUND_UP(data_dim, 64);
      // Use float only.
      rotator_ = new rabitqlib::rotator_impl::FhtKacRotator(data_dim, pad_dim);
    }

    float *get_q_to_centroids(float *query_ctx) {
      char *c_ptr = ((char *) query_ctx) + (pad_dim * QUERY_BITS / 8);
      return (float *) c_ptr;
    }

    float *get_meta(float *query_ctx) {
      char *c_ptr = ((char *) query_ctx) + (pad_dim * QUERY_BITS / 8) + NUM_CLUSTERS * 2 * sizeof(float);
      return (float *) c_ptr;
    }

    float *get_rotated_query(float *query_ctx) {
      char *c_ptr = ((char *) query_ctx) + (pad_dim * QUERY_BITS / 8) + NUM_CLUSTERS * 2 * sizeof(float) +
                    META_FLOATS * sizeof(float);
      return (float *) c_ptr;
    }

    void quantize_single(float *point, pivot_id_t pivot_id, char *out) {
      if (unlikely(pivot_id >= NUM_CLUSTERS)) {
        LOG(ERROR) << "pivot_id " << (int) pivot_id << " >= NUM_CLUSTERS " << NUM_CLUSTERS;
        exit(-1);
      }

      std::vector<float> rotated_data(this->pad_dim);
      rotator_->rotate(point, rotated_data.data());
      size_t full_bin_bytes = rabitqlib::BinDataMap<float>::data_bytes(this->pad_dim);
      size_t tmp_bytes = full_bin_bytes + ex_data_size;
      std::vector<uint64_t> quant_data(DIV_ROUND_UP(tmp_bytes, sizeof(uint64_t)));  // keep 8B alignment
      rabitqlib::quant::quantize_split_single(rotated_data.data(), rotated_pivots + (pivot_id * this->pad_dim),
                                              this->pad_dim, EX_BITS, (char *) quant_data.data(),
                                              (char *) quant_data.data() + full_bin_bytes, rabitq_metric);
      memcpy(out, &pivot_id, sizeof(pivot_id_t));
      memcpy(out + sizeof(pivot_id_t), quant_data.data(), bin_data_size);  // Do not save f_error.
      if constexpr (HAS_EX) {
        memcpy(out + sizeof(pivot_id_t) + bin_data_size, (char *) quant_data.data() + full_bin_bytes, ex_data_size);
      }
    }

    std::vector<float> generate_pivots(const float *train_data, size_t num_train, unsigned dim, unsigned num_centers,
                                       unsigned max_k_means_reps) {
      std::vector<float> pivot_data(num_centers * dim);
      kmeans::kmeanspp_selecting_pivots(const_cast<float *>(train_data), num_train, dim, pivot_data.data(),
                                        num_centers);
      kmeans::run_elkan(const_cast<float *>(train_data), num_train, dim, pivot_data.data(), num_centers,
                        max_k_means_reps, nullptr, nullptr);
      return pivot_data;
    }

    inline float split_single_estdist(const char *bin_data, uint64_t *query_bin, float *meta, float g_add = 0,
                                      float g_error = 0) {
      rabitqlib::ConstBinDataMap<float> cur_bin(bin_data, pad_dim);
      float ip_x0_qr =
          warmup_ip_x0_q<QUERY_BITS>(cur_bin.bin_code(), query_bin, meta[M_DELTA], meta[M_VL], pad_dim, QUERY_BITS);
      return cur_bin.f_add() + g_add + cur_bin.f_rescale() * (ip_x0_qr + meta[M_K1XSUMQ]);
    };

    inline float split_single_ex_estdist(const char *bin_data, const char *ex_data, uint64_t *query_bin, float *meta,
                                         const float *rotated_query, float g_add = 0, float g_error = 0) {
      rabitqlib::ConstBinDataMap<float> cur_bin(bin_data, pad_dim);
      float ip_x0_qr =
          warmup_ip_x0_q<QUERY_BITS>(cur_bin.bin_code(), query_bin, meta[M_DELTA], meta[M_VL], pad_dim, QUERY_BITS);
      rabitqlib::ConstExDataMap<float> cur_ex(ex_data, pad_dim, EX_BITS);
      return cur_ex.f_add_ex() + g_add +
             cur_ex.f_rescale_ex() * (static_cast<float>(1 << EX_BITS) * ip_x0_qr +
                                      ex_ip_func_(rotated_query, cur_ex.ex_code(), pad_dim) + meta[M_KBXSUMQ]);
    };
  };
}  // namespace pipeann
#else
namespace pipeann {
  template<typename T, size_t BITS = 1>
  class RaBitQNeighbor : public AbstractNeighbor<T> {
   public:
    RaBitQNeighbor(pipeann::Metric metric) : AbstractNeighbor<T>(metric) {
      LOG(ERROR) << "RaBitQNeighbor requires AVX512 support.";
      exit(-1);
    }

    std::string get_name() {
      return "RaBitQNeighbor";
    }
    // rev_id_map: new_id -> old_id.
    AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map, uint64_t new_npoints,
                                 uint32_t nthreads) {
      return this;
    }
    void initialize_query(const T *query, QueryBuffer *query_buf) {
    }
    // Compute dists using assymetric distance computation.
    void compute_dists(QueryBuffer *query_buf, const uint32_t *ids, const uint64_t n_ids) {
    }
    // Compute dists using PQ all-to-all.
    void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *aligned_scratch) {
    }
    // Load the neighbor data (e.g., PQ) from disk.
    void load(const char *index_prefix) {
    }
    // Save the neighbor data (e.g., PQ) to disk.
    void save(const char *index_prefix) {
    }
    // Call load after build to load the neighbors.
    void build(const std::string &index_prefix, const std::string &data_bin, uint32_t bytes_per_nbr) {
    }
    void insert(T *point, uint32_t loc) {
    }

    uint64_t npoints = 0;
  };
}  // namespace pipeann
#endif
