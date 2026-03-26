#pragma once
#include <fcntl.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <malloc.h>
#include <unistd.h>
#include <sys/stat.h>
#include <immintrin.h>

#include "utils/log.h"

/*
 * Necessary definitions and
 * functions used both by both in-memory and on-disk index.
 * Avoid including other headers in this project!
 */
constexpr uint64_t SECTOR_LEN = 4096;

#define ROUND_UP(X, Y) ((((uint64_t) (X) / (Y)) + ((uint64_t) (X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t) (X) / (Y)) + ((uint64_t) (X) % (Y) != 0))

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/* Used to replace exception. */
inline void crash() {
  // #ifndef NDEBUG
  __builtin_trap();
  // #endif
}

static inline bool file_exists(const std::string &name, bool dirCheck = false) {
  int val;
  struct stat buffer;
  val = stat(name.c_str(), &buffer);

  DLOG(INFO) << " Stat(" << name.c_str() << ") returned: " << val;
  if (val != 0) {
    switch (errno) {
      case EINVAL:
        DLOG(INFO) << "Invalid argument passed to stat()";
        break;
      case ENOENT:
        DLOG(INFO) << "File " << name.c_str() << " does not exist";
        break;
      default:
        DLOG(INFO) << "Unexpected error in stat():" << errno;
        break;
    }
    return false;
  } else {
    // the file entry exists. If reqd, check if this is a directory.
    return dirCheck ? buffer.st_mode & S_IFDIR : true;
  }
}

inline void open_file_to_write(std::ofstream &writer, const std::string &filename) {
  writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  if (!file_exists(filename))
    writer.open(filename, std::ios::binary | std::ios::out);
  else
    writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);
  if (writer.fail()) {
    LOG(ERROR) << std::string("Failed to open file") + filename + " for write because " << std::strerror(errno);
    crash();
  }
}

inline uint64_t get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    uint64_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    LOG(ERROR) << "Could not open file: " << fname;
    return 0;
  }
}

inline int delete_file(const std::string &fileName) {
  if (file_exists(fileName)) {
    auto rc = ::remove(fileName.c_str());
    if (rc != 0) {
      LOG(ERROR) << "Could not delete file: " << fileName
                 << " even though it exists. This might indicate a permissions issue. "
                    "If you see this message, please contact the pipeann team.";
    }
    return rc;
  } else {
    return 0;
  }
}

namespace pipeann {
  enum Metric { L2 = 0, INNER_PRODUCT = 1, COSINE = 2 };

  inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
    *ptr = ::aligned_alloc(align, size);
    assert(*ptr != nullptr);
  }

  inline void aligned_free(void *ptr) {
    if (ptr == nullptr) {
      return;
    }
    free(ptr);
  }

  inline void get_bin_metadata(const std::string &bin_file, size_t &nrows, size_t &ncols, size_t offset = 0) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    int nrows_32, ncols_32;
    reader.seekg(offset, reader.beg);
    reader.read((char *) &nrows_32, sizeof(int));
    reader.read((char *) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
  }

  template<typename T>
  inline void load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim,
                            size_t file_offset = 0) {
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    DLOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    data = new T[npts * dim];
    reader.read((char *) data, npts * dim * sizeof(T));
  }

  template<typename T>
  inline void load_bin_impl(std::basic_istream<char> &reader, std::vector<T> &data, size_t &npts, size_t &dim,
                            size_t file_offset = 0) {
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    DLOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    data.resize(npts * dim);
    reader.read((char *) (data.data()), npts * dim * sizeof(T));
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0) {
    DLOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);

    load_bin_impl<T>(reader, data, npts, dim, offset);
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, std::vector<T> &data, size_t &npts, size_t &dim,
                       size_t offset = 0) {
    DLOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);

    load_bin_impl<T>(reader, data, npts, dim, offset);
  }

  inline void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim,
                            uint32_t **tags = nullptr) {
    std::ifstream reader(bin_file, std::ios::binary);
    DLOG(INFO) << "Reading truthset file " << bin_file.c_str() << "...";
    size_t actual_file_size = get_file_size(bin_file);

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    DLOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    size_t with_tags_actual_file_size = 3 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
      truthset_type = 2;

    if (actual_file_size == with_tags_actual_file_size)
      truthset_type = 3;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
             << expected_file_size_just_ids;
      LOG(INFO) << stream.str();
      crash();
    }

    ids = new uint32_t[npts * dim];
    reader.read((char *) ids, npts * dim * sizeof(uint32_t));

    if ((truthset_type == 1) || (truthset_type == 3)) {
      dists = new float[npts * dim];
      reader.read((char *) dists, npts * dim * sizeof(float));
    }
    if (truthset_type == 3) {
      *tags = new uint32_t[npts * dim];
      reader.read((char *) *tags, npts * dim * sizeof(uint32_t));
    }
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, std::unique_ptr<T[]> &data, size_t &npts, size_t &dim,
                       size_t offset = 0) {
    T *ptr;
    load_bin<T>(bin_file, ptr, npts, dim, offset);
    data.reset(ptr);
  }

  template<typename T>
  inline uint64_t save_bin(const std::string &filename, T *data, size_t npts, size_t ndims, size_t offset = 0) {
    std::ofstream writer;
    open_file_to_write(writer, filename);

    DLOG(INFO) << "Writing bin: " << filename.c_str();
    writer.seekp(offset, writer.beg);
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    DLOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

    writer.write((char *) data, npts * ndims * sizeof(T));
    writer.close();
    DLOG(INFO) << "Finished writing bin.";
    return bytes_written;
  }

  template<typename InType, typename OutType>
  void convert_types(const InType *srcmat, OutType *destmat, size_t npts, size_t dim) {
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t) npts; i++) {
      for (uint64_t j = 0; j < dim; j++) {
        destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
      }
    }
  }

  // NOTE: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char *) vec + d, _MM_HINT_T0);
  }

  inline double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                                 unsigned *our_results, unsigned dim_or, unsigned recall_at) {
    double total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        float *gt_dist_vec = gs_dist + dim_gs * i;
        tie_breaker = recall_at - 1;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);

      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  struct PiPNNConfig {
    int cmax = 2048;
    int cmin = 256;
    float p_samp = 0.01f;
    int pmax = 1000;
    int first_layer_fanout = 6;
    int second_layer_fanout = 10;
    int k = 2;                    // k-NN in leaf build
    int num_hyperplanes = 12;
    int climit = 128;
    uint32_t seed = 42;

    void print() {
      LOG(INFO) << "PiPNNConfig: cmax=" << cmax << " cmin=" << cmin << " p_samp=" << p_samp << " pmax=" << pmax
                << " first_layer_fanout=" << first_layer_fanout << " second_layer_fanout=" << second_layer_fanout
                << " k=" << k << " num_hyperplanes=" << num_hyperplanes << " climit=" << climit << " seed=" << seed;
    }
  };

  /* Parameters for index build. */
  struct IndexBuildParameters {
    uint32_t R = 0;              // maximum out-neighbors.
    uint32_t L = 0;              // build L.
    uint32_t C = 384;            // delete pruning capacity.
    float alpha = 1.2f;          // alpha for Vamana.
    uint32_t num_threads = 0;    // num_threads used.
    bool saturate_graph = true;  // saturate graph during build (using kNN neighbors).
    uint32_t beam_width = 4;     // insert beam width. (for SSD)
    bool use_pipnn = false;
    PiPNNConfig pipnn_config;

    void set(uint32_t R, uint32_t L, uint32_t C, float alpha = 1.2, uint32_t num_threads = 0,
             bool saturate_graph = true, uint32_t beam_width = 4) {
      this->R = R;
      this->L = L;
      this->C = C;
      this->alpha = alpha;
      this->num_threads = num_threads;
      this->beam_width = beam_width;
      this->saturate_graph = saturate_graph;
    }

    void pipnn_set(int cmax, int cmin, float p_samp, int pmax, int first_layer_fanout, int second_layer_fanout,
                   int k, int num_hyperplanes, int climit, uint32_t seed = 42) {
      this->use_pipnn = true;
      this->pipnn_config = {cmax, cmin, p_samp, pmax, first_layer_fanout, second_layer_fanout,
                            k, num_hyperplanes, climit, seed};
    }

    void print() {
      LOG(INFO) << "IndexBuildParameters with L: " << L << " R: " << R << " C: " << C << " alpha: " << alpha
                << " num_threads: " << num_threads << " beam_width: " << beam_width
                << " saturate_graph: " << saturate_graph;
      if (use_pipnn) {
        pipnn_config.print();
      }
    }
  };

  /* Graph neighbor-related definitions. */
  struct Neighbor {
    unsigned id;
    float distance;
    bool flag;
    bool visited;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f), visited(false) {
    }

    inline bool operator<(const Neighbor &other) const {
      return (distance < other.distance) || (distance == other.distance && id < other.id);
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
    inline bool operator>(const Neighbor &other) const {
      return (distance > other.distance) || (distance == other.distance && id > other.id);
    }
  };

  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }
    // check equal ID

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }

  /*
   * For float, we normalize to 1.
   * For non-float types, we do not normalize to 1 to avoid underflow.
   * Instead, we normalize to TYPE_MAX:
   * - For uint8_t, we normalize to 255.
   * - For int8_t, we normalize to 127.
   */
  template<typename T>
  inline void normalize_data(T *data_out, const T *data_in, size_t dim) {
    float norm_sq = 0.0f;

#pragma omp simd reduction(+ : norm_sq) aligned(data_in : 8)
    for (size_t i = 0; i < dim; i++) {
      norm_sq += (float) (data_in[i]) * (float) (data_in[i]);
    }

    if (unlikely(norm_sq < FLT_EPSILON)) {
      memset(data_out, 0, dim * sizeof(T));
      return;
    }

    float norm = std::sqrt(norm_sq);
    float scale = 1.0f;

    if constexpr (std::is_same_v<T, uint8_t>) {
      scale = 255.0f / norm;
    } else if constexpr (std::is_same_v<T, int8_t>) {
      scale = 127.0f / norm;
    } else if constexpr (std::is_same_v<T, float>) {
      scale = 1.0f / norm;
    } else {
      LOG(ERROR) << "Unsupported type: " << typeid(T).name();
      crash();
    }

    // Round up for integer types.
#pragma omp simd aligned(data_out, data_in : 8)
    for (size_t i = 0; i < dim; i++) {
      float val = (float) data_in[i] * scale;
      if constexpr (std::is_same_v<T, float>) {
        data_out[i] = val;
      } else {
        data_out[i] = static_cast<T>(std::round(val));
      }
    }
  }

  // We reserve kInvalidID to mark an invalid ID or location.
  static constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();
}  // namespace pipeann
