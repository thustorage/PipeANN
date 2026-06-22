#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <functional>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdlib.h>

// External libraries
#include <cblas.h>
#include "omp.h"

// Project headers
#include "utils.h"
#include "distance.h"
#include "filter/attribute.h"
#include "filter/selector.h"
#include "filter/dsl_compiler.h"

#define PARTSIZE 1000000
#define ALIGNMENT 512

// Filter function: (query_index, local_point_id) -> bool
using FilterFn = std::function<bool(uint64_t query_idx, uint32_t point_id)>;

// Pair for Priority Queue: <Distance, PointID>
using ResultPair = std::pair<float, uint32_t>;

// Structure for exact_knn
using pairIF = std::pair<int, float>;
struct cmpmaxstruct {
  bool operator()(const pairIF &l, const pairIF &r) {
    return l.second < r.second;
  };
};
using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

void exact_knn(const size_t dim, const size_t k,
               int *const closest_points,         // k * num_queries preallocated
               float *const dist_closest_points,  // k * num_queries preallocated
               size_t npoints, float *points_in, size_t nqueries, float *queries_in, pipeann::Metric metric,
               const FilterFn &filter) {
  pipeann::Distance<float> *distance = pipeann::get_distance_function<float>(metric);

  static constexpr uint64_t kMaxDistMatrixSize = 32ul * 1024 * 1024;  // max: 128MB float, larger triggers OpenBLAS bug.
  size_t q_batch_size = (kMaxDistMatrixSize / npoints) / 2 * 2;       // bulk compare fails at batch_size = 1.
  size_t dist_matrix_size = (size_t) q_batch_size * (size_t) npoints;
  float *dist_matrix = new float[dist_matrix_size];
  for (uint64_t b = 0; b < DIV_ROUND_UP(nqueries, q_batch_size); ++b) {
    int64_t q_b = b * q_batch_size;
    int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

    distance->bulk_compare(queries_in + q_b * dim, q_e - q_b, points_in, npoints, dim, dist_matrix);

#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t q = q_b; q < q_e; q++) {
      maxPQIFCS point_dist;
      for (uint64_t p = 0; p < npoints; p++) {
        if (!filter(q, p))
          continue;
        float dist = dist_matrix[(q - q_b) * npoints + p];
        if (point_dist.size() < k) {
          point_dist.emplace(p, dist);
        } else if (point_dist.top().second > dist) {
          point_dist.emplace(p, dist);
          point_dist.pop();
        }
      }
      // Write results (pad with -1 if fewer than k matches)
      size_t valid = point_dist.size();
      for (int64_t l = 0; l < (int64_t) k; ++l) {
        int64_t idx = (k - 1 - l) + q * k;
        if (l < (int64_t) valid) {
          closest_points[idx] = point_dist.top().first;
          dist_closest_points[idx] = point_dist.top().second;
          point_dist.pop();
        } else {
          closest_points[idx] = -1;
          dist_closest_points[idx] = std::numeric_limits<float>::max();
        }
      }
    }
    std::cerr << "Computed exact filtered k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
  }

  delete[] dist_matrix;
  delete distance;

  if (metric == pipeann::Metric::COSINE) {
    delete[] points_in;
    delete[] queries_in;
  }
}

template<typename T>
inline int get_num_parts(const char *filename) {
  std::ifstream reader(filename, std::ios::binary);
  std::cerr << "Reading bin file " << filename << " ...\n";
  int npts_i32, ndims_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &ndims_i32, sizeof(int));
  std::cerr << "Header: #pts = " << npts_i32 << ", #dims = " << ndims_i32 << std::endl;
  reader.close();
  uint32_t num_parts = (npts_i32 % PARTSIZE) == 0 ? (uint32_t) (npts_i32 / PARTSIZE)
                                                  : (uint32_t) std::floor((double) npts_i32 / (double) PARTSIZE) + 1;
  std::cerr << "Number of parts: " << num_parts << std::endl;
  return num_parts;
}

template<typename T>
inline void load_bin_as_float(const char *filename, float *&data, uint64_t &npts, uint64_t &ndims, int part_num) {
  std::ifstream reader(filename, std::ios::binary);
  std::cerr << "Reading bin file " << filename << " ...\n";
  int npts_i32, ndims_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &ndims_i32, sizeof(int));

  uint64_t start_id = (uint64_t) part_num * PARTSIZE;
  // Use uint64_t to avoid overflow before min
  uint64_t end_id = (std::min)((uint64_t) start_id + PARTSIZE, (uint64_t) npts_i32);
  npts = end_id - start_id;
  ndims = (unsigned) ndims_i32;
  std::cerr << "#pts in part = " << npts << ", #dims = " << ndims << ", size = " << npts * ndims * sizeof(T) << "B"
            << std::endl;

  // Safe Seek
  reader.seekg((std::streampos) start_id * ndims * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);

  T *data_T = new T[npts * ndims];
  reader.read((char *) data_T, sizeof(T) * npts * ndims);
  std::cerr << "Finished reading part of the bin file." << std::endl;
  reader.close();

  pipeann::alloc_aligned((void **) &data, npts * ndims * sizeof(float), ALIGNMENT);
#pragma omp parallel for schedule(dynamic, 32768)
  for (int64_t i = 0; i < (int64_t) npts; i++) {
    for (int64_t j = 0; j < (int64_t) ndims; j++) {
      float cur_val_float = (float) data_T[i * ndims + j];
      std::memcpy((char *) (data + i * ndims + j), (char *) &cur_val_float, sizeof(float));
    }
  }
  delete[] data_T;
  std::cerr << "Finished converting part data to float." << std::endl;
}

inline void save_groundtruth_as_one_file(const std::string filename, int32_t *data, float *distances, size_t npts,
                                         size_t ndims, uint32_t *tags = nullptr) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  std::cerr << "Saving truthset: " << npts << " queries, top-" << ndims << std::endl;

  writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  writer.write((char *) distances, npts * ndims * sizeof(float));
  if (tags != nullptr) {
    writer.write((char *) tags, npts * ndims * sizeof(uint32_t));
  } else {
    writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  }

  writer.close();
  std::cerr << "Finished writing truthset" << std::endl;
}

template<typename T>
int aux_main(int argc, char **argv, pipeann::Metric metric, const std::string &label_config_file) {
  uint64_t npoints, nqueries, dim;
  std::string base_file(argv[3]);
  std::string query_file(argv[4]);
  size_t k = atoi(argv[5]);
  std::string gt_file(argv[6]);

  float *base_data;
  float *query_data;

  int num_parts = get_num_parts<T>(base_file.c_str());
  load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);

  if (metric == pipeann::Metric::COSINE) {
    for (uint64_t i = 0; i < nqueries; i++) {
      pipeann::normalize_data(query_data + i * dim, query_data + i * dim, dim);
    }
  }

  // Read n_base_points from the base bin file header
  uint64_t n_base_points;
  {
    std::ifstream reader(base_file, std::ios::binary);
    int npts_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    n_base_points = npts_i32;
    reader.close();
  }

  // Load selector and labels from JSON config (unified SQL-like filter + bindings).
  pipeann::Selector *selector = nullptr;
  std::map<uint32_t, pipeann::AttrIndex *> base_stores;
  std::vector<pipeann::Attributes> query_attrs;

  if (!label_config_file.empty()) {
    std::cerr << "Loading label config from " << label_config_file << " with " << n_base_points << " base points"
              << std::endl;
    std::tie(selector, query_attrs, base_stores) = pipeann::dsl::load_filter_from_json(label_config_file, n_base_points);

    for (auto &[key, store] : base_stores) {
      store->load_attrs();
      std::cerr << "Loaded base label store key=" << key << std::endl;
    }
    std::cerr << "Loaded selector with " << query_attrs.size() << " query labels" << std::endl;
  }

  std::vector<std::vector<std::pair<uint32_t, float>>> results(nqueries);

  int *closest_points = new int[nqueries * k];
  float *dist_closest_points = new float[nqueries * k];

  for (int p = 0; p < num_parts; p++) {
    size_t start_id = p * PARTSIZE;
    load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p);
    if (metric == pipeann::Metric::COSINE) {
      for (uint64_t i = 0; i < npoints; i++) {
        pipeann::normalize_data(base_data + i * dim, base_data + i * dim, dim);
      }
    }

    // Build filter function: checks is_member for each (query, point) pair.
    // If no selector, all points pass.
    FilterFn filter;
    if (selector != nullptr) {
      filter = [&](uint64_t query_idx, uint32_t local_point_id) -> bool {
        uint32_t global_id = local_point_id + start_id;
        pipeann::Attributes target_attrs;
        for (const auto &[key, store] : base_stores) {
          target_attrs.set(key, store->get(global_id));
        }
        return selector->is_member(global_id, query_attrs[query_idx], target_attrs);
      };
    } else {
      filter = [](uint64_t, uint32_t) -> bool { return true; };
    }

    int *closest_points_part = new int[nqueries * k];
    float *dist_closest_points_part = new float[nqueries * k];

    exact_knn(dim, k, closest_points_part, dist_closest_points_part, npoints, base_data, nqueries, query_data, metric,
              filter);

    for (uint64_t i = 0; i < nqueries; i++) {
      for (uint64_t j = 0; j < k; j++) {
        if (closest_points_part[i * k + j] >= 0) {
          results[i].push_back(std::make_pair((uint32_t) (closest_points_part[i * k + j] + start_id),
                                              dist_closest_points_part[i * k + j]));
        }
      }
    }

    delete[] closest_points_part;
    delete[] dist_closest_points_part;
    pipeann::aligned_free(base_data);
  }

  // Sort and select top-k for each query
  for (uint64_t i = 0; i < nqueries; i++) {
    std::vector<std::pair<uint32_t, float>> &cur_res = results[i];
    std::sort(
        cur_res.begin(), cur_res.end(),
        [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) { return a.second < b.second; });

    size_t valid_count = std::min(k, cur_res.size());
    for (uint64_t j = 0; j < k; j++) {
      if (j < valid_count) {
        closest_points[i * k + j] = (int32_t) cur_res[j].first;
        if (metric == pipeann::Metric::INNER_PRODUCT)
          dist_closest_points[i * k + j] = -cur_res[j].second;
        else
          dist_closest_points[i * k + j] = cur_res[j].second;
      } else {
        closest_points[i * k + j] = -1;
        dist_closest_points[i * k + j] = std::numeric_limits<float>::max();
      }
    }

    if (valid_count < k) {
      std::cerr << "WARNING: Query " << i << " only found " << valid_count << " matching results (requested " << k
                << ")" << std::endl;
    }
  }

  uint32_t *tags = nullptr;
  if (std::string(argv[7]) != std::string("null")) {
    std::cerr << "Loading tags from " << argv[7] << "\n";
    tags = new uint32_t[nqueries * k];
    uint32_t *all_tags;
    std::string tag_file = std::string(argv[7]);
    size_t tag_pts, tag_dim;
    pipeann::load_bin(tag_file, all_tags, tag_pts, tag_dim);

    std::cerr << "Loaded tags for " << tag_pts << " points.\n";
    for (uint64_t i = 0; i < nqueries * k; i++) {
      if (closest_points[i] >= 0) {
        tags[i] = all_tags[closest_points[i]];
      } else {
        tags[i] = 0;
      }
    }
  }

  save_groundtruth_as_one_file(gt_file, closest_points, dist_closest_points, nqueries, k, tags);
  pipeann::aligned_free(query_data);
  delete[] closest_points;
  delete[] dist_closest_points;
  if (tags != nullptr) {
    delete[] tags;
  }

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0]
              << " <int8/uint8/float> <distance function (l2/cosine/mips)> <base bin file> <query bin file> <K> "
                 "<output-truthset-file> <tag_file (null if none)> [label_config.json]"
              << std::endl;
    return -1;
  }

  std::string data_type(argv[1]);
  std::string dist_fn(argv[2]);
  pipeann::Metric metric = pipeann::get_metric(dist_fn);

  std::string label_config_file;
  if (argc >= 9 && std::string(argv[8]) != "null") {
    label_config_file = argv[8];
  }

  if (data_type == std::string("float")) {
    return aux_main<float>(argc, argv, metric, label_config_file);
  } else if (data_type == std::string("int8")) {
    return aux_main<int8_t>(argc, argv, metric, label_config_file);
  } else if (data_type == std::string("uint8")) {
    return aux_main<uint8_t>(argc, argv, metric, label_config_file);
  } else {
    std::cerr << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
    return -1;
  }
}
