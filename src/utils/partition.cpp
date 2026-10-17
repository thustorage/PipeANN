#include <utils/kmeans_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <unordered_set>

#include "utils/cached_io.h"
#include "index.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include "utils/tsl/robin_map.h"

#include <cassert>
#include "utils/partition.h"
#include "nbr/rabitq_quantizer.h"
#include <filesystem>

static constexpr uint64_t MAX_BLOCK_SIZE = 16384;  // 64MB for 1024-dim float vectors, 2MB for 128-dim uint8 vectors.

template<typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate,
                      size_t offset) {
  std::ifstream base_reader(base_file.c_str());
  base_reader.seekg(offset, std::ios::beg);

  std::ofstream sample_writer(std::string(output_prefix + "_data.bin").c_str(), std::ios::binary);
  std::ofstream sample_id_writer(std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  auto x = rd();
  std::mt19937 generator(x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  size_t npts, nd;
  uint32_t npts_u32, nd_u32;
  uint32_t num_sampled_pts_u32 = 0;
  uint32_t one_const = 1;

  base_reader.read((char *) &npts_u32, sizeof(uint32_t));
  base_reader.read((char *) &nd_u32, sizeof(uint32_t));
  LOG(INFO) << "Loading base " << base_file << ". #points: " << npts_u32 << ". #dim: " << nd_u32 << ".";
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.write((char *) &nd_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &one_const, sizeof(uint32_t));

  npts = npts_u32;
  nd = nd_u32;
  std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

  for (size_t i = 0; i < npts; i++) {
    float sample = distribution(generator);
    if (sample < (float) sampling_rate) {
      base_reader.read((char *) cur_row.get(), sizeof(T) * nd);
      sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
      uint32_t cur_i_u32 = (uint32_t) i;
      sample_id_writer.write((char *) &cur_i_u32, sizeof(uint32_t));
      num_sampled_pts_u32++;
    } else {
      base_reader.seekg(sizeof(T) * nd, base_reader.cur);  // skip this vector
    }
  }

  if (num_sampled_pts_u32 == 0) {
    // We have read something from file, so write it.
    sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
    num_sampled_pts_u32 = 1;
  }
  sample_writer.seekp(0, std::ios::beg);
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.seekp(0, std::ios::beg);
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.close();
  sample_id_writer.close();
  LOG(INFO) << "Wrote " << num_sampled_pts_u32 << " points to sample file: " << output_prefix + "_data.bin";
}

// Sorted, unique IDs keep sampling reads sequential without scanning the whole file.
static std::vector<size_t> sample_ids(size_t count, double rate) {
  const size_t n = std::max<size_t>(1, std::min<size_t>(count, std::ceil(count * std::clamp(rate, 0.0, 1.0))));
  std::mt19937 rng(std::random_device{}());
  std::unordered_set<size_t> selected;
  selected.reserve(n);
  for (size_t j = count - n; j < count; ++j) {
    const size_t id = std::uniform_int_distribution<size_t>(0, j)(rng);
    if (!selected.insert(id).second) selected.insert(j);
  }
  std::vector<size_t> ids(selected.begin(), selected.end());
  std::sort(ids.begin(), ids.end());
  return ids;
}

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims) {
  std::ifstream input(data_file, std::ios::binary);
  uint32_t npts = 0, dim = 0;
  input.read(reinterpret_cast<char *>(&npts), sizeof(npts));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!input || !npts || !dim) throw std::invalid_argument("Invalid sampling input");
  ndims = dim;
  const auto ids = sample_ids(npts, p_val);
  slice_size = ids.size();
  auto samples = std::make_unique<float[]>(slice_size * ndims);
  std::vector<T> row(ndims);
  for (size_t i = 0; i < ids.size(); ++i) {
    input.seekg(8 + ids[i] * ndims * sizeof(T));
    input.read(reinterpret_cast<char *>(row.data()), ndims * sizeof(T));
    if (!input) throw std::runtime_error("Cannot read training sample");
    std::copy(row.begin(), row.end(), samples.get() + i * ndims);
  }
  sampled_data = samples.release();
}

// Reads prepared records without loading the entire dataset. Decoded vectors
// exist only in the k-means sample and current block; persisted codes stay intact.
template<typename T>
class PartitionReader {
 public:
  size_t count, dim, record_bytes, width;
  std::string prefix;
  std::unique_ptr<pipeann::RaBitQQuantizer<T, 8>> codec;
  std::ifstream input;

  PartitionReader(const std::string &path, pipeann::Metric metric) : prefix(path), input(path, std::ios::binary) {
    pipeann::get_bin_metadata(path, count, width);
    const std::string suffix = "_processed.bin";
    if (prefix.size() >= suffix.size() && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0)
      prefix.resize(prefix.size() - suffix.size());
    if (std::filesystem::exists(prefix + "_rabitq_metadata.bin")) {
      codec = std::make_unique<pipeann::RaBitQQuantizer<T, 8>>(metric);
      codec->load(prefix);
      dim = codec->dim;
      record_bytes = codec->vector_size();
      if (width != record_bytes || record_bytes != pipeann::RaBitQQuantizer<T, 8>::vector_size(dim))
        throw std::invalid_argument("Partitioning requires RBQ8 records");
    } else {
      dim = width;
      record_bytes = dim * sizeof(T);
    }
    if (!count || !dim || std::filesystem::file_size(path) != 8 + count * record_bytes)
      throw std::invalid_argument("Invalid partition input: " + path);
  }

  void read(size_t first, size_t rows, uint8_t *records) {
    input.clear();
    input.seekg(8 + first * record_bytes);
    input.read(reinterpret_cast<char *>(records), rows * record_bytes);
    if (!input) throw std::runtime_error("Cannot read partition records");
  }

  void decode(const uint8_t *records, size_t rows, float *out) const {
    if (!codec) {
      pipeann::convert_types<T, float>(reinterpret_cast<const T *>(records), out, rows, dim);
      return;
    }
#pragma omp parallel
    {
      const auto &quantizer = *codec;
      std::vector<float> scratch(quantizer.padded_dim);
#pragma omp for schedule(static)
      for (size_t i = 0; i < rows; ++i) {
        quantizer.decode_rotated(records + i * record_bytes, scratch.data());
        quantizer.transform(scratch.data(), true);
        std::copy_n(scratch.data(), dim, out + i * dim);
      }
    }
  }

  std::vector<float> sample(double rate) {
    const auto ids = sample_ids(count, rate);
    const size_t n = ids.size();
    std::vector<float> samples(n * dim);
    const size_t block_size = std::min<size_t>(MAX_BLOCK_SIZE, n);
    std::vector<uint8_t> records(block_size * record_bytes);
    for (size_t start = 0; start < n; start += block_size) {
      const size_t rows = std::min(block_size, n - start);
      for (size_t i = 0; i < rows;) {
        size_t end = i + 1;
        while (end < rows && ids[start + end] == ids[start + i] + end - i) ++end;
        read(ids[start + i], end - i, records.data() + i * record_bytes);
        i = end;
      }
      decode(records.data(), rows, samples.data() + start * dim);
    }
    return samples;
  }
};

static std::vector<size_t> estimate_cluster_sizes(float *sample, size_t num_sample, size_t num_points,
                                                 float *pivots, size_t num_centers, size_t dim, size_t k_base) {
  std::vector<size_t> counts(num_centers);
  const size_t block_size = std::min<size_t>(MAX_BLOCK_SIZE, num_sample);
  std::vector<uint32_t> closest(block_size * k_base);
  for (size_t start = 0; start < num_sample; start += block_size) {
    const size_t rows = std::min(block_size, num_sample - start);
    kmeans::compute_closest_centers(sample + start * dim, rows, dim, pivots, num_centers, k_base, closest.data());
    for (size_t i = 0; i < rows * k_base; ++i) ++counts[closest[i]];
  }
  for (auto &count : counts)
    count = size_t(std::ceil(double(count) * num_points / num_sample));
  return counts;
}

template<typename T>
static int shard_data_into_clusters(PartitionReader<T> &reader, float *pivots, const size_t num_centers,
                                    const size_t k_base, const std::string &prefix_path) {
  const size_t num_points = reader.count, dim = reader.dim, record_bytes = reader.record_bytes;
  const uint32_t output_width = reader.width;
  std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);
  std::vector<std::ofstream> shard_data_writer(num_centers);
  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  uint32_t dummy_size = 0;
  uint32_t const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    const std::string shard_prefix = prefix_path + "_subshard-" + std::to_string(i);
    const std::string vector_prefix = shard_prefix + "_data.bin";
    const std::string data_filename = vector_prefix + (reader.codec ? "_processed.bin" : "");
    if (reader.codec)
      for (const char *suffix : pipeann::RABITQ_METADATA_SUFFIXES)
        std::filesystem::copy_file(reader.prefix + suffix, vector_prefix + suffix,
                                   std::filesystem::copy_options::overwrite_existing);
    else
      for (const char *suffix : pipeann::RABITQ_METADATA_SUFFIXES)
        std::filesystem::remove(vector_prefix + suffix);
    std::string idmap_filename = shard_prefix + "_ids_uint32.bin";
    shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_data_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_data_writer[i].write((char *) &output_width, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
    if (!shard_data_writer[i] || !shard_idmap_writer[i])
      throw std::runtime_error("Cannot create partition files: " + shard_prefix);
    shard_counts[i] = 0;
  }

  size_t BLOCK_SIZE = std::min((size_t) MAX_BLOCK_SIZE, num_points);
  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * k_base);
  std::vector<uint8_t> records(block_size * record_bytes);
  std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = std::min((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    reader.read(start_id, cur_blk_size, records.data());
    reader.decode(records.data(), cur_blk_size, block_data_float.get());

    kmeans::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
                                    block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t) (start_id + p);
        const char *record = reinterpret_cast<const char *>(records.data() + p * record_bytes);
        shard_data_writer[shard_id].write(record, record_bytes);
        shard_idmap_writer[shard_id].write((char *) &original_point_map_id, sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  LOG(INFO) << "Actual shard sizes: ";
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t) shard_counts[i];
    total_count += cur_shard_count;
    LOG(INFO) << cur_shard_count << " ";
    shard_data_writer[i].seekp(0);
    shard_data_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_data_writer[i].close();
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
    if (!shard_data_writer[i] || !shard_idmap_writer[i])
      throw std::runtime_error("Cannot write partition files");
  }

  LOG(INFO) << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get " << total_count
            << " points across " << num_centers << " shards ";
  return 0;
}

template<typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base, pipeann::Metric metric) {
  if (!std::isfinite(ram_budget) || ram_budget <= 0 || !k_base || !std::isfinite(sampling_rate))
    throw std::invalid_argument("Invalid partition parameters");
  PartitionReader<T> reader(data_file, metric);
  auto samples = reader.sample(sampling_rate);
  const size_t train_dim = reader.dim, num_train = samples.size() / train_dim;
  size_t num_parts = std::max<size_t>(3, k_base);
  std::vector<float> pivots;
  while (true) {
    if (num_parts > num_train)
      throw std::invalid_argument("RAM budget requires more partitions than training samples");
    pivots.resize(num_parts * train_dim);
    LOG(INFO) << "Processing global k-means with " << num_parts << " parts";
    kmeans::kmeanspp_selecting_pivots(samples.data(), num_train, train_dim, pivots.data(), num_parts);
    kmeans::run_elkan(samples.data(), num_train, train_dim, pivots.data(), num_parts, 20, nullptr, nullptr);
    const auto cluster_sizes = estimate_cluster_sizes(samples.data(), num_train, reader.count, pivots.data(),
                                                       num_parts, train_dim, k_base);
    double max_ram_usage = 0;
    for (size_t size : cluster_sizes)
      max_ram_usage = std::max(max_ram_usage,
          pipeann::estimate_ram_usage(size, reader.record_bytes, 1, graph_degree));
    LOG(INFO) << "With " << num_parts << " parts, max estimated RAM usage: " << max_ram_usage / (1024 * 1024 * 1024)
              << "GB, budget given is " << ram_budget;
    if (max_ram_usage <= 1024 * 1024 * 1024 * ram_budget) break;
    ++num_parts;
  }

  LOG(INFO) << "Saving global k-center pivots";
  pipeann::save_bin<float>(prefix_path + "_centroids.bin", pivots.data(), num_parts, train_dim);

  shard_data_into_clusters(reader, pivots.data(), num_parts, k_base, prefix_path);
  return num_parts;
}

template void gen_random_slice<int8_t>(const std::string base_file, const std::string output_prefix,
                                       double sampling_rate, size_t offset);
template void gen_random_slice<uint8_t>(const std::string base_file, const std::string output_prefix,
                                        double sampling_rate, size_t offset);
template void gen_random_slice<float>(const std::string base_file, const std::string output_prefix,
                                      double sampling_rate, size_t offset);

template void gen_random_slice<float>(const std::string data_file, double p_val, float *&sampled_data,
                                      size_t &slice_size, size_t &ndims);
template void gen_random_slice<uint8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                        size_t &slice_size, size_t &ndims);
template void gen_random_slice<int8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                       size_t &slice_size, size_t &ndims);

template int partition_with_ram_budget<int8_t>(const std::string data_file, const double sampling_rate,
                                               double ram_budget, size_t graph_degree, const std::string prefix_path,
                                               size_t k_base, pipeann::Metric metric);
template int partition_with_ram_budget<uint8_t>(const std::string data_file, const double sampling_rate,
                                                double ram_budget, size_t graph_degree, const std::string prefix_path,
                                                size_t k_base, pipeann::Metric metric);
template int partition_with_ram_budget<float>(const std::string data_file, const double sampling_rate,
                                              double ram_budget, size_t graph_degree, const std::string prefix_path,
                                              size_t k_base, pipeann::Metric metric);
