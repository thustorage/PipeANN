#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cblas.h>
#include <chrono>
#include <cblas.h>

#include "utils/index_build_utils.h"
#include "utils/tsl/robin_set.h"
#include "filter/attribute.h"
#include "utils/cached_io.h"
#include "index.h"
#include "nbr/rabitq_nbr.h"
#include "ssd_index.h"
#include "ssd_index_defs.h"
#include "omp.h"
#include "utils/partition.h"
#include "utils.h"

namespace pipeann {
  template<typename T>
  std::string prepare_build_vectors(const std::string &filename, Metric metric, VectorType node_type) {
    size_t n, dim;
    get_bin_metadata(filename, n, dim);
    if (!n || !dim || std::filesystem::file_size(filename) != 8 + n * dim * sizeof(T))
      throw std::invalid_argument("RAW vector file shape/size mismatch: " + filename);
    if (node_type != VectorType::RAW && node_type != VectorType::RABITQ8)
      throw std::invalid_argument("Build nodes must be RAW or RaBitQ8");

    const std::string &prefix = filename;
    // Re-preparing RAW replaces any previous RBQ representation at this prefix.
    if (node_type == VectorType::RAW)
      for (const char *suffix : RABITQ_METADATA_SUFFIXES)
        std::filesystem::remove(prefix + suffix);
    std::string input = filename;
    if (metric == COSINE) {
      input = prefix + (node_type == VectorType::RAW ? "_processed.bin" : "_normalized.tmp.bin");
      LOG(INFO) << "Streaming normalization to " << input;
      std::ifstream in(filename, std::ios::binary);
      in.seekg(8);
      std::ofstream out(input, std::ios::binary);
      const uint32_t header[] = {static_cast<uint32_t>(n), static_cast<uint32_t>(dim)};
      out.write(reinterpret_cast<const char *>(header), sizeof(header));
      const size_t block_size = std::min<size_t>(4096, n);
      std::vector<T> block(block_size * dim);
      for (size_t begin = 0; begin < n; begin += block_size) {
        const size_t rows = std::min(block_size, n - begin);
        in.read(reinterpret_cast<char *>(block.data()), rows * dim * sizeof(T));
        if (!in) throw std::runtime_error("Cannot read build input vectors");
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
          normalize_data(block.data() + i * dim, block.data() + i * dim, dim);
        out.write(reinterpret_cast<const char *>(block.data()), rows * dim * sizeof(T));
      }
      out.close();
      if (!out) throw std::runtime_error("Cannot write normalized vectors");
    }
    if (node_type == VectorType::RAW) return input;

    auto codec = std::make_unique<RaBitQNeighbor<T, 8>>(metric);
    codec->training_centers = 1;
    codec->build(prefix, input, 8);
    if (input != filename) std::remove(input.c_str());
    return prefix + "_processed.bin";
  }

  void remove_prepared_vectors(const std::string &filename, const std::string &prepared, VectorType node_type) {
    if (prepared != filename) std::filesystem::remove(prepared);
    if (node_type == VectorType::RABITQ8)
      for (const char *suffix : RABITQ_METADATA_SUFFIXES)
        std::filesystem::remove(filename + suffix);
  }

  template std::string prepare_build_vectors<float>(const std::string &, Metric, VectorType);
  template std::string prepare_build_vectors<int8_t>(const std::string &, Metric, VectorType);
  template std::string prepare_build_vectors<uint8_t>(const std::string &, Metric, VectorType);

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    size_t npts, dim;
    pipeann::load_bin(fname, ivecs, npts, dim);
  }

  template<typename T, typename TagT>
  int merge_shards(const std::string &shard_prefix, const std::string &shard_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const uint64_t nshards, uint16_t R, uint16_t R_dense,
                   uint16_t R_ood, const std::string &output_disk_file, const std::string &tag_file,
                   AttrWriter *attr_writer) {
    R_dense = std::max(R_dense, R);
    const uint16_t R_base = R - R_ood;
    const uint16_t shard_R_ood = (uint16_t) (2 * (uint32_t) R_ood / 3);

    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (uint64_t s = 0; s < nshards; s++) {
      read_idmap(idmaps_prefix + std::to_string(s) + idmaps_suffix, idmaps[s]);
    }

    uint64_t nnodes = 0;
    uint64_t nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap)
        nnodes = std::max(nnodes, (uint64_t) id);
      nelems += idmap.size();
    }
    nnodes++;
    LOG(INFO) << "# nodes: " << nnodes << ", max. degree: " << R;

    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (uint64_t s = 0; s < nshards; s++) {
      for (uint64_t i = 0; i < idmaps[s].size(); i++) {
        node_shard.emplace_back((uint32_t) idmaps[s][i], (uint32_t) s);
      }
    }
    std::sort(node_shard.begin(), node_shard.end());

    struct ShardReader {
      cached_ifstream in;
      SSDIndexMetadata<T> meta;
      std::vector<char> buf;
      uint64_t bytes_per_read = 0;
      uint64_t nodes_per_read = 0;
      uint64_t cur_in_buf = 0;
    };
    constexpr uint64_t kBlk = 64 * 1024 * 1024;
    std::vector<ShardReader> readers(nshards);
    uint64_t ndims = 0;
    for (uint64_t s = 0; s < nshards; s++) {
      auto path = shard_prefix + std::to_string(s) + shard_suffix;
      readers[s].meta.load_from_disk_index(path);
      readers[s].in.open(path, kBlk, SECTOR_LEN);

      readers[s].bytes_per_read =
          readers[s].meta.nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(readers[s].meta.max_node_len, SECTOR_LEN);
      readers[s].nodes_per_read = readers[s].meta.nnodes_per_sector > 0 ? readers[s].meta.nnodes_per_sector : 1;
      readers[s].buf.resize(readers[s].bytes_per_read);
      readers[s].cur_in_buf = readers[s].nodes_per_read;  // force initial read
    }

    ndims = readers[0].meta.data_dim;

    const uint64_t attr_size = attr_writer ? attr_writer->attr_size() : 0;
    const uint64_t vec_bytes = readers[0].meta.vector_bytes();
    for (const auto &r : readers)
      if (r.meta.vector_bytes() != vec_bytes || r.meta.vector_type != readers[0].meta.vector_type)
        throw std::runtime_error("Incompatible shard vector encodings");
    const uint64_t max_node_len = ((uint64_t) R_dense + 1) * sizeof(uint32_t) + vec_bytes + attr_size;
    const uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
    SSDIndexMetadata<T> out_meta(nnodes, ndims, idmaps[0][readers[0].meta.entry_point], max_node_len, nnodes_per_sector,
                                 R, attr_size, R_ood, readers[0].meta.vector_type);
    out_meta.print();

    const uint64_t bytes_per_write = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
    const uint64_t nodes_per_write = nnodes_per_sector > 0 ? nnodes_per_sector : 1;
    std::vector<char> sector_buf(bytes_per_write, 0);

    std::remove(output_disk_file.c_str());
    cached_ofstream out;
    out.open(output_disk_file, kBlk);
    out.write(sector_buf.data(), SECTOR_LEN);  // metadata placeholder

    std::random_device rng;
    std::mt19937 urng(rng());

    enum { ONE_HOP, OOD, DENSE, NHOOD_TYPE_CNT };
    // Per-node dedup sets keyed by global id; only hold the current node's neighbors
    // (a few hundred entries), so they stay cache-resident even at billion scale.
    tsl::robin_set<unsigned> nhood_set[NHOOD_TYPE_CNT];
    std::vector<unsigned> nhood[NHOOD_TYPE_CNT];

    std::vector<unsigned> final_nhood;
    final_nhood.reserve(R_dense);
    std::vector<uint8_t> coord_buf(vec_bytes, 0);

    uint64_t cur_sector_start = 0;

    auto finalize_node = [&](uint64_t global_id) {
      const uint64_t sector_start = (global_id / nodes_per_write) * nodes_per_write;
      if (sector_start != cur_sector_start) {
        out.write(sector_buf.data(), bytes_per_write);
        memset(sector_buf.data(), 0, bytes_per_write);
        cur_sector_start = sector_start;
      }

      DiskNode<T> node(sector_buf.data(), (uint32_t) global_id, out_meta);
      memcpy(node.coords, coord_buf.data(), vec_bytes);

      for (auto &nhood_vec : nhood) {
        std::shuffle(nhood_vec.begin(), nhood_vec.end(), urng);
      }

      // Fill with 1-hop then 2-hop up to R_dense; OOD overwrites tail R_ood slots of the first R.
      final_nhood.clear();
      size_t n_1hop = std::min(nhood[ONE_HOP].size(), (size_t) R_dense);
      final_nhood.insert(final_nhood.end(), nhood[ONE_HOP].begin(), nhood[ONE_HOP].begin() + n_1hop);
      for (size_t i = 0; i < nhood[DENSE].size() && final_nhood.size() < R_dense; i++) {
        unsigned nbr = nhood[DENSE][i];
        if (!nhood_set[ONE_HOP].count(nbr)) {
          final_nhood.push_back(nbr);
        }
      }

      const uint16_t nnbrs = (uint16_t) std::min((size_t) R, final_nhood.size());

      size_t n_ood_nbrs = 0;
      for (size_t i = 0; i < nhood[OOD].size() && n_ood_nbrs < R_ood && n_ood_nbrs < nnbrs; i++) {
        unsigned nbr = nhood[OOD][i];
        if (nhood_set[ONE_HOP].count(nbr) || nhood_set[DENSE].count(nbr))
          continue;
        final_nhood[nnbrs - 1 - n_ood_nbrs] = nbr;  // reverse order into tail R_ood slots.
        n_ood_nbrs++;
      }

      node.nnbrs = nnbrs;
      node.n_dense_nbrs = (uint16_t) (final_nhood.size() - nnbrs);
      memcpy(node.nbrs, final_nhood.data(), node.nnbrs * sizeof(uint32_t));
      memcpy(node.dense_nbrs, final_nhood.data() + nnbrs, node.n_dense_nbrs * sizeof(uint32_t));

      if (attr_size > 0) {
        attr_writer->write((uint32_t) global_id, node.attrs);
      }

      for (size_t i = 0; i < NHOOD_TYPE_CNT; i++) {
        nhood_set[i].clear();
        nhood[i].clear();
      }
    };

    LOG(INFO) << "Starting merge";

    uint64_t cur_id = 0;
    for (const auto &id_shard : node_shard) {
      const uint32_t global_id = id_shard.first;
      const uint32_t shard = id_shard.second;
      if (global_id != cur_id) {
        finalize_node(cur_id);
        cur_id = global_id;
      }

      auto &r = readers[shard];
      if (r.cur_in_buf >= r.nodes_per_read) {
        r.in.read(r.buf.data(), r.bytes_per_read);
        r.cur_in_buf = 0;
      }
      // loc == cur_in_buf so DiskNode picks position (cur_in_buf % nnodes_per_sector) in the sector.
      DiskNode<T> shard_node(r.buf.data(), r.cur_in_buf, r.meta);
      r.cur_in_buf++;
      memcpy(coord_buf.data(), shard_node.coords, vec_bytes);

      for (uint16_t j = 0; j < shard_node.nnbrs; j++) {
        const unsigned renamed = idmaps[shard][shard_node.nbrs[j]];
        if (j < shard_node.nnbrs - shard_R_ood) {  // base nbr.
          if (nhood_set[ONE_HOP].insert(renamed).second) {
            nhood[ONE_HOP].push_back(renamed);
          }
        } else {  // OOD nbr.
          if (nhood_set[OOD].insert(renamed).second) {
            nhood[OOD].push_back(renamed);
          }
        }
      }

      for (uint16_t j = 0; j < shard_node.n_dense_nbrs; j++) {  // dense nbr.
        const unsigned renamed = idmaps[shard][shard_node.dense_nbrs[j]];
        if (nhood_set[DENSE].insert(renamed).second) {
          nhood[DENSE].push_back(renamed);
        }
      }
    }
    finalize_node(cur_id);

    out.write(sector_buf.data(), bytes_per_write);
    out.close();

    if (!tag_file.empty() && file_exists(tag_file)) {
      std::filesystem::copy_file(tag_file, output_disk_file + ".tags",
                                 std::filesystem::copy_options::overwrite_existing);
    } else {
      // generate identical tags.
      std::vector<TagT> identical_tags(nnodes);
      std::iota(identical_tags.begin(), identical_tags.end(), TagT{0});
      pipeann::save_bin<TagT>(output_disk_file + ".tags", identical_tags.data(), nnodes, 1);
    }

    // Records retain their shard encoding; publish the matching shared codec.
    if (out_meta.vector_type == VectorType::RABITQ8)
      for (const char *suffix : RABITQ_METADATA_SUFFIXES)
        std::filesystem::copy_file(shard_prefix + "0" + shard_suffix + suffix, output_disk_file + suffix,
                                   std::filesystem::copy_options::overwrite_existing);
    out_meta.save_to_disk_index(output_disk_file);
    LOG(INFO) << "merge_shards: wrote " << output_disk_file;
    return 0;
  }

  template<typename T, typename TagT>
  int build_merged_index(const std::string &base_file, pipeann::Metric _compareMetric, uint16_t R, uint32_t L_or_L1,
                         double ram_budget, std::string disk_index_path, const char *tag_file,
                         uint16_t R_dense, uint32_t num_threads, uint32_t L2, const std::string &train_query_path,
                         uint16_t R_ood, uint32_t L_ood, AttrWriter *attr_writer, VectorType node_type) {
    size_t base_num, base_dim;
    pipeann::get_bin_metadata(base_file, base_num, base_dim);
    if (!base_num || !base_dim || !std::isfinite(ram_budget) || ram_budget <= 0 ||
        std::filesystem::file_size(base_file) != 8 + base_num * base_dim * sizeof(T))
      throw std::invalid_argument("Invalid build input or RAM budget");
    if (node_type != VectorType::RAW && node_type != VectorType::RABITQ8)
      throw std::invalid_argument("Build nodes must be RAW or RaBitQ8");

    const bool encoded = node_type == VectorType::RABITQ8;
    const size_t record_bytes = encoded ? RaBitQQuantizer<T, 8>::vector_size(base_dim) : base_dim * sizeof(T);
    const double full_index_ram = estimate_ram_usage(base_num, record_bytes, 1, R);
    if (L2 != 0 && full_index_ram >= ram_budget * 1024 * 1024 * 1024)
      throw std::invalid_argument("PiPNN requires the full index to fit in RAM");

    const auto graph_input = prepare_build_vectors<T>(base_file, _compareMetric, node_type);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG(INFO) << "Full index fits in RAM, building in one shot";
      pipeann::IndexBuildParameters paras;
      paras.use_pipnn = (L2 != 0);
      paras.set(R, L_or_L1, 750, 1.2, num_threads, true);
      paras.pipnn_set(2048, 256, 0.01f, 1000, L_or_L1, L2, 2, 128);

      Index<T, TagT> index(load_neighbor<T>(graph_input, _compareMetric));
      index.build(paras, tag_file, train_query_path, R_ood, L_ood);
      SSDIndex<T, TagT>::save_from_mem(index, disk_index_path, R_dense, attr_writer);
      remove_prepared_vectors(base_file, graph_input, node_type);
      return 0;
    }

    constexpr size_t kMaxPartitionSamples = 256000;
    const double sample_p = std::min(1.0, double(kMaxPartitionSamples) / base_num);
    const std::string merged_index_prefix = disk_index_path + "_tempFiles";
    const int num_parts = partition_with_ram_budget<T>(graph_input, sample_p, ram_budget, 2 * R / 3,
                                                       merged_index_prefix, 2, _compareMetric);
    remove_prepared_vectors(base_file, graph_input, node_type);

    const uint16_t shard_R_ood = (uint16_t) (2 * (uint32_t) R_ood / 3);
    const uint16_t shard_R_dense = (uint16_t) (2 * (uint32_t) R_dense / 3);
    for (int p = 0; p < num_parts; p++) {
      const std::string shard_prefix = merged_index_prefix + "_subshard-" + std::to_string(p);
      const std::string shard_raw_file = shard_prefix + "_data.bin";
      const std::string shard_disk_file = shard_prefix + "_disk.index";
      const auto shard_input = encoded ? shard_raw_file + "_processed.bin" : shard_raw_file;
      pipeann::IndexBuildParameters paras;
      paras.set(2 * R / 3, L_or_L1, 750, 1.2, num_threads, false);
      Index<T, TagT> index(load_neighbor<T>(shard_input, _compareMetric));
      index.build(paras, (const char *) nullptr, train_query_path, shard_R_ood, L_ood);
      SSDIndex<T, TagT>::save_from_mem(index, shard_disk_file, shard_R_dense);
      remove_prepared_vectors(shard_raw_file, shard_input, node_type);
      std::remove(shard_raw_file.c_str());
    }

    pipeann::merge_shards<T, TagT>(merged_index_prefix + "_subshard-", "_disk.index",
                                   merged_index_prefix + "_subshard-", "_ids_uint32.bin", num_parts, R, R_dense, R_ood,
                                   disk_index_path, tag_file ? std::string(tag_file) : std::string(""), attr_writer);

    for (int p = 0; p < num_parts; p++) {
      const std::string shard_prefix = merged_index_prefix + "_subshard-" + std::to_string(p);
      const std::string shard_disk_file = shard_prefix + "_disk.index";
      std::remove((shard_prefix + "_ids_uint32.bin").c_str());
      std::remove(shard_disk_file.c_str());
      std::remove((shard_disk_file + ".tags").c_str());
      if (encoded) {
        for (const char *suffix : RABITQ_METADATA_SUFFIXES) {
          std::remove((shard_disk_file + suffix).c_str());
        }
      }
    }
    std::remove((merged_index_prefix + "_centroids.bin").c_str());
    return 0;
  }

  template<typename T, typename TagT>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, uint16_t R, uint32_t L_or_L1, uint32_t M,
                        uint32_t num_threads, uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                        const char *tag_file, AbstractNeighbor<T> *nbr_handler, AttrWriter *attr_writer,
                        uint16_t R_dense, uint32_t L2, const std::string &train_query_path, uint16_t R_ood,
                        uint32_t L_ood, VectorType node_type) {
    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string disk_index_path = index_prefix_path + "_disk.index";

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    if (L2 == 0) {
      LOG(INFO) << "Starting Vamana index build: R=" << R << " R_dense=" << R_dense << " L=" << L_or_L1
                << " Build RAM budget: " << M << "GB T: " << num_threads << " bytes per neighbor: " << bytes_per_nbr;
    } else {
      LOG(INFO) << "Starting PiPNN index build: R=" << R << " R_dense=" << R_dense << " L1=" << L_or_L1 << " L2=" << L2
                << " Build RAM budget: " << M << "GB T: " << num_threads << " bytes per neighbor: " << bytes_per_nbr;
    }

    auto s = std::chrono::high_resolution_clock::now();
    const auto neighbor_input = prepare_build_vectors<T>(dataFilePath, _compareMetric, VectorType::RAW);
    nbr_handler->build(index_prefix_path, neighbor_input, bytes_per_nbr);
    remove_prepared_vectors(dataFilePath, neighbor_input, VectorType::RAW);

    auto start = std::chrono::high_resolution_clock::now();
    nbr_handler->clear();  // free compressed neighbor codes during graph construction.
    pipeann::build_merged_index<T, TagT>(dataFilePath, _compareMetric, R, L_or_L1, M, disk_index_path,
                                         tag_file, R_dense, num_threads, L2, train_query_path, R_ood, L_ood,
                                         attr_writer, node_type);
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << (L2 == 0 ? "Vamana" : "PiPNN")
              << " index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  template bool build_disk_index<int8_t, uint32_t>(const char *dataPath, const char *indexFilePath, uint16_t R,
                                                   uint32_t L, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                                                   pipeann::Metric _compareMetric, const char *tag_file,
                                                   AbstractNeighbor<int8_t> *nbr_handler, AttrWriter *label,
                                                   uint16_t R_dense, uint32_t L2, const std::string &train_query_path,
                                                   uint16_t R_ood, uint32_t L_ood, VectorType node_type);
  template bool build_disk_index<uint8_t, uint32_t>(const char *dataPath, const char *indexFilePath, uint16_t R,
                                                    uint32_t L, uint32_t M, uint32_t num_threads,
                                                    uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                                                    const char *tag_file, AbstractNeighbor<uint8_t> *nbr_handler,
                                                    AttrWriter *label, uint16_t R_dense, uint32_t L2,
                                                    const std::string &train_query_path, uint16_t R_ood, uint32_t L_ood,
                                                    VectorType node_type);
  template bool build_disk_index<float, uint32_t>(const char *dataPath, const char *indexFilePath, uint16_t R,
                                                  uint32_t L, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                                                  pipeann::Metric _compareMetric, const char *tag_file,
                                                  AbstractNeighbor<float> *nbr_handler, AttrWriter *label,
                                                  uint16_t R_dense, uint32_t L2, const std::string &train_query_path,
                                                  uint16_t R_ood, uint32_t L_ood, VectorType node_type);

  template int build_merged_index<int8_t, uint32_t>(const std::string &base_file, pipeann::Metric _compareMetric,
                                                    uint16_t R, uint32_t L_or_L1,
                                                    double ram_budget, std::string disk_index_path,
                                                    const char *tag_file, uint16_t R_dense, uint32_t num_threads,
                                                    uint32_t L2, const std::string &train_query_path, uint16_t R_ood,
                                                    uint32_t L_ood, AttrWriter *attr_writer,
                                                    VectorType node_type);
  template int build_merged_index<float, uint32_t>(const std::string &base_file, pipeann::Metric _compareMetric,
                                                   uint16_t R, uint32_t L_or_L1,
                                                   double ram_budget, std::string disk_index_path, const char *tag_file,
                                                   uint16_t R_dense, uint32_t num_threads, uint32_t L2,
                                                   const std::string &train_query_path, uint16_t R_ood, uint32_t L_ood,
                                                   AttrWriter *attr_writer, VectorType node_type);
  template int build_merged_index<uint8_t, uint32_t>(const std::string &base_file, pipeann::Metric _compareMetric,
                                                     uint16_t R, uint32_t L_or_L1,
                                                     double ram_budget, std::string disk_index_path,
                                                     const char *tag_file, uint16_t R_dense, uint32_t num_threads,
                                                     uint32_t L2, const std::string &train_query_path, uint16_t R_ood,
                                                     uint32_t L_ood, AttrWriter *attr_writer,
                                                     VectorType node_type);

  template int merge_shards<int8_t, uint32_t>(const std::string &shard_prefix, const std::string &shard_suffix,
                                              const std::string &idmaps_prefix, const std::string &idmaps_suffix,
                                              uint64_t nshards, uint16_t R, uint16_t R_dense, uint16_t R_ood,
                                              const std::string &output_disk_file, const std::string &tag_file,
                                              AttrWriter *attr_writer);
  template int merge_shards<uint8_t, uint32_t>(const std::string &shard_prefix, const std::string &shard_suffix,
                                               const std::string &idmaps_prefix, const std::string &idmaps_suffix,
                                               uint64_t nshards, uint16_t R, uint16_t R_dense, uint16_t R_ood,
                                               const std::string &output_disk_file, const std::string &tag_file,
                                               AttrWriter *attr_writer);
  template int merge_shards<float, uint32_t>(const std::string &shard_prefix, const std::string &shard_suffix,
                                             const std::string &idmaps_prefix, const std::string &idmaps_suffix,
                                             uint64_t nshards, uint16_t R, uint16_t R_dense, uint16_t R_ood,
                                             const std::string &output_disk_file, const std::string &tag_file,
                                             AttrWriter *attr_writer);
};  // namespace pipeann
