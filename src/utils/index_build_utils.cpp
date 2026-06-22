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
#include "filter/attribute.h"
#include "utils/cached_io.h"
#include "index.h"
#include "ssd_index.h"
#include "ssd_index_defs.h"
#include "omp.h"
#include "utils/partition.h"
#include "utils.h"

namespace pipeann {
  template<typename T>
  void normalize_data_file(const std::string &inFileName, const std::string &outFileName) {
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    int npts_s32, ndims_s32;
    readr.read((char *) &npts_s32, sizeof(int32_t));
    readr.read((char *) &ndims_s32, sizeof(int32_t));

    writr.write((char *) &npts_s32, sizeof(int32_t));
    writr.write((char *) &ndims_s32, sizeof(int32_t));

    uint64_t npts = (uint64_t) npts_s32, ndims = (uint64_t) ndims_s32;
    LOG(INFO) << "Normalizing vectors in file: " << inFileName;
    LOG(INFO) << "Dataset: #pts = " << npts << ", # dims = " << ndims;

    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    LOG(INFO) << "# blks: " << nblks;

    T *read_buf = new T[blk_size * ndims];
    for (uint64_t i = 0; i < nblks; i++) {
      uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
      readr.read((char *) read_buf, cblk_size * ndims * sizeof(T));
#pragma omp parallel for schedule(static, 4096)
      for (uint64_t j = 0; j < cblk_size; j++) {
        normalize_data(read_buf + j * ndims, read_buf + j * ndims, ndims);
      }
      writr.write((char *) read_buf, cblk_size * ndims * sizeof(T));
    }
    delete[] read_buf;

    LOG(INFO) << "Wrote normalized points to file: " << outFileName;
  }

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
    const uint64_t max_node_len = ((uint64_t) R_dense + 1) * sizeof(uint32_t) + ndims * sizeof(T) + attr_size;
    const uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
    SSDIndexMetadata<T> out_meta(nnodes, ndims, idmaps[0][readers[0].meta.entry_point], max_node_len,
                                 nnodes_per_sector, R, attr_size, R_ood);
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

    enum {ONE_HOP, OOD, DENSE, NHOOD_TYPE_CNT};
    std::vector<bool> nhood_set[NHOOD_TYPE_CNT];
    for (int i = 0; i < NHOOD_TYPE_CNT; i++) {
      nhood_set[i].resize(nnodes, false);
    }
    std::vector<unsigned> nhood[NHOOD_TYPE_CNT];
    
    std::vector<unsigned> final_nhood;
    final_nhood.reserve(R_dense);
    std::vector<T> coord_buf(ndims, 0);

    uint64_t cur_sector_start = 0;

    auto finalize_node = [&](uint64_t global_id) {
      const uint64_t sector_start = (global_id / nodes_per_write) * nodes_per_write;
      if (sector_start != cur_sector_start) {
        out.write(sector_buf.data(), bytes_per_write);
        memset(sector_buf.data(), 0, bytes_per_write);
        cur_sector_start = sector_start;
      }

      DiskNode<T> node(sector_buf.data(), (uint32_t) global_id, out_meta);
      memcpy(node.coords, coord_buf.data(), ndims * sizeof(T));

      for (auto &nhood_vec : nhood) {
        std::shuffle(nhood_vec.begin(), nhood_vec.end(), urng);
      }

      // Fill with 1-hop then 2-hop up to R_dense; OOD overwrites tail R_ood slots of the first R.
      final_nhood.clear();
      size_t n_1hop = std::min(nhood[ONE_HOP].size(), (size_t) R_dense);
      final_nhood.insert(final_nhood.end(), nhood[ONE_HOP].begin(), nhood[ONE_HOP].begin() + n_1hop);
      for (size_t i = 0; i < nhood[DENSE].size() && final_nhood.size() < R_dense; i++) {
        unsigned nbr = nhood[DENSE][i];
        if (!nhood_set[ONE_HOP][nbr]) {
          final_nhood.push_back(nbr);
        }
      }

      const uint16_t nnbrs = (uint16_t) std::min((size_t) R, final_nhood.size());

      size_t n_ood_nbrs = 0;
      for (size_t i = 0; i < nhood[OOD].size() && n_ood_nbrs < R_ood && n_ood_nbrs < nnbrs; i++) {
        unsigned nbr = nhood[OOD][i];
        if (nhood_set[ONE_HOP][nbr] || nhood_set[DENSE][nbr]) continue;
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
        nhood_set[i].assign(nnodes, false);
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
      memcpy(coord_buf.data(), shard_node.coords, ndims * sizeof(T));

      for (uint16_t j = 0; j < shard_node.nnbrs; j++) {
        const unsigned renamed = idmaps[shard][shard_node.nbrs[j]];
        if (j < shard_node.nnbrs - shard_R_ood) { // base nbr.
          if (!nhood_set[ONE_HOP][renamed]) {
            nhood_set[ONE_HOP][renamed] = true;
            nhood[ONE_HOP].push_back(renamed);
          }
        } else { // OOD nbr.
          if (!nhood_set[OOD][renamed]) {
            nhood_set[OOD][renamed] = true;
            nhood[OOD].push_back(renamed);
          }
        }
      }

      for (uint16_t j = 0; j < shard_node.n_dense_nbrs; j++) { // dense nbr.
        const unsigned renamed = idmaps[shard][shard_node.dense_nbrs[j]];
        if (!nhood_set[DENSE][renamed]) {
          nhood_set[DENSE][renamed] = true;
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

    out_meta.save_to_disk_index(output_disk_file);
    LOG(INFO) << "merge_shards: wrote " << output_disk_file;
    return 0;
  }

  template<typename T, typename TagT>
  int build_merged_index(std::string base_file, pipeann::Metric _compareMetric, uint16_t R, uint32_t L_or_L1,
                         double sampling_rate, double ram_budget, std::string disk_index_path, const char *tag_file,
                         uint16_t R_dense, uint32_t num_threads, uint32_t L2, const std::string &train_query_path,
                         uint16_t R_ood, uint32_t L_ood, AttrWriter *attr_writer) {
    size_t base_num, base_dim;
    pipeann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram = estimate_ram_usage(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG(INFO) << "Full index fits in RAM, building in one shot";
      pipeann::IndexBuildParameters paras;

      paras.use_pipnn = (L2 != 0);
      paras.set(R, L_or_L1, 750, 1.2, num_threads, true);                // only used by Vamana.
      paras.pipnn_set(2048, 256, 0.01f, 1000, L_or_L1, L2, 2, 12, 128);  // only used by PiPNN.

      auto _pvamanaIndex = std::make_unique<pipeann::Index<T, TagT>>(_compareMetric, base_dim);
      _pvamanaIndex->build(base_file.c_str(), base_num, paras, tag_file, false, train_query_path, R_ood, L_ood);
      SSDIndex<T, TagT>::save_from_mem(*_pvamanaIndex, disk_index_path, R_dense, attr_writer);
      return 0;
    }

    if (L2 != 0) {
      LOG(ERROR) << "PiPNN requires the full index to fit in RAM. Estimated RAM usage: "
                 << full_index_ram / (1024 * 1024 * 1024) << "GB, budget given is " << ram_budget << "GB";
      exit(-1);
    }

    std::string merged_index_prefix = disk_index_path + "_tempFiles";
    int num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget, 2 * R / 3, merged_index_prefix, 2);

    // Keep shard/final R_ood proportions consistent (shard_R = 2*R/3, so shard_R_ood = 2*R_ood/3).
    const uint16_t shard_R_ood = (uint16_t) (2 * (uint32_t) R_ood / 3);
    const uint16_t shard_R_dense = (uint16_t) (2 * (uint32_t) R_dense / 3);
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_disk_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_disk.index";

      pipeann::IndexBuildParameters paras;
      paras.set(2 * R / 3, L_or_L1, 750, 1.2, num_threads, false);
      uint64_t shard_base_dim, shard_base_pts;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      auto _pvamanaIndex = std::make_unique<pipeann::Index<T, TagT>>(_compareMetric, shard_base_dim);
      _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras, (const char *) nullptr, false,
                           train_query_path, shard_R_ood, L_ood);
      SSDIndex<T, TagT>::save_from_mem(*_pvamanaIndex, shard_disk_file, shard_R_dense);
    }

    pipeann::merge_shards<T, TagT>(merged_index_prefix + "_subshard-", "_disk.index",
                                   merged_index_prefix + "_subshard-", "_ids_uint32.bin", num_parts, R, R_dense, R_ood,
                                   disk_index_path, tag_file ? std::string(tag_file) : std::string(""), attr_writer);

    // Cleanup shard artifacts.
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
      std::string shard_disk_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_disk.index";
      std::string shard_disk_tags = shard_disk_file + ".tags";

      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_disk_file.c_str());
      std::remove(shard_disk_tags.c_str());
    }
    return 0;
  }

  template<typename T, typename TagT>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, uint16_t R, uint32_t L_or_L1, uint32_t M,
                        uint32_t num_threads, uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                        const char *tag_file, AbstractNeighbor<T> *nbr_handler, AttrWriter *attr_writer,
                        uint16_t R_dense, uint32_t L2, const std::string &train_query_path, uint16_t R_ood,
                        uint32_t L_ood) {
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

    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
      normalize_data_file<T>(dataFilePath, normalized_file_path);
    }

    auto s = std::chrono::high_resolution_clock::now();
    nbr_handler->build(index_prefix_path, normalized_file_path, bytes_per_nbr);

    auto start = std::chrono::high_resolution_clock::now();
    auto p_val = nbr_handler->get_sample_p();
    nbr_handler->clear(); // free compressed vectors; object stays alive for later load().
    pipeann::build_merged_index<T, TagT>(normalized_file_path, _compareMetric, R, L_or_L1, p_val, M, disk_index_path,
                                         tag_file, R_dense, num_threads, L2, train_query_path, R_ood, L_ood,
                                         attr_writer);
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << (L2 == 0 ? "Vamana" : "PiPNN")
              << " index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (normalized_file_path != dataFilePath) {
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

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
                                                   uint16_t R_ood, uint32_t L_ood);
  template bool build_disk_index<uint8_t, uint32_t>(const char *dataPath, const char *indexFilePath, uint16_t R,
                                                    uint32_t L, uint32_t M, uint32_t num_threads,
                                                    uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                                                    const char *tag_file, AbstractNeighbor<uint8_t> *nbr_handler,
                                                    AttrWriter *label, uint16_t R_dense, uint32_t L2,
                                                    const std::string &train_query_path, uint16_t R_ood,
                                                    uint32_t L_ood);
  template bool build_disk_index<float, uint32_t>(const char *dataPath, const char *indexFilePath, uint16_t R,
                                                  uint32_t L, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                                                  pipeann::Metric _compareMetric, const char *tag_file,
                                                  AbstractNeighbor<float> *nbr_handler, AttrWriter *label,
                                                  uint16_t R_dense, uint32_t L2, const std::string &train_query_path,
                                                  uint16_t R_ood, uint32_t L_ood);

  template int build_merged_index<int8_t, uint32_t>(std::string base_file, pipeann::Metric _compareMetric, uint16_t R,
                                                    uint32_t L_or_L1, double sampling_rate, double ram_budget,
                                                    std::string disk_index_path, const char *tag_file, uint16_t R_dense,
                                                    uint32_t num_threads, uint32_t L2,
                                                    const std::string &train_query_path, uint16_t R_ood, uint32_t L_ood,
                                                    AttrWriter *attr_writer);
  template int build_merged_index<float, uint32_t>(std::string base_file, pipeann::Metric _compareMetric, uint16_t R,
                                                   uint32_t L_or_L1, double sampling_rate, double ram_budget,
                                                   std::string disk_index_path, const char *tag_file, uint16_t R_dense,
                                                   uint32_t num_threads, uint32_t L2,
                                                   const std::string &train_query_path, uint16_t R_ood, uint32_t L_ood,
                                                   AttrWriter *attr_writer);
  template int build_merged_index<uint8_t, uint32_t>(std::string base_file, pipeann::Metric _compareMetric, uint16_t R,
                                                     uint32_t L_or_L1, double sampling_rate, double ram_budget,
                                                     std::string disk_index_path, const char *tag_file,
                                                     uint16_t R_dense, uint32_t num_threads, uint32_t L2,
                                                     const std::string &train_query_path, uint16_t R_ood,
                                                     uint32_t L_ood, AttrWriter *attr_writer);

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
