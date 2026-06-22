#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#include "utils/page_cache.h"
#include "utils/prune_neighbors.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  libcuckoo::cuckoohash_map<uint32_t, uint32_t> SSDIndex<T, TagT>::merge_deletes(
      const std::string &in_path_prefix, const std::string &out_path_prefix, const std::vector<TagT> &deleted_nodes,
      const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads) {
    const uint32_t n_sampled_nbrs = this->params.sampled_nbrs_for_delete;
    if (nthreads == 0) {
      LOG(INFO) << "nthreads for merge unspecified, using max threads: " << omp_get_max_threads();
      nthreads = omp_get_max_threads();
    }

    void *ctx = reader->get_ctx();

    while (!bg_tasks.empty()) {
      sleep(5);  // simple way to wait for background IO thread.
    }
    std::string disk_index_out = out_path_prefix + "_disk.index";
    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    libcuckoo::cuckoohash_map<uint32_t, uint32_t> id_map, rev_id_map;           // old_id -> new_id & new_id -> old_id
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    std::atomic<uint64_t> new_npoints = 0;
    Timer delete_timer;

    constexpr uint64_t LOC_PER_POPULATE = 128 * 8;  // small to avoid blocking search threads.
    // process this many locs per iteration
    const uint64_t LOC_PER_MERGE = meta_.nnodes_per_sector > 0 ? ROUND_UP(131072, meta_.nnodes_per_sector) : 131072;
    const uint64_t VEC_IN_WBUF = 2 * LOC_PER_MERGE;  // sliding window buffer holds 2 batches

    // Calculate buffer size based on whether we have large or small nodes
    uint64_t sectors_per_batch = meta_.nnodes_per_sector > 0
                                     ? LOC_PER_MERGE / meta_.nnodes_per_sector
                                     : LOC_PER_MERGE * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
    uint64_t buf_size_per_batch = sectors_per_batch * SECTOR_LEN;

    char *rbuf = (char *) reader->alloc_io_buf(buf_size_per_batch, SECTOR_LEN);
    char *wbuf = (char *) reader->alloc_io_buf(2 * buf_size_per_batch, SECTOR_LEN);  // sliding window buffer.
    LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id
              << ", nnodes_per_sector: " << meta_.nnodes_per_sector
              << ", buf_size_per_batch: " << buf_size_per_batch / 1024 / 1024 << "MB";

    uint32_t populate_nthreads = std::min(nthreads, 4u);  // restrict the flow.

    for (uint64_t loc_st = 0; loc_st < cur_loc; loc_st += LOC_PER_POPULATE) {
      uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_POPULATE);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
      reader->read(read_reqs, ctx);

#pragma omp parallel for num_threads(populate_nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        // populate nhood.
        uint64_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, alloc ID.
          // allocate ID.
          uint64_t new_id = new_npoints.fetch_add(1);
          id_map.insert(id, new_id);
          rev_id_map.insert(new_id, id);
          continue;
        }

        // 3. deleted, populate nhoods.
        uint64_t loc_sector = loc_sector_no(loc);
        auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
        DiskNode<T> node = node_from_page(page_rbuf, loc);
        std::vector<uint32_t> nhood;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
            nhood.push_back(node.nbrs[i]);  // filtered neighborhoods.
          }
        }
        // sample for less space consumption.
        if (nhood.size() > n_sampled_nbrs) {
          // std::shuffle(nhood.begin(), nhood.end(), std::default_random_engine());
          nhood.resize(n_sampled_nbrs);  // nearest.
        }
        deleted_nhoods.insert(id, nhood);
      }
    }
    LOG(INFO) << "Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms, new npoints: " << new_npoints.load() << " "
              << "id_map size: " << id_map.size();
    LOG(INFO) << "Deleted nodes size: " << deleted_nodes.size() << ", deleted_nhoods size: " << deleted_nhoods.size();

    // Step 2: prune neighbors, populate PQ and tags.
    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
    uint64_t wb_id = 0;
    std::atomic<uint64_t> n_used_id = 0;
    auto write_back = [&]() {
      // write one buffer.
      uint64_t buf_id = (wb_id % VEC_IN_WBUF) / LOC_PER_MERGE;
      auto b = wbuf + buf_id * buf_size_per_batch;
      std::vector<IORequest> write_reqs;
      uint64_t id_delta = std::min(LOC_PER_MERGE, n_used_id - wb_id);

      // Calculate write length: from wb_id to wb_id + id_delta - 1
      uint64_t st_sector = loc_sector_no(wb_id);
      uint64_t ed_sector = loc_sector_no(wb_id + id_delta - 1);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t write_len = (ed_sector - st_sector + 1) * SECTOR_LEN;

      write_reqs.push_back(IORequest(st_sector * SECTOR_LEN, write_len, b, 0, 0));
      reader->write_fd(fd, write_reqs, ctx);
      wb_id += id_delta;
      LOG(INFO) << "Write back " << wb_id << "/" << n_used_id << " IDs.";
    };

    std::vector<TagT> new_tags(new_npoints);

    for (uint64_t loc_st = 0; loc_st < cur_loc; loc_st += LOC_PER_MERGE) {
      uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_MERGE);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
      reader->read(read_reqs, ctx);  // read in fd

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted.
          continue;
        }

        uint64_t loc_sector = loc_sector_no(loc);
        auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
        DiskNode<T> node = node_from_page(page_rbuf, loc);
        // prune neighbors.
        std::unordered_set<uint32_t> nhood_set;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
            // deleted, insert neighbors.
            const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
            nhood_set.insert(nhoods.begin(), nhoods.end());
          } else {
            nhood_set.insert(node.nbrs[i]);
            // LOG(INFO) << id << " insert " << node.nbrs[i];
          }
        }
        nhood_set.erase(id);  // remove self.
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->params.R) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          // Use dynamic buffer instead of pre-initialized buffer to save space.
          uint8_t *pq_buf = nullptr;
          pipeann::alloc_aligned((void **) &pq_buf, nhood.size() * AbstractNeighbor<T>::MAX_BYTES_PER_NBR, 256);
          nbr_handler->compute_dists(id, nhood.data(), nhood.size(), dists.data(), pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          auto nbr = this->nbr_handler;
          pipeann::prune_neighbors(pool, nhood, params, metric, [nbr, pq_buf](uint32_t a, uint32_t b) {
            float dist;
            nbr->compute_dists(a, &b, 1, &dist, pq_buf);
            return dist;
          });
          pipeann::aligned_free(pq_buf);
        }

        // map to new IDs.
        for (auto &nbr : nhood) {
          nbr = id_map.find(nbr);
          if (unlikely(nbr > new_npoints)) {
            LOG(ERROR) << "Invalid neighbor ID: " << nbr << ", new_npoints: " << new_npoints;
          }
        }

        // write neighbors.
        uint64_t new_id = id_map.find(id);
        uint64_t off = new_id % VEC_IN_WBUF;
        // loc_sector_no(off) - loc_sector_no(0) == loc_sector_no(new_id) - loc_sector_no(start_id)
        auto page_wbuf = wbuf + (loc_sector_no(off) - loc_sector_no(0)) * SECTOR_LEN;
        DiskNode<T> w_node = node_from_page(page_wbuf, new_id);

        memcpy(w_node.coords, node.coords, meta_.data_dim * sizeof(T));
        w_node.nnbrs = nhood.size();
        *(w_node.nbrs - 1) = w_node.nnbrs;
        memcpy(w_node.nbrs, nhood.data(), w_node.nnbrs * sizeof(uint32_t));
        // Keep embedded attrs so exact filter verification still works after save+reload.
        if (meta_.attr_size > 0) {
          memcpy(w_node.attrs, node.attrs, meta_.attr_size);
        }
        ++n_used_id;
        // copy tags.
        new_tags[new_id] = id2tag(id);
      }

      LOG(INFO) << "Processed " << loc_ed << "/" << cur_loc << " locs, n_used_id: " << n_used_id << ".";
      if (n_used_id - wb_id >= LOC_PER_MERGE) {
        write_back();
      }
    }

    while (wb_id < n_used_id) {
      write_back();
    }
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    // duplicate neighbor handler.
    auto new_nbr_handler = this->nbr_handler->shuffle(rev_id_map, new_npoints, nthreads);

    while (deleted_nodes_set.find(id2tag(meta_.entry_point)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node. Medoid ID: " << meta_.entry_point
                << " tag: " << id2tag(meta_.entry_point);
      const auto &nhoods = deleted_nhoods.find(meta_.entry_point);
      meta_.entry_point = nhoods[0];
    }
    close(fd);
    // free buf
    reader->free_io_buf(rbuf);
    reader->free_io_buf(wbuf);
    
    // set metadata, PQ and tags.
    merge_lock.lock();  // unlock in reload().
    // metadata.
    this->meta_.npoints = new_npoints;
    this->meta_.entry_point = id_map.find(meta_.entry_point);
    // PQ.
    auto tmp = this->nbr_handler;
    this->nbr_handler = new_nbr_handler;
    delete tmp;
    // tags.
    tags.clear();
    if (enable_tag2id) tag2id_.clear();
    // no need to clear id2loc & loc2id as they are arrays.
    // out-of-bound loc2id is initialized in reload().
#pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < new_tags.size(); ++i) {
      tags.insert_or_assign(i, new_tags[i]);
      track_tag2id(i, new_tags[i]);
      set_id2loc(i, i);
      set_loc2id(i, i);
    }

    this->write_metadata_and_pq(in_path_prefix, out_path_prefix, &new_tags);
    LOG(INFO) << "Write metadata and PQ finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    return id_map;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                                std::vector<TagT> *new_tags) {
    // Use meta_ directly since it's already been updated
    meta_.print();
    std::string disk_index_out = out_path_prefix + "_disk.index";
    meta_.save_to_disk_index(disk_index_out);

    // Step 3. Write tags and PQ.
    std::vector<TagT> tags_vec;
    if (new_tags == nullptr) {
      tags_vec.resize(meta_.npoints);
      for (uint64_t i = 0; i < meta_.npoints; ++i) {
        tags_vec[i] = id2tag(i);
      }
      new_tags = &tags_vec;
    }
    pipeann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), meta_.npoints, 1, 0);
    nbr_handler->save(out_path_prefix.c_str());
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::reload(const char *index_prefix) {
    std::string iprefix = std::string(index_prefix);
    std::string disk_index_file = iprefix + "_disk.index";
    this->disk_index_file = disk_index_file;

    reader->close();
    reader->open(disk_index_file, true, false);

    // reload metadata.
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // No need to reload PQ, as it is already reloaded in merge_deletes.
    for (uint32_t i = this->meta_.npoints; i < this->cur_loc; ++i) {
      set_loc2id(i, kInvalidID);  // reset loc2id.
    }
    while (!this->empty_pages.empty()) {
      this->empty_pages.pop();
    }
    merge_lock.unlock();
    LOG(INFO) << "Reload finished, cur_id: " << this->cur_id << ", cur_loc: " << this->cur_loc;
    return;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
