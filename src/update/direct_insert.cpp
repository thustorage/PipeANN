#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <random>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"
#include "utils/prune_neighbors.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  #define UPDATE_BUF_SIZE ((2 * MAX_N_EDGES + 1) * io_size)

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place(const T *point1, const TagT &tag, const Attributes *attrs) {
    QueryBuffer *read_data = this->pop_query_buf(point1);
    T *point = read_data->aligned_query<T>();  // normalized point for cosine.
    void *ctx = reader->get_ctx(); // initialize ctx here, avoid SQ polling for insert.

    uint32_t target_id = cur_id++;

    // write neighbor (e.g., PQ).
    nbr_handler->insert(point, target_id);

    std::vector<Neighbor> exp_node_info;
    InsertContext insert_ctx(kExpandedNodesFactor * this->params.L, this->aligned_dim);
    this->do_pipe_search(point1, params.L, 0, params.L, params.beam_width, exp_node_info, nullptr, &insert_ctx);
    std::vector<uint32_t> new_nhood;

    // SPACEV1B frequently chooses entry_point as a neighbor, which causes contention.
    auto params_copy = params; // < 100 bytes, copy should be fast.
    params_copy.R += 1;
    pipeann::prune_neighbors(exp_node_info, new_nhood, params_copy, metric, [this, &insert_ctx](uint32_t a, uint32_t b) {
      return this->dist_cmp->compare(insert_ctx.coord_map[a], insert_ctx.coord_map[b], this->meta_.data_dim);
    });

    // Produce one more neighbor, and remove entry_point in the final results.
    // Only points close enough to the entry point can connect to it.
    off_t medoid_threshold = params.R / 4;
    auto it = std::find(new_nhood.begin(), new_nhood.end(), this->meta_.entry_point_id);
    if (it != new_nhood.end() && it - new_nhood.begin() >= medoid_threshold) {
      *it = new_nhood.back(); // swap it with the back.
      new_nhood.pop_back();
    }

    // Keep R neighbors. The last nbr is likely to be for saturation,
    // so removing it does not affect graph quality.
    if (new_nhood.size() > params.R) {
      new_nhood.pop_back();
    }

    std::set<uint64_t> pages_need_to_read;

    // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.
#ifdef IN_PLACE_RECORD_UPDATE
    std::vector<uint64_t> locs;
    for (auto &nbr : new_nhood) {
      auto loc = id2loc(nbr);
      locs.emplace_back(loc);
      pages_need_to_read.insert(loc_sector_no(loc));
    }
    locs.push_back(target_id);
    pages_need_to_read.insert(loc_sector_no(target_id));
    set_id2loc(target_id, target_id);

    // update loc2id, target_id <-> target_id.
    cur_loc++;  // for target ID, atomic update.
    set_loc2id(target_id, target_id);
#else
    std::vector<uint64_t> locs = this->alloc_loc(new_nhood.size() + 1, insert_ctx.hint_pages, pages_need_to_read);
#endif

    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, io_size, nullptr, 0, 0));
    }
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto pages_locked = pipeann::lockReqs(this->page_lock_table, pages_to_rmw);
    lock_vec(vec_lock_table, target_id, new_nhood);

    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    // dynamically allocate update_buf to reduce memory footprint.
    // 2x MAX_N_EDGES for read + write, the update_buf is freed in bg_io_thread.
    assert(read_data->update_buf == nullptr);
    read_data->update_buf = (char *) reader->alloc_io_buf(UPDATE_BUF_SIZE, SECTOR_LEN);
    auto &update_buf = read_data->update_buf;

    std::vector<IORequest> reads, writes_4k, writes;
    assert(new_nhood.size() < MAX_N_EDGES);
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto loc = id2loc(new_nhood[i]);
      reads.push_back(IORequest(loc_sector_no(loc) * SECTOR_LEN, io_size, update_buf + i * io_size, 0, 0));
      page_buf_map[loc_sector_no(loc)] = update_buf + i * io_size;
    }

    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      writes_4k.push_back(IORequest(off, io_size, update_buf + i * io_size, 0, 0));
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
        reads.push_back(IORequest(off, io_size, update_buf + i * io_size, 0, 0));
      }
      page_buf_map[off / SECTOR_LEN] = update_buf + i * io_size;
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
    uint64_t start_idx = 0;
    uint64_t cur_off = writes_4k[0].offset;
    for (uint32_t i = 1; i < writes_4k.size(); ++i) {
      if (writes_4k[i].offset != cur_off + io_size) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, io_size * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
        start_idx = i;
      }
      cur_off = writes_4k[i].offset;
    }
    writes_4k.pop_back();

    reader->read_alloc(reads, ctx, &insert_ctx.page_ref);

    // update the target node.
    auto sector = loc_sector_no(locs[new_nhood.size()]);
    DiskNode<T> target_node = node_from_page(page_buf_map[sector], locs[new_nhood.size()]);
    memcpy(target_node.coords, point, meta_.data_dim * sizeof(T));
    target_node.nnbrs = new_nhood.size();
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);
    track_tag2id(target_id, tag);

    // write attributes to the target node if provided.
    if (attrs != nullptr && meta_.attr_size > 0) {
      if (unlikely(meta_.attr_size < attrs->serialized_size())) {
        LOG(ERROR) << "Attribute size " << meta_.attr_size << " is smaller than serialized attribute size "
                   << attrs->serialized_size();
        exit(-1);
      }
      attrs->serialize((char *) target_node.attrs);
    }

    // Collect 2-hop neighbors from the neighbors' neighbor lists.
    uint16_t max_dense = meta_.range_dense > meta_.range ? (uint16_t) (meta_.range_dense - meta_.range) : 0;
    if (max_dense > 0) {
      target_node.n_dense_nbrs = sample_two_hop_nbrs(
          target_id, max_dense, target_node.dense_nbrs, [&](uint32_t id) -> std::pair<const uint32_t *, size_t> {
            if (id == target_id) {
              return {new_nhood.data(), new_nhood.size()};
            }
            auto loc = id2loc(id);
            auto r_sector = loc_sector_no(loc);
            DiskNode<T> nbr_node = node_from_page(page_buf_map[r_sector], loc);
            /* owned by page_buf_map, read-only in sample_two_hop_nbrs. */
            return {nbr_node.nbrs, nbr_node.nnbrs};
          });

    } else {
      target_node.n_dense_nbrs = 0;
    }

    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto loc = id2loc(new_nhood[i]);
      auto r_sector = loc_sector_no(loc);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        LOG(ERROR) << new_nhood[i] << " "
                   << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }

      DiskNode<T> r_nbr_node = node_from_page(page_buf_map[r_sector], loc);
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      if (nhood.size() > this->params.R) {  // delta prune neighbors
        auto &thread_pq_buf = read_data->nbr_vec_scratch;
        auto nbr = this->nbr_handler;
        pipeann::delta_prune_neighbors(
            nhood, new_nhood[i], target_id, params, metric,
            [nbr, &thread_pq_buf](uint32_t center, const uint32_t *ids, uint32_t n, float *dists_out) {
              nbr->compute_dists(center, ids, n, dists_out, thread_pq_buf);
            });
      }

      auto w_sector = loc_sector_no(locs[i]);
      DiskNode<T> w_nbr_node = node_from_page(page_buf_map[w_sector], locs[i]);
      // Neighbor relocation must preserve the full node payload, including attrs.
      memcpy(w_nbr_node.coords, r_nbr_node.coords, meta_.max_node_len);
      w_nbr_node.nnbrs = nhood.size();  // write to nnbrs reference; n_dense_nbrs preserved by memcpy.
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
    }

    std::vector<uint64_t> write_page_ref;
    reader->wbc_write(writes, ctx, &write_page_ref);

#ifndef IN_PLACE_RECORD_UPDATE
    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    set_id2loc(target_id, locs[new_nhood.size()]);
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      set_id2loc(new_nhood[i], locs[i]);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    new_nhood.push_back(target_id);
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_idx(idx_lock_table, locked);
#endif
    unlock_vec(vec_lock_table, target_id, new_nhood);

    // commit writes (in the background thread.)
#ifdef BG_IO_THREAD
    if (!insert_ctx.page_ref.empty()) {
      auto bg_task = new BgTask{.thread_data = read_data,
                                .writes = std::move(writes),
                                .pages_to_unlock = std::move(pages_locked),
                                .pages_to_deref = std::move(write_page_ref),
                                .terminate = false};
      bg_tasks.push(bg_task);
      bg_tasks.push_notify_all();
    } else {
      pipeann::unlockReqs(this->page_lock_table, pages_locked);
    }
    reader->deref(&insert_ctx.page_ref);
#else
    reader->write(writes, ctx);
    reader->free_io_buf(read_data->update_buf, UPDATE_BUF_SIZE);
    read_data->update_buf = nullptr;

    pipeann::unlockReqs(this->page_lock_table, pages_locked);
    reader->deref(&write_page_ref);

    reader->deref(&insert_ctx.page_ref);
    this->push_query_buf(read_data);
#endif
    return target_id;
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    auto ctx = reader->get_ctx();
    auto timer = pipeann::Timer();
    uint64_t n_tasks = 0;

    while (true) {
      auto task = bg_tasks.pop();
      while (task == nullptr) {
        this->bg_tasks.wait_for_push_notify();
        task = bg_tasks.pop();
      }

      if (unlikely(task->terminate)) {
        delete task;
        break;
      }

      reader->write(task->writes, ctx);
      reader->free_io_buf(task->thread_data->update_buf, UPDATE_BUF_SIZE);
      task->thread_data->update_buf = nullptr;

      pipeann::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref);
      this->push_query_buf(task->thread_data);
      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
