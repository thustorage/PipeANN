#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index_defs.h"
#include "ssd_index.h"
#include "candidate_pool.h"
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
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::coro_search(T **queries, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT **res_tags, float **res_dists,
                                        const uint64_t beam_width, int N) {
    // beam search with intra-thread parallelism.
    static constexpr int kMaxCoroPerThread = 8;
    static constexpr int kMaxVectorDim = 512;
    struct alignas(SECTOR_LEN) CoroDataOne {
      // buffer.
      char sectors[SECTOR_LEN * 128];  // align to SECTOR_LEN.
      T query[kMaxVectorDim];
      T data_buf[ROUND_UP(kMaxVectorDim, 256)];
      uint64_t sector_idx;

      QueryBuffer query_buf;
      // search state.
      std::vector<Neighbor> full_retset;
      CandidatePool<> pool;
      tsl::robin_set<uint64_t> visited;

      std::vector<unsigned> frontier;
      using fnhood_t = std::tuple<unsigned, unsigned, char *>;
      std::vector<fnhood_t> frontier_nhoods;
      std::vector<IORequest> frontier_read_reqs;

      SSDIndex<T> *parent;
      unsigned cmps;
      // A batch has been issued but not yet explored. Termination must wait for
      // it: visited is set at pick time, so first_unvisited_ (all_visited) runs
      // ahead of the neighbors an outstanding batch will still insert.
      bool outstanding;

      void print() {
        LOG(INFO) << "Full retset size " << full_retset.size() << " retset size: " << pool.size()
                  << " visited size: " << visited.size() << " frontier size: " << frontier.size()
                  << " frontier nhood size: " << frontier_nhoods.size()
                  << " frontier read reqs size: " << frontier_read_reqs.size();
      }

      void reset(uint64_t l_search) {
        sector_idx = 0;
        visited.clear();  // does not deallocate memory.
        pool.reset(l_search, l_search + 1);
        full_retset.clear();
        cmps = 0;
        outstanding = false;
      }

      void compute_and_add_to_retset(const unsigned *node_ids, const uint64_t n_ids) {
        parent->nbr_handler->compute_dists(&query_buf, node_ids, n_ids);
        for (uint64_t i = 0; i < n_ids; ++i) {
          // Unfiltered search: seeds and expansion nodes share is_member_approx so
          // the pool stays monotone in distance (is_member_approx is the primary
          // sort key and is meaningful only for in-filter search).
          pool.insert(Neighbor(node_ids[i], query_buf.aligned_dist_scratch[i]));
          visited.insert(node_ids[i]);
        }
      };

      void issue_next_io_batch(const uint64_t beam_width, void *ctx) {
        if (search_ends()) {
          return;
        }
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        sector_idx = 0;

        // Pick the closest not-yet-visited candidates, marking them visited at
        // pick time. Reads land in a later step, so termination is gated on
        // `outstanding` rather than on the visited frontier alone.
        for (uint32_t num_seen = 0; num_seen < beam_width; ++num_seen) {
          Neighbor *node = pool.next_unvisited();
          if (node == nullptr) {
            break;
          }
          frontier.push_back(node->id);
        }

        // read nhoods of frontier ids
        std::vector<uint32_t> locked;
        if (!frontier.empty()) {
          for (uint64_t i = 0; i < frontier.size(); i++) {
            uint32_t loc = frontier[i];
            uint64_t offset = parent->loc_sector_no(loc) * SECTOR_LEN;
            auto sector_buf = sectors + sector_idx * parent->io_size;
            fnhood_t fnhood = std::make_tuple(loc, loc, sector_buf);
            sector_idx++;
            frontier_nhoods.push_back(fnhood);

            frontier_read_reqs.emplace_back(IORequest(offset, parent->io_size, sector_buf, 0, 0));
          }
          parent->reader->send_io(frontier_read_reqs, ctx, false);
          outstanding = true;
        }
      }

      bool io_finished(void *ctx) {
        parent->reader->poll(ctx);
        for (auto &req : frontier_read_reqs) {
          if (!req.finished) {
            return false;
          }
        }
        return true;
      }

      void explore_frontier(uint64_t l_search) {
        for (auto &frontier_nhood : frontier_nhoods) {
          auto [id, loc, sector_buf] = frontier_nhood;
          auto node = parent->node_from_page(sector_buf, loc);

          T *node_fp_coords_copy = data_buf;
          memcpy(node_fp_coords_copy, node.coords, parent->meta_.data_dim * sizeof(T));
          float cur_expanded_dist =
              parent->dist_cmp->compare(query, node_fp_coords_copy, (unsigned) parent->aligned_dim);

          Neighbor n(id, cur_expanded_dist);
          full_retset.push_back(n);

          // compute node_nbrs <-> query dist in PQ space
          parent->nbr_handler->compute_dists(&query_buf, node.nbrs, node.nnbrs);

          // process prefetch-ed nhood
          for (uint64_t m = 0; m < node.nnbrs; ++m) {
            unsigned id = node.nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              float dist = query_buf.aligned_dist_scratch[m];
              pool.insert(Neighbor(id, dist));
            }
          }
        }
        outstanding = false;
      }

      bool search_ends() {
        // this->print();
        return pool.all_visited() && !outstanding;
      }
    };

    struct alignas(4096) CoroData {
      CoroDataOne data[kMaxCoroPerThread];
      CoroData(SSDIndex<T> *parent) {
        for (int i = 0; i < kMaxCoroPerThread; ++i) {
          data[i].parent = parent;
          parent->init_query_buf(data[i].query_buf);
        }
      }
    };

    static __thread CoroData *data;
    if (unlikely(data == nullptr)) {
      data = new CoroData(this);
    }

    if (unlikely(N > kMaxCoroPerThread)) {
      LOG(ERROR) << "N > kMaxCoroPerThread";
      exit(-1);
    }

    // do not use the thread data's buf.
    QueryBuffer *thread_data = pop_query_buf(queries[0]);
    void *ctx = reader->get_ctx();
    // lambda to batch compute query<-> node distances in PQ space

    for (int v = 0; v < N; ++v) {
      auto &coro_data = data->data[v];
      auto &query1 = queries[v];
      memcpy(coro_data.query, query1, this->meta_.data_dim * sizeof(T));

      auto &query = coro_data.query;

      // pointers to buffers for data
      T *data_buf = coro_data.data_buf;
      pipeann::cpu_prefetch_t1((char *) data_buf);

      // query <-> PQ chunk centers distances
      nbr_handler->initialize_query(query, &coro_data.query_buf);

      coro_data.reset(l_search);

      if (mem_L) {
        std::vector<unsigned> mem_tags(mem_L);
        std::vector<float> mem_dists(mem_L);
        mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
        coro_data.compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
      } else {
        // Do not use optimized start point.
        coro_data.compute_and_add_to_retset(&meta_.entry_point_id, 1);
      }
    }

    // SEARCH!
    for (int i = 0; i < N; ++i) {
      auto &coro_data = data->data[i];
      coro_data.issue_next_io_batch(beam_width, ctx);
    }

    bool all_finished = false;
    while (!all_finished) {
      all_finished = true;
      for (int i = 0; i < N; ++i) {
        auto &coro_data = data->data[i];
        if (!coro_data.search_ends()) {
          all_finished = false;
          if (!coro_data.io_finished(ctx)) {
            continue;
          }
          // LOG(INFO) << "Full retset size: " << coro_data.full_retset.size();
          coro_data.explore_frontier(l_search);
          coro_data.issue_next_io_batch(beam_width, ctx);
        }
      }
    }

    for (int v = 0; v < N; ++v) {
      // re-sort by distance
      auto &full_retset = data->data[v].full_retset;
      std::sort(full_retset.begin(), full_retset.end(),
                [](const Neighbor &left, const Neighbor &right) { return left < right; });

      uint64_t t = 0;
      for (uint64_t i = 0; i < full_retset.size() && t < k_search; i++) {
        if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
          continue;  // deduplicate.
        }
        res_tags[v][t] = full_retset[i].id;  // use ID to replace tags
        if (res_dists[v] != nullptr) {
          res_dists[v][t] = full_retset[i].distance;
        }
        t++;
      }
    }

    this->push_query_buf(thread_data);
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
