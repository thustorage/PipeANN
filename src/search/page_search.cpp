#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
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
  size_t SSDIndex<T, TagT>::page_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats) {
    QueryBuffer *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }
    const T *query = query_buf->aligned_query<T>();

    // reset query
    query_buf->reset();

    // pointers to current vector for comparison
    T *data_buf = query_buf->coord_scratch<T>();
    pipeann::cpu_prefetch_t1((char *) data_buf);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    nbr_handler->initialize_query(query, query_buf);
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer, io_timer, cpu_timer;
    CandidatePool pool(l_search, l_search + 1);
    FlatVisitedSet &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);

    auto compute_exact_dists_and_push = [&](const DiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node.coords, meta_.data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist));
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](DiskNode<T> &node) {
      unsigned *node_nbrs = node.nbrs;
      unsigned nnbrs = node.nnbrs;
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.insert(node_nbrs[m])) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
        }
      }
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, node_nbrs, nbors_cand_size);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = query_buf->aligned_dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          pool.insert(Neighbor(nbor_id, nbor_dist));
        }
      }
    };

    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids) {
      nbr_handler->compute_dists(query_buf, node_ids, n_ids);
      for (uint64_t i = 0; i < n_ids; ++i) {
        pool.insert(Neighbor(node_ids[i], query_buf->aligned_dist_scratch[i]));
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    stats->io_us = 0;
    stats->cpu_us = 0;
    // search in in-memory index.
    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      compute_and_add_to_retset(&meta_.entry_point_id, 1);
    }

    unsigned num_ios = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char *>;  // <node_id, page_id, page_layout, page_buf>
    std::vector<page_fnhood_t> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<IORequest> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;  // <node_id, page_id, page_layout>
    std::vector<io_ss_t> last_io_snapshot;
    last_io_snapshot.reserve(2 * beam_width);

    std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);

    // search on disk.
    while (!pool.all_visited()) {
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;

      // Pick the beam: pull the closest unvisited nodes (next_unvisited marks each
      // visited and advances the frontier), fetching each distinct page at most
      // once. Co-located nodes on an already-picked page are still marked visited
      // here and computed via the last_io_snapshot path; termination is plain
      // all_visited().
      for (uint32_t num_seen = 0; num_seen < beam_width;) {
        Neighbor *node = pool.next_unvisited();
        if (node == nullptr) {
          break;
        }
        const unsigned pid = id2page(node->id);
        if (page_visited.find(pid) == page_visited.end()) {
          page_visited.insert(pid);
          frontier.push_back(node->id);
          ++num_seen;
        }
      }

      // read nhoods of frontier ids
      int n_ios = 0;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;

        for (uint64_t i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          uint64_t page_id = id2page(id);
          auto buf = sector_scratch + sector_scratch_idx * io_size;
          PageArr layout = get_page_layout(page_id);
          page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          // read the page to the temporary buffer
          frontier_read_reqs.emplace_back(IORequest(page_id * SECTOR_LEN, io_size, buf, page_id * SECTOR_LEN, io_size));
          if (stats != nullptr) {
            stats->n_ios++;
          }
          num_ios++;
        }

        n_ios = reader->send_read(frontier_read_reqs, ctx);
      }

      // compute remaining nodes in the pages that are fetched in the previous
      // round
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
        auto &[last_io_id, pid, page_layout] = last_io_snapshot[i];
        char *sector_buf = last_pages.data() + i * SECTOR_LEN;

        // minus one for the vector that is computed previously
        std::vector<std::pair<float, DiskNode<T>>> vis_cand;  // <dist, node>
        vis_cand.reserve(meta_.nnodes_per_sector);

        // compute exact distances of the vectors within the page
        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          const unsigned id = page_layout[j];
          if (id == last_io_id || id == kAllocatedID || id == kInvalidID) {
            continue;
          }
          DiskNode<T> node = node_from_page(sector_buf, j);
          float dist = compute_exact_dists_and_push(node, id);
          vis_cand.emplace_back(dist, node);
        }

        // compute PQ distances for neighbours of the vectors in the page
        for (unsigned k = 0; k < vis_cand.size(); ++k) {
          compute_and_push_nbrs(vis_cand[k].second);
        }
      }
      last_io_snapshot.clear();
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();

      auto io_time_st = std::chrono::high_resolution_clock::now();
      // get last submitted io results, blocking
      if (!frontier.empty()) {
        for (int i = 0; i < n_ios; ++i) {
          while (!frontier_read_reqs[i].finished) {
            reader->poll(ctx);
          }
        }
      }
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();

      auto cpu_st = std::chrono::high_resolution_clock::now();
      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
        // fill in the last_io_ids() and last_pages() with neighbor buffers.
        memcpy(last_pages.data() + last_io_snapshot.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          unsigned cur_id = layout[j];
          if (cur_id == id) {
            DiskNode<T> node = node_from_page(sector_buf, j);
            compute_exact_dists_and_push(node, id);
            compute_and_push_nbrs(node);
          }
        }
      }
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    }

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    auto t = copy_top_k(full_retset, k_search, res_tags, distances);

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
