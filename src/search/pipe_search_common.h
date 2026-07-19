#pragma once
// Included by pipe_search.cpp and spec_filter_search.cpp which provide all necessary headers.

#include <limits>
#include <type_traits>
#ifdef USE_URING
#include "liburing.h"
#endif

#include "candidate_pool.h"

namespace pipeann {

  // Common pipelined search loop used by pipe_search, spec_postfilter_search, and spec_infilter_search.
  //
  // Template callbacks:
  //   SpecFn: (unsigned id) -> bool
  //     Approximate membership check for neighbor filtering during graph traversal.
  //     pipe_search / post_filter: always return true.
  //     in_filter: selector->is_member_approx(id, query_attrs).
  //
  //   VerifyFn: (unsigned id, const DiskNode<T>& node) -> bool
  //     Exact membership verification after reading a node from disk.
  //     pipe_search: always return true.
  //     filter search: selector->is_member(id, query_attrs, target_attrs).
  //
  //   PrefetchFn: (unsigned id) -> void
  //     Prefetch the memory is_member_approx(id) will read. Called a few neighbors
  //     ahead of the membership check to hide the random-access miss.
  //     pipe_search / post_filter: no-op.
  //     in_filter: selector->prefetch_approx(id, query_attrs).
  //
  template<typename T, typename TagT>
  template<typename SpecFn, typename VerifyFn, typename PrefetchFn>
  void SSDIndex<T, TagT>::pipe_search_common(const T *query1, uint64_t k_search, uint32_t mem_L, uint64_t l_search,
                                             uint64_t l_pool, uint64_t beam_width, bool use_dense_nbrs,
                                             SpecFn is_member_approx, VerifyFn is_member, PrefetchFn prefetch_approx,
                                             std::vector<Neighbor> &full_retset, QueryStats *stats,
                                             InsertContext *insert_ctx, float range_partial, NodeOut *node_out) {
    QueryBuffer *query_buf = pop_query_buf(query1);
#ifdef USE_URING
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);
#else
    void *ctx = reader->get_ctx();
#endif

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }
    const uint64_t read_io_size = use_dense_nbrs ? io_size_dense : io_size;

    const T *query = query_buf->aligned_query<T>();
    query_buf->reset();

    T *data_buf = query_buf->coord_scratch<T>();
    pipeann::cpu_prefetch_t1((char *) data_buf);

    char *sector_scratch = query_buf->sector_scratch;
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer;

    auto &visited = *(query_buf->visited);
    full_retset.reserve(l_pool * kExpandedNodesFactor);

    // Maps ids whose page has arrived to the parsed node. The candidate pool
    // consults it (through the ready predicate below) to pick the next candidate
    // to exact-rank and to skip re-reads.
    std::unordered_map<unsigned, DiskNode<T>> id_buf_map;
    // Sized to the deepest plausible set of landed-but-unexpanded pages so the
    // hot-loop inserts never rehash.
    id_buf_map.reserve(MAX_N_SECTOR_READS);

    // The candidate pool (approx members = TP+FP, plus promoted bridges), capped
    // at l_pool = in_filter_l_eq so it has room for the FP headroom needed to
    // settle l_search EXACT members (terminate() counts exact is_member).
    CandidatePool pool(l_pool, mem_L + this->params.R + l_pool * kExpandedNodesFactor,
                       [&id_buf_map](unsigned id) { return id_buf_map.find(id) != id_buf_map.end(); });

    double member_frac_ema = 0.0;
    // Promote a candidate only if it ranks in the top kBridgeFrac of the pool,
    // and only while the query walks a region locally starved of members.
    constexpr double kBridgeFrac = 0.05;
    constexpr double kBandDensityGate = 0.15;
    // cand_pool is a small pool of the closest non-member connectivity candidates.
    const unsigned kCandCap = std::max<unsigned>(4u, (unsigned) (kBridgeFrac * l_search));
    CandidatePool cand_pool(kCandCap, kCandCap + 1);

    // --- Exact distance computation + membership verification ---
    uint64_t coord_buf_idx = 0;
    auto compute_exact_dists_and_push = [&](const DiskNode<T> &node, const unsigned id) -> bool {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node.coords, meta_.data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

      if (!is_member(id, node)) {
        return false;
      }

      full_retset.push_back(Neighbor(id, cur_expanded_dist));

      if (insert_ctx != nullptr) {
        if (unlikely(coord_buf_idx >= kExpandedNodesFactor * l_pool)) {
          LOG(ERROR) << "Please allocate larger coord_buf.";
          crash();
        }
        T *coord_ptr = insert_ctx->coord_buf + coord_buf_idx * aligned_dim;
        memcpy(coord_ptr, node.coords, meta_.data_dim * sizeof(T));
        insert_ctx->coord_map.insert(std::make_pair(id, coord_ptr));
        coord_buf_idx++;
      }

      // Milvus output_fields passthrough: stash this matched node's vector +
      // filterable attrs keyed by tag, so the caller can serve output_fields
      // without re-reading the page. Collected for every membership-passing node
      // (more than top-k, but the map is small relative to l_pool and the caller
      // only reads the top-k tags it returns).
      if (node_out != nullptr) {
        NodePayload payload;
        payload.coords.assign(node.coords, node.coords + meta_.data_dim);
        if (meta_.attr_size > 0)
          payload.attrs = Attributes::deserialize((char *) node.attrs);
        node_out->insert(std::make_pair(id2tag(id), std::move(payload)));
      }
      return true;
    };

    // --- Unified neighbor expansion (1-hop approx -> 2-hop approx -> 1-hop connectivity) ---
    uint64_t n_computes = 0;
    auto compute_and_push_nbrs = [&](DiskNode<T> &node) {
      unsigned nbors_cand_size = 0;
      uint32_t *nbr_ids = query_buf->nbr_id_scratch;
      // Count matching neighbors among those scanned (folded into the collection
      // loops, no extra pass): match/seen is this node's local member density.
      uint64_t seen = 0, match = 0;

      // Prefetch-ahead distance: is_member_approx randomly probes a per-vector array
      // (bucket id / bloom filter) sized with the dataset, so each check risks a cache
      // miss. Prefetching the neighbor kPrefetchDist ahead overlaps that miss with the
      // current check's work. No-op for non-filter searches (prefetch_approx is empty).
      // 8 measured best on SIFT1B (deeper prefetch evicts before use).
      constexpr unsigned kPrefetchDist = 8;

      // 1-hop neighbors that are approximate members.
      for (unsigned m = 0; m < node.nnbrs && m < kPrefetchDist; ++m) {
        prefetch_approx(node.nbrs[m]);
        visited.prefetch(node.nbrs[m]);
      }

      for (unsigned m = 0; m < node.nnbrs && nbors_cand_size < node.nnbrs; ++m, ++seen) {
        if (m + kPrefetchDist < node.nnbrs) {
          prefetch_approx(node.nbrs[m + kPrefetchDist]);
          visited.prefetch(node.nbrs[m + kPrefetchDist]);
        }
        if (is_member_approx(node.nbrs[m])) {
          ++match;
          if (visited.insert(node.nbrs[m])) {
            nbr_handler->prefetch(node.nbrs[m]);
            nbr_ids[nbors_cand_size++] = node.nbrs[m];
          }
        }
      }

      unsigned n_member = nbors_cand_size;

      // Only valid for in-filter search where dense neighbors were actually read from disk.
      if (use_dense_nbrs) {
        // 2-hop (dense) neighbors that are approximate members.
        for (unsigned m = 0; m < node.n_dense_nbrs && m < kPrefetchDist; ++m) {
          prefetch_approx(node.dense_nbrs[m]);
          visited.prefetch(node.dense_nbrs[m]);
        }

        for (unsigned m = 0; m < node.n_dense_nbrs && nbors_cand_size < node.nnbrs; ++m, ++seen) {
          if (m + kPrefetchDist < node.n_dense_nbrs) {
            prefetch_approx(node.dense_nbrs[m + kPrefetchDist]);
            visited.prefetch(node.dense_nbrs[m + kPrefetchDist]);
          }
          if (is_member_approx(node.dense_nbrs[m])) {
            ++match;
            if (visited.insert(node.dense_nbrs[m])) {
              nbr_handler->prefetch(node.dense_nbrs[m]);
              nbr_ids[nbors_cand_size++] = node.dense_nbrs[m];
            }
          }
        }
        n_member = nbors_cand_size;
        // Remaining 1-hop neighbors for connectivity.
        for (unsigned m = 0; m < node.nnbrs && nbors_cand_size < node.nnbrs; ++m) {
          if (visited.insert(node.nbrs[m])) {
            nbr_handler->prefetch(node.nbrs[m]);
            nbr_ids[nbors_cand_size++] = node.nbrs[m];
          }
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, nbr_ids, nbors_cand_size);
        if (stats != nullptr) {
          stats->n_cmps += nbors_cand_size;
        }

        // Approx members (TP+FP) [0, n_member) -> candidate pool, merged as one
        // sorted batch so the pool tail shifts once per expansion instead of
        // once per accepted candidate.
        pool.insert_batch(nbr_ids, dist_scratch, n_member);

        // Non-member connectivity candidates -> cand_pool (closest kCandCap kept).
        for (unsigned m = n_member; m < nbors_cand_size; ++m) {
          cand_pool.insert(Neighbor(nbr_ids[m], dist_scratch[m]));
        }

        // Lightweight edge-fill: on a sparse step (density EMA below gate),
        // promote EVERY cand_pool entry within the top-kBridgeFrac distance band
        // of the pool, so enough close bridges enter the frontier.
        member_frac_ema = 0.2 * ((double) match / (double) seen) + 0.8 * member_frac_ema;
        if (cand_pool.size() > 0 && member_frac_ema < kBandDensityGate) {
          unsigned bi = std::min<unsigned>(pool.size() - 1, (unsigned) (l_search * kBridgeFrac));
          const float band = pool[bi].distance;  // fixed threshold for this expansion
          unsigned promoted = 0;
          cand_pool.scan([&](const Neighbor &c) {
            if (c.distance > band) {
              return true;  // sorted ascending -> the rest are farther
            }
            pool.insert(c);
            ++promoted;
            return false;
          });
          cand_pool.drop_front(promoted);  // drop the promoted (closest) prefix
        }
      }
    };

    // --- Stats init ---
    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }

    // --- In-memory index initialization ---
    int64_t cur_beam_width = std::min<uint64_t>(4, beam_width);
    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

    if (mem_L) {
      // Seed with the mem-index distances (already ascending) so the first beam
      // reads can go out before the PQ query table is built; the build + seed
      // re-scoring below then overlap the first SSD read.
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      const uint64_t n_seed = std::min<uint64_t>(mem_L, l_pool);
      for (uint64_t i = 0; i < n_seed; ++i) {
        visited.insert(mem_tags[i]);
        pool.seed(Neighbor(mem_tags[i], mem_dists[i]));
      }
    } else {
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      visited.insert(meta_.entry_point_id);
      pool.insert(Neighbor(meta_.entry_point_id, dist_scratch[0]));
    }

    // --- IO pipeline ---
    struct io_t {
      Neighbor nbr;
      unsigned page_id;
      unsigned loc;
      IORequest *read_req;
      uint32_t retset_idx;  // index into retset for rerank-only reads (UINT32_MAX during graph search)
      bool operator>(const io_t &rhs) const {
        return nbr.distance > rhs.nbr.distance;
      }
      bool operator<(const io_t &rhs) const {
        return nbr.distance < rhs.nbr.distance;
      }
      bool finished() {
        return read_req->finished;
      }
    };
    std::queue<io_t> on_flight_ios;

    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;
      this->lock_idx(idx_lock_table, item.id, true);
      const unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * read_io_size;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, read_io_size, buf, u_loc_offset(loc), meta_.max_node_len,
                      sector_scratch);
      reader->send_read(req, ctx);

      on_flight_ios.push(io_t{item, pid, loc, &req, UINT32_MAX});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios += (double) read_io_size / SECTOR_LEN;
      }
      return true;
    };

    auto poll_all = [&]() -> std::pair<int, int> {
      if (insert_ctx != nullptr) {
        reader->poll_alloc(ctx, &insert_ctx->page_ref);
      } else {
        reader->poll(ctx);
      }
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        id_buf_map.insert(std::make_pair(io.nbr.id, node_from_page((char *) io.read_req->buf, io.loc)));
        if (insert_ctx != nullptr) {
          insert_ctx->hint_pages.push_back(io.page_id);
        }
        io.nbr.distance <= pool.worst().distance ? ++n_in : ++n_out;
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0;
      while (n_sent < n) {
        Neighbor *item = pool.next_read_nbr();
        if (item == nullptr) {
          break;
        }
        n_sent += send_read_req(*item);
      }
      return n_sent != 0;
    };

    // --- Node selection ---
    // Highest frontier marker reached so far; gates rerank expansion (read by
    // calc_best_node, advanced by the main loop), so it must outlive the lambda.
    unsigned max_marker = 0;
    auto calc_best_node = [&]() -> int {
      // next_calc_nbr can only find work if some landed page is still
      // unexpanded (id_buf_map non-empty); the O(1) check skips its O(pool)
      // ready-probe scan on the frequent spins where no IO has landed yet.
      Neighbor *node = id_buf_map.empty() ? nullptr : pool.next_calc_nbr();
      if (node != nullptr) {
        auto it = id_buf_map.find(node->id);
        auto &[id, dnode] = *it;
        if (compute_exact_dists_and_push(dnode, id)) {
          node->is_member = true;
        }
        compute_and_push_nbrs(dnode);
        id_buf_map.erase(it);
      }
      return (int) pool.frontier();
    };

    // --- Termination: N valid members OR all visited OR range boundary crossed ---
    // Range early-stop is driven by the approximate retset distance. We expand
    // the exact threshold with a fixed slack factor so only sufficiently large
    // approximate distances trigger range_crossed. Defaults to +inf
    // (disabled) so non-range callers keep original behavior.
    constexpr float kRangeEarlyStopFactor = 2.0f;
    const float approx_range_partial = range_partial * kRangeEarlyStopFactor;
    auto terminate = [&]() -> bool {
      if (pool.first_unvisited() >= pool.size())
        return true;
      // Exact-members fast path: VerifyFn == AlwaysTrue means every settled
      // candidate is a member, so the settled-prefix member count IS the prefix
      // length (first_unvisited, kept current by frontier()). Skips the O(l_search)
      // prefix scan below, which otherwise runs once per main-loop spin. Only
      // bypasses the scan when range early-stop is off (the scan also watches
      // for the range crossing).
      if constexpr (std::is_same_v<VerifyFn, AlwaysTrue>) {
        if (approx_range_partial == std::numeric_limits<float>::infinity()) {
          return pool.first_unvisited() >= l_search;
        }
      }
      // Bounded to the contiguous visited (settled) prefix: count is_member, and
      // break early if any entry's distance crosses approx_range_partial. Distance
      // within the prefix is not monotone (sort key is (is_member_approx desc,
      // distance asc), so distance can drop at the member/non-member boundary), so
      // we check each entry. scan walks [0, size()); stopping at the first
      // !visited confines it to exactly the settled prefix.
      uint64_t is_member_cnt = 0;
      bool crossed = false;
      pool.scan([&](const Neighbor &n) {
        if (!n.visited)
          return true;
        is_member_cnt += n.is_member;
        if (n.distance > approx_range_partial) {
          crossed = true;
          return true;
        }
        // Enough settled members: the verdict is already "terminate", stop
        // rescanning the rest of the prefix (it only grows with l_search).
        return is_member_cnt >= l_search;
      });
      return crossed || is_member_cnt >= l_search;
    };

    // --- Main search loop ---
    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());

    if (mem_L) {  // Overlap the PQ query-table build with the first read.
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      pool.rescore_seeds(dist_scratch);
    }
    unsigned marker = 0;

    int cur_n_in = 0, cur_tot = 0;
    while (!terminate()) {
      auto [n_in, n_out] = poll_all();
      std::ignore = n_in;
      std::ignore = n_out;

      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        constexpr double kWasteThreshold = 0.1;
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }

      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
        send_best_read_req(1);
      }
      marker = calc_best_node();
      max_marker = std::max(max_marker, marker);
    }

    // Drain remaining in-flight IOs.
    while (!on_flight_ios.empty()) {
      poll_all();
    }

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
    }

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

}  // namespace pipeann
