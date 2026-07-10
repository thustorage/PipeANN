#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <limits>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

#include "pipe_search_common.h"

namespace pipeann {
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::spec_filter_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                               Selector *base_selector, const Attributes &query_attrs, TagT *res_tags,
                                               float *res_dists, const uint64_t beam_width, QueryStats *stats,
                                               NodeOut *node_out) {
    // In-filter state lives on the selector itself, so each query needs its
    // own cloned selector tree.
    Selector *selector = base_selector->copy();

    // stats.
    if (stats != nullptr) {
      memset(stats, 0, sizeof(QueryStats));
    }

    struct Cost {
      uint64_t ssd_reads = std::numeric_limits<uint64_t>::max();
      uint64_t dist_cmp = std::numeric_limits<uint64_t>::max();

      double total_cost() const {
        constexpr double kSSDReadWeight = 10.0f, kDistCmpWeight = 1.0f;
        return ssd_reads * kSSDReadWeight + dist_cmp * kDistCmpWeight;
      }

      bool operator<(const Cost &other) const {
        return total_cost() < other.total_cost();
      }
    } filter_cost[N_FILTER_TYPES];

    // first, estimate selectivity and precision.
    double selectivity = std::max(selector->estimate_selectivity(query_attrs), std::numeric_limits<double>::epsilon());
    double precision_in = std::max(selector->estimate_precision(query_attrs), std::numeric_limits<double>::epsilon());
    uint32_t prefilter_reads = selector->estimate_prefilter_reads(query_attrs);
    // Pre-filter scans labels to get sN/p candidates, does sN/p PQ comparisons, and reranks L/p vectors.
    uint64_t prefilter_candidates = (uint64_t) (selectivity * meta_.npoints);
    filter_cost[PRE_FILTER] = {.ssd_reads = (uint64_t) (prefilter_reads + l_search * io_size / SECTOR_LEN),
                               .dist_cmp = prefilter_candidates};

    // In-filter strategy leverages the denser graph (range_dense > range) to mitigate low selectivity.
    // By having more candidate edges, the effective probability of finding a valid neighbor increases.
    uint32_t infilter_reads = selector->estimate_infilter_reads(query_attrs);
    // is_member_approx overhead: γ * L/s * R (configurable γ, default 0.05).
    constexpr double kGamma = 0.05;

    double sRd_over_p = selectivity * meta_.range_dense / precision_in;
    double in_filter_l_eq = l_search;

    if (sRd_over_p <= meta_.range) {
      // Low selectivity: false positives are extra edges for connectivity, overhead ignored.
      // Equivalent candidate pool: L/s * R/R_d.
      in_filter_l_eq = l_search / selectivity * meta_.range / meta_.range_dense;
    } else {
      // High selectivity: false positives introduce non-matching neighbors.
      // Candidate pool should contain L/p vectors in total.
      in_filter_l_eq = l_search / precision_in;
    }

    filter_cost[IN_FILTER] = {.ssd_reads = (uint64_t) (infilter_reads + in_filter_l_eq * io_size_dense / SECTOR_LEN),
                              .dist_cmp = (uint64_t) (in_filter_l_eq * (meta_.range + kGamma * meta_.range_dense))};

    // Post-filter does not involve precision: L/s SSD reads, L/s * R distance computations.
    filter_cost[POST_FILTER] = {.ssd_reads = (uint64_t) (l_search / selectivity * io_size / SECTOR_LEN),
                                .dist_cmp = (uint64_t) (meta_.range * l_search / selectivity)};

    pipeann::FilterType min_filter_type = PRE_FILTER;
    for (int i = 0; i < N_FILTER_TYPES; i++) {
      if (filter_cost[i] < filter_cost[min_filter_type]) {
        min_filter_type = static_cast<pipeann::FilterType>(i);
      }
    }

    if (stats != nullptr) {
      stats->n_filter[min_filter_type]++;
      stats->n_est_filter_reads[min_filter_type] += filter_cost[min_filter_type].ssd_reads;
      stats->n_est_filter_cmps[min_filter_type] += filter_cost[min_filter_type].dist_cmp;
    }
    size_t ret = 0;
    switch (min_filter_type) {
      case PRE_FILTER:
        ret = spec_prefilter_search(query, k_search, l_search, selector, query_attrs, res_tags, res_dists, beam_width,
                                    stats, node_out);
        break;
      case IN_FILTER:
        ret = spec_infilter_search(query, k_search, l_search, in_filter_l_eq, selector, query_attrs, res_tags,
                                   res_dists, beam_width, stats, node_out);
        break;
      case POST_FILTER:
        ret = spec_postfilter_search(query, k_search, l_search, l_search / selectivity, selector, query_attrs, res_tags,
                                     res_dists, beam_width, stats, node_out);
        break;
      default:
        break;
    }
    delete selector;

    // The sub-paths tag filter non-members with distance FLT_MAX during exact
    // is_member rerank, and copy_top_k emits results in ascending distance order,
    // so any sentinels sit at the tail. Trim ret to the first non-member to drop
    // the filter-violating padding when fewer than k vectors match.
    while (ret > 0 && res_dists[ret - 1] >= std::numeric_limits<float>::max()) {
      ret--;
    }

    if (stats != nullptr) {
      // for preparation, the estimated read is accurate.
      if (min_filter_type == PRE_FILTER) {
        stats->n_filter_reads[PRE_FILTER] += prefilter_reads;
      } else if (min_filter_type == IN_FILTER) {
        stats->n_filter_reads[IN_FILTER] += infilter_reads;
      }
      // For regular I/O.
      stats->n_filter_reads[min_filter_type] += stats->n_ios;
    }

    return ret;
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::spec_prefilter_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                  Selector *selector, const Attributes &query_attrs, TagT *res_tags,
                                                  float *res_dists, const uint64_t beam_width, QueryStats *stats,
                                                  NodeOut *node_out) {
    QueryBuffer *query_buf = pop_query_buf(query1);
    auto ctx = reader->get_ctx();
    const T *query = query_buf->aligned_query<T>();

    Timer tot_timer, prefilter_timer, compute_timer, rerank_timer;
    tot_timer.reset();
    prefilter_timer.reset();
    // design space occurs here: low selectivity + high selectivity, then verify is better than directly filter?
    // therefore, prefilter should be speculative.
    VectorIDList vector_id_set = selector->pre_filter(query_attrs, reader.get());
    if (stats != nullptr) {
      stats->filter_io_us[PRE_FILTER] += prefilter_timer.elapsed();
    }

    compute_timer.reset();
    nbr_handler->initialize_query(query, query_buf);
    std::priority_queue<Neighbor> pq;
    // compare PQ distance with all the vectors in vector_id_set. MAX_N_EDGES at a time.
    std::vector<uint32_t> vector_ids(vector_id_set.begin(), vector_id_set.end());
    for (size_t i = 0; i < vector_ids.size(); i += MAX_N_EDGES) {
      uint32_t n_ids = std::min(MAX_N_EDGES, vector_ids.size() - i);
      nbr_handler->compute_dists(query_buf, vector_ids.data() + i, n_ids);
      for (uint32_t j = 0; j < n_ids; ++j) {
        if (pq.size() >= l_search && query_buf->aligned_dist_scratch[j] < pq.top().distance) {
          pq.pop();
        }
        if (pq.size() < l_search) {
          pq.push(Neighbor(vector_ids[i + j], query_buf->aligned_dist_scratch[j]));
        }
      }
    }
    // rerank with exact distance.
    uint64_t pq_sz = pq.size();
    std::vector<Neighbor> retset(pq_sz);
    for (size_t i = 0; i < pq_sz; ++i) {
      retset[pq_sz - i - 1] = pq.top();
      pq.pop();
    }
    if (stats != nullptr) {
      stats->filter_cpu_us[PRE_FILTER] += compute_timer.elapsed();
    }

    // TODO: no concurrency control now.
    rerank_timer.reset();

    // Double-buffer pipelined read.
    constexpr uint64_t N_READS = MAX_N_SECTOR_READS / 2;
    std::vector<IORequest> reqs[2];

    auto send_read_cur = [&](size_t i) {
      uint64_t idx = (i / N_READS) % 2;
      std::vector<IORequest> &cur_reqs = reqs[idx];
      size_t n = std::min(N_READS, retset.size() - i);
      for (size_t j = 0; j < n; ++j) {
        const unsigned loc = id2loc(retset[i + j].id), pid = loc_sector_no(loc);
        cur_reqs.push_back(IORequest(pid * SECTOR_LEN, io_size,
                                     query_buf->sector_scratch + (idx * N_READS + j) * io_size, u_loc_offset(loc),
                                     meta_.max_node_len, query_buf->sector_scratch));
      }
      size_t n_sent = reader->send_read(cur_reqs, ctx);
      if (stats != nullptr) {
        stats->n_ios += (double) io_size / SECTOR_LEN * n_sent;
      }
    };

    auto poll_and_compute_prev = [&](size_t i) {
      // compute the previous batch.
      uint64_t idx = (i / N_READS + 1) % 2;
      std::vector<IORequest> &prev_reqs = reqs[idx];
      for (size_t j = 0; j < prev_reqs.size(); ++j) {
        while (!prev_reqs[j].finished) {
          reader->poll(ctx);
        }
        auto node = node_from_page((char *) prev_reqs[j].buf, id2loc(retset[i - N_READS + j].id));
        // Verify membership using the labels read from SSD (handles false positives from speculative pre-filter).
        auto target_attrs = Attributes::deserialize((char *) node.attrs);
        bool is_member =
            (selector == nullptr || selector->is_member(retset[i - N_READS + j].id, query_attrs, target_attrs));
        if (stats != nullptr) {
          stats->n_filter_accessed_vectors[PRE_FILTER]++;
          stats->n_filter_false_positives[PRE_FILTER] += !is_member;
        }
        retset[i - N_READS + j].distance = is_member ? dist_cmp->compare(query, node.coords, (unsigned) aligned_dim)
                                                     : std::numeric_limits<float>::max();
        if (is_member && node_out != nullptr) {
          NodePayload payload;
          payload.coords.assign(node.coords, node.coords + meta_.data_dim);
          if (meta_.attr_size > 0) payload.attrs = std::move(target_attrs);
          node_out->insert(std::make_pair(id2tag(retset[i - N_READS + j].id), std::move(payload)));
        }
      }
      prev_reqs.clear();
    };

    size_t i = 0;
    for (i = 0; i < retset.size(); i += N_READS) {
      send_read_cur(i);
      poll_and_compute_prev(i);
    }
    poll_and_compute_prev(i);
    if (stats != nullptr) {
      stats->filter_io_us1[PRE_FILTER] += rerank_timer.elapsed();
    }

    std::sort(retset.begin(), retset.end(), [](const Neighbor &left, const Neighbor &right) { return left < right; });
    auto t = copy_top_k(retset, k_search, res_tags, res_dists);
    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us += tot_timer.elapsed();
    }
    return t;
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::spec_postfilter_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                   const uint64_t l_max, Selector *selector,
                                                   const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                                                   const uint64_t beam_width, QueryStats *stats, NodeOut *node_out) {
    auto always_member = [](unsigned) -> bool { return true; };
    auto no_prefetch = [](unsigned) {};
    auto verify = [&](unsigned id, const DiskNode<T> &node) -> bool {
      auto target_attrs = Attributes::deserialize((char *) node.attrs);
      bool is_member = (selector == nullptr || selector->is_member(id, query_attrs, target_attrs));
      if (stats != nullptr) {
        stats->n_filter_accessed_vectors[POST_FILTER]++;
        stats->n_filter_false_positives[POST_FILTER] += !is_member;
      }
      return is_member;
    };

    std::vector<Neighbor> full_retset;
    this->pipe_search_common(query1, l_search, 0, l_search, l_max, beam_width, false, always_member, verify,
                             no_prefetch, full_retset, stats, nullptr, std::numeric_limits<float>::infinity(),
                             node_out);

    return copy_top_k(full_retset, k_search, res_tags, res_dists);
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::spec_infilter_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 const uint64_t l_max, Selector *selector,
                                                 const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                                                 const uint64_t beam_width, QueryStats *stats, NodeOut *node_out) {
    // Prepare in-filter state (e.g., scan cold attributes).
    Timer prefilter_timer;
    prefilter_timer.reset();
    selector->prepare_in_filter(query_attrs, reader.get());
    if (stats != nullptr) {
      stats->filter_io_us[IN_FILTER] += prefilter_timer.elapsed();
    }

    auto verify = [&](unsigned id, const DiskNode<T> &node) -> bool {
      auto target_attrs = Attributes::deserialize((char *) node.attrs);
      bool is_member = (selector == nullptr || selector->is_member(id, query_attrs, target_attrs));
      if (stats != nullptr) {
        stats->n_filter_accessed_vectors[IN_FILTER]++;
        stats->n_filter_false_positives[IN_FILTER] += !is_member;
      }
      return is_member;
    };

    auto is_member_approx = [&](unsigned id) -> bool { return selector->is_member_approx(id, query_attrs); };
    // Prefetch the per-vector membership data a few neighbors ahead of the check;
    // hides the random-access miss that dominates in-filter CPU (~27% latency cut,
    // measured on SIFT1B b9 @1t, iso-recall/iso-IO).
    auto prefetch_approx = [&](unsigned id) { selector->prefetch_approx(id, query_attrs); };

    std::vector<Neighbor> full_retset;
    // Pin the attr indexes' shared locks for the whole traversal so the
    // per-neighbor is_member_approx calls above run lock-free.
    selector->lock_shared();
    this->pipe_search_common(query1, l_search, 0, l_search, l_max, beam_width, true, is_member_approx, verify,
                             prefetch_approx, full_retset, stats, nullptr, std::numeric_limits<float>::infinity(),
                             node_out);
    selector->unlock_shared();

    return copy_top_k(full_retset, k_search, res_tags, res_dists);
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
