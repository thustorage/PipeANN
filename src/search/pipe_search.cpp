#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "pipe_search_common.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_pipe_search(const T *query, uint64_t k_search, uint32_t mem_L, uint64_t l_search,
                                         const uint64_t beam_width, std::vector<Neighbor> &expanded_nodes_info,
                                         QueryStats *stats, InsertContext *insert_ctx, NodeOut *node_out) {
    auto always_member = [](unsigned) -> bool { return true; };
    auto no_prefetch = [](unsigned) {};
    // VerifyFn = AlwaysTrue is a type-level signal: pipe_search_common detects
    // it (is_same_v) and switches terminate() to the O(1) member count.
    this->pipe_search_common(query, k_search, mem_L, l_search, l_search, beam_width, false, always_member,
                             AlwaysTrue(), no_prefetch, expanded_nodes_info, stats, insert_ctx,
                             std::numeric_limits<float>::infinity(), node_out);
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats, NodeOut *node_out) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_pipe_search(query, k_search, mem_L, l_search, beam_width, expanded_nodes_info, stats, nullptr, node_out);
    return copy_top_k(expanded_nodes_info, k_search, res_tags, distances);
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::range_search(const T *query, const float range, TagT *res_tags, float *res_dists,
                                         const uint64_t beam_width, const uint32_t mem_L, const uint64_t l_search,
                                         QueryStats *stats) {
    std::shared_lock lk(merge_lock);
    auto always_member = [](unsigned) -> bool { return true; };
    auto no_prefetch = [](unsigned) {};

    const float range_partial = get_partial_order_distance<T>(range, this->metric);
    std::vector<Neighbor> full_retset;
    this->pipe_search_common(query, l_search, mem_L, l_search, l_search, beam_width, false, always_member,
                             AlwaysTrue(), no_prefetch, full_retset, stats, nullptr, range_partial);
    return copy_top_k(full_retset, l_search, res_tags, res_dists, range_partial);
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
