#pragma once

#include <cassert>
#include <random>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "ssd_index_defs.h"
#include "utils/percentile_stats.h"
#include "utils/tsl/robin_set.h"
#include "utils/lock_table.h"

#include "distance.h"
#include "utils.h"
#include "utils/pipnn.h"

#define OVERHEAD_FACTOR 1.1
#define SLACK_FACTOR 1.3

namespace pipeann {
  inline double estimate_ram_usage(size_t size, size_t dim, size_t datasize, size_t degree) {
    double graph_size = (double) size * (double) degree * (double) sizeof(unsigned) * SLACK_FACTOR;
    size_t data_size = size * dim * datasize;
    return OVERHEAD_FACTOR * (graph_size + data_size);
  }

  template<typename T, typename TagT = uint32_t>
  class Index {
   public:
    // Constructor. The index will be dynamically resized.
    Index(Metric m, const size_t dim, uint64_t max_points = 0);
    Index(Metric m, const SSDIndexMetadata<T> &meta);

    ~Index();

    size_t get_num_points();

    // For cosine metric, if data is not pre-normalized, normalize_cosine should be set to true.
    // For SSD index, data is pre-normalized for correct PQ initialization, so normalize_cosine should be set to false.
    // R_ood = NGFix refine out-degree (total degree stays params.R = R_base + R_ood).
    // If R_ood > 0 and train_query_path is non-empty, run NGFix refine after the base Vamana build.
    void build(const char *filename, const size_t num_points_to_load, IndexBuildParameters &params,
               const std::vector<TagT> &tags, bool normalize_cosine = true, const std::string &train_query_path = "",
               uint32_t R_ood = 0, uint32_t L_ood = 1500);
    void build(const char *filename, const size_t num_points_to_load, IndexBuildParameters &params,
               const char *tag_filename = nullptr, bool normalize_cosine = true,
               const std::string &train_query_path = "", uint32_t R_ood = 0, uint32_t L_ood = 1500);

    // Added search overload that takes L as params, so that we
    // can customize L on a per-query basis without tampering with "IndexBuildParameters"
    std::pair<uint32_t, uint32_t> search(const T *query, const size_t K, const unsigned L, unsigned *indices,
                                         float *distances = nullptr, QueryStats *stats = nullptr,
                                         const std::vector<uint32_t> *cand_ids = nullptr);

    size_t search_with_tags(const T *query, const size_t K, const unsigned L, TagT *tags, float *distances,
                            const std::vector<uint32_t> *cand_ids = nullptr);

    // Public Functions for Incremental Support

    /* insertions possible only when id corresponding to tag does not already
     * exist in the graph */
    // only keep point, tag, params
    int insert_point(const T *point, const IndexBuildParameters &params, const TagT tag);

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK.
    int lazy_delete(const TagT &tag);

    // return immediately after "approx" converge.
    uint32_t search_with_tags_fast(const T *normalized_query, const unsigned L, TagT *tags, float *dists);

    void consolidate(IndexBuildParameters &params);

    /*  Internals of the library */
    std::vector<std::vector<unsigned>> _final_graph;  // 1-hop neighbors.

    // determines navigating node of the graph by calculating medoid of data
    unsigned calculate_entry_point();

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(const T *node_coords, const unsigned Lindex,
                                                         const std::vector<unsigned> &init_ids,
                                                         std::vector<Neighbor> &expanded_nodes_info,
                                                         tsl::robin_set<unsigned> &expanded_nodes_ids,
                                                         std::vector<Neighbor> &best_L_nodes,
                                                         QueryStats *stats = nullptr);

    void get_expanded_nodes(const size_t node, const unsigned Lindex, std::vector<Neighbor> &expanded_nodes_info);

    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list, const IndexBuildParameters &params);

    void link(IndexBuildParameters &params);

    // Graph Build Algorithm of PiPNN.
    void pipnn_link(IndexBuildParameters &params);

    void final_prune(IndexBuildParameters &params);

    void ngfix_refine(const std::string &train_query_path, uint32_t L_ood);

    std::vector<std::vector<uint32_t>> pipnn_partition(const std::vector<uint32_t> &node_ids,
                                                       IndexBuildParameters &params);
    std::vector<WeightedEdge> pipnn_build_leaf_nodes(const std::vector<uint32_t> &nodes, int k);

    // Support for Incremental Indexing
    uint32_t reserve_location();

    // Support for resizing the index
    void resize(size_t new_max_points);

    size_t max_points() const {
      return _data.size() / _dim;
    }

    // Use SSDIndex::save_from_mem and load_to_mem for saving/loading an in-memory index.

    // Write _location_to_tag to {tags_file}. Removes an existing file if the mapping is empty.
    void save_tags(const std::string &tags_file) const;

    // Populate _location_to_tag / _tag_to_location from {tags_file}, or use equal mapping
    // (tag == loc) for the first _nd locations if the file does not exist.
    void load_tags(const std::string &tags_file);

    std::vector<T> _data;
    Distance<T> *_distance = nullptr;
    pipeann::Metric _dist_metric;

    size_t _dim;
    size_t _nd = 0;       // number of active points i.e. existing in the graph
    uint16_t range = 0;   // maximum 1-hop out-degree (R_total = R_base + R_ood).
    uint16_t R_base = 0;  // Base build out-degree (e.g., Vamana or PiPNN), set during build().
    uint16_t R_ood = 0;   // NGFix refine out-degree, set during build().
    unsigned _ep = 0;

    // flags for dynamic indexing
    std::unordered_map<TagT, unsigned> _tag_to_location;
    std::unordered_map<unsigned, TagT> _location_to_tag;

    tsl::robin_set<unsigned> _delete_set;
    tsl::robin_set<unsigned> _empty_slots;

    // NGFix refine build-only sidecars. Cleared after build().
    // _ngfix_count[u] = number of refine edges currently in _final_graph[u][R_base..R_base+R_ood).
    // _ngfix_ehs[u][i] = EH score for refine edge at position R_base+i.
    std::vector<uint16_t> _ngfix_count;
    std::vector<std::vector<uint16_t>> _ngfix_ehs;

    pipeann::LockTable *_locks = nullptr;

    std::shared_timed_mutex _tag_lock;     // Lock on _tag_to_location, _location_to_tag, _delete_set
    std::shared_timed_mutex _update_lock;  // coordinate save() and any change
                                           // being done to the graph.
    std::mutex _change_lock;               // Lock taken to synchronously modify _nd

    const float INDEX_GROWTH_FACTOR = 1.5f;

   private:
    // NGFix refine: query-driven OOD edge augmentation.
    // After base build (R_base edges saturated), appends up to R_ood refine edges per node
    // at positions [R_base, R). Sidecars _ngfix_count/_ngfix_ehs are build-only.
    void ngfix_calc_hardness(const unsigned *gt, size_t Nq, size_t S, std::vector<std::vector<uint16_t>> &H);
    void ngfix_pass(const T *query, const unsigned *gt, size_t Nq, size_t Kh);
    void rfix_pass(const T *query, const unsigned *gt, size_t Nq, uint32_t L_ood);
    void add_ngfix_neighbor(unsigned u, unsigned v, uint16_t eh);
  };
}  // namespace pipeann
