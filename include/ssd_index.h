#pragma once
#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <set>
#include <omp.h>

#include <memory>

#include "aligned_file_reader.h"
#include "ssd_index_defs.h"
#include "filter/attribute.h"
#include "filter/selector.h"
#include "utils/arch_compat.h"
#include "utils/concurrent_queue.h"
#include "utils/lock_table.h"
#include "utils/percentile_stats.h"
#include "utils/tsl/robin_map.h"
#include "nbr/abstract_nbr.h"
#include "utils.h"
#include "index.h"

enum SearchMode { BEAM_SEARCH = 0, PAGE_SEARCH = 1, PIPE_SEARCH = 2, CORO_SEARCH = 3 };

namespace pipeann {
  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;
    static constexpr uint32_t kExpandedNodesFactor = 10;

    std::unique_ptr<Index<T, uint32_t>> mem_index_;  // in-memory navigation graph

    // Index metadata (consolidated).
    SSDIndexMetadata<T> meta_;

    // For updates:
    // meta.npoints is the index's initial size (constant between two merges).
    // cur_id is the ID to be allocated (+1 for each insert, starting from meta.npoints)
    // cur_loc is the tail of the index file, which will be greater than cur_id if overprovisioned.
    // - This is because holes may exist in the index for update combining.
    std::atomic<uint64_t> cur_id, cur_loc;

    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader, AbstractNeighbor<T> *nbr = nullptr,
             bool tags = false, IndexBuildParameters *parameters = nullptr, bool enable_tag2id = false);

    ~SSDIndex();

    // Use node_from_page() to create instances for DiskNode.
    // Params: location (id2loc first if using ID), the page-aligned buffer.
    // The offset is calculated in the constructor.
    inline DiskNode<T> node_from_page(char *page_buf, uint32_t loc) {
      return DiskNode<T>(page_buf, loc, meta_);
    }

    // Unaligned offset to location.
    inline uint64_t u_loc_offset(uint64_t loc) const {
      return meta_.u_loc_offset(loc);
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) const {
      return meta_.u_loc_offset_nbr(loc);
    }

    inline uint64_t loc_sector_no(uint64_t loc) const {
      return meta_.loc_sector_no(loc);
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) const {
      return meta_.sector_to_loc(sector_no, sector_off);
    }

    void init_metadata(const SSDIndexMetadata<T> &meta) {
      meta.print();
      this->meta_ = meta;
      this->cur_id = this->cur_loc = meta.npoints;
      this->aligned_dim = ROUND_UP(meta_.data_dim, 8);
      this->params.R = meta.range;
      this->io_size = meta.io_size();
      this->io_size_dense = meta.io_size_dense();
      LOG(INFO) << "IO size (normal): " << io_size << " IO size (dense): " << io_size_dense;

      // Aligned.
      if (meta_.nnodes_per_sector != 0 && meta_.npoints % meta_.nnodes_per_sector != 0) {
        cur_loc += meta_.nnodes_per_sector - (meta_.npoints % meta_.nnodes_per_sector);
      }
      LOG(INFO) << "Cur location: " << this->cur_loc;

      // Update-related metadata, if not initialized in constructor, initialize here.
      if (params.L == 0) {
        // Experience values.
        LOG(INFO) << "Automatically set the update-related parameters.";
        params.set(meta.range, meta.range + 32, 384, 1.2, 0, true, 8);
        params.print();
      }
    }

    void init_query_buf(QueryBuffer &buf) {
      pipeann::alloc_aligned((void **) &buf.coord_scratch_, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      buf.sector_scratch = (char *) this->reader->alloc_io_buf(MAX_N_SECTOR_READS * io_size_dense, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.nbr_vec_scratch,
                             MAX_N_EDGES * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(uint8_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_id_scratch, MAX_N_EDGES * sizeof(uint32_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_ctx_scratch, ROUND_UP(nbr_handler->query_ctx_size(), 256), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, MAX_N_EDGES * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_, this->aligned_dim * sizeof(T), 8 * sizeof(T));

      buf.visited = new tsl::robin_set<uint64_t>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * io_size_dense);
      memset(buf.coord_scratch_, 0, this->aligned_dim * sizeof(T));
      memset(buf.aligned_query_, 0, this->aligned_dim * sizeof(T));
    }

    QueryBuffer *pop_query_buf(const T *query) {
      QueryBuffer *data = this->thread_data_queue.pop();
      while (data == this->thread_data_queue.null_T) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (this->metric == Metric::COSINE) {
          // Data has been normalized. Normalize search vector too.
          pipeann::normalize_data(data->aligned_query<T>(), query, meta_.data_dim);
        } else {
          memcpy(data->aligned_query<T>(), query, meta_.data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // Load compressed data, and obtains the handle to the disk-resident index.
    // When force_recopy is true, SPDK is forced to copy the disk index regardless of marker.
    // For other io engines, force_recopy does nothing.
    int load(const char *index_prefix, bool force_recopy = false);

    // Load the companion nav graph at {mem_index_path} (SSD-format disk file) into mem_index_.
    void load_mem_index(const std::string &mem_index_path);

    // Write an in-memory Index to an SSD-format disk file at {disk_file}, plus {disk_file}.tags.
    // If target_dense_width > mem.range, sample (target_dense_width - mem.range) 2-hop
    // neighbors per node on the fly and store them as dense edges; otherwise only
    // 1-hop edges are written.
    static void save_from_mem(Index<T, TagT> &mem, const std::string &disk_file, uint16_t target_dense_width = 0,
                              AttrWriter *attr_writer = nullptr);

    // Read an SSD-format disk file at {disk_file} (+ {disk_file}.tags if present) into an Index.
    // Caller owns the returned pointer.
    static Index<T, TagT> *load_to_mem(const std::string &disk_file, Metric metric);

    // Read-only search algorithms (static datasets only).
    size_t coro_search(T **queries, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT **res_tags, float **res_dists, const uint64_t beam_width, int N);

    // Optional sink for returning matched nodes' payload (vector coords + decoded
    // attrs) from a search, keyed by tag. Used by the Milvus layer to serve
    // output_fields without a second disk read. nullptr (the default) => no
    // payload is collected and the search path is unchanged.
    struct NodePayload {
      std::vector<T> coords;  // data_dim floats (the raw vector)
      Attributes attrs;       // node's on-disk filterable scalar attributes
    };
    using NodeOut = tsl::robin_map<TagT, NodePayload>;

    size_t page_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    // Search supporting updates (with concurency control).
    size_t beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       NodeOut *node_out = nullptr);

    // Range search: return every tag whose distance to query is <= range (in user-facing metric).
    // Early stop when all vectors with PQ_distance > kRangeEarlyStopFactor * range are explored, may miss some results.
    // Caller pre-allocates res_tags/res_dists to at least l_search entries.
    size_t range_search(const T *query, const float range, TagT *res_tags, float *res_dists, const uint64_t beam_width,
                        const uint32_t mem_L = 0, const uint64_t l_search = 8192, QueryStats *stats = nullptr);

    size_t spec_filter_search(const T *query, const uint64_t k_search, const uint64_t l_search, Selector *selector,
                              const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                              const uint64_t beam_width, QueryStats *stats = nullptr, NodeOut *node_out = nullptr);

    size_t spec_prefilter_search(const T *query, const uint64_t k_search, const uint64_t l_search, Selector *selector,
                                 const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                                 const uint64_t beam_width, QueryStats *stats = nullptr, NodeOut *node_out = nullptr);

    size_t spec_postfilter_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                  const uint64_t l_max, Selector *selector, const Attributes &query_attrs,
                                  TagT *res_tags, float *res_dists, const uint64_t beam_width,
                                  QueryStats *stats = nullptr, NodeOut *node_out = nullptr);

    size_t spec_infilter_search(const T *query, const uint64_t k_search, const uint64_t l_search, const uint64_t l_max,
                                Selector *selector, const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                                const uint64_t beam_width, QueryStats *stats = nullptr, NodeOut *node_out = nullptr);

    int insert_in_place(const T *point, const TagT &tag, const Attributes *attrs = nullptr);

    // Merge deletes (NOTE: index read-only during merge.)
    // Returns id_map: old_id -> new_id.
    libcuckoo::cuckoohash_map<uint32_t, uint32_t> merge_deletes(const std::string &in_path_prefix,
                                                                const std::string &out_path_prefix,
                                                                const std::vector<TagT> &deleted_nodes,
                                                                const tsl::robin_set<TagT> &deleted_nodes_set,
                                                                uint32_t nthreads);

    // After merge, reload the index.
    void reload(const char *index_prefix);

    void write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                               std::vector<TagT> *new_tags = nullptr);

    void copy_index(const std::string &prefix_in, const std::string &prefix_out);

    TagT id2tag(uint32_t id);

    // Batch-read several nodes' payloads (vector coords + on-disk filterable
    // attrs) by internal id, using the AlignedFileReader. Used by the Milvus
    // pure-scalar Query path, which has internal ids (from pre_filter) before
    // the id2tag translation. Results are inserted into *out keyed by tag.
    void get_nodes_by_ids(const std::vector<uint32_t> &ids, NodeOut *out);

    // Batch-read node payloads addressed by tag instead of internal id. Resolves
    // each wanted tag back to an internal id via the id->tag map, then delegates
    // to get_nodes_by_ids. Used by the Milvus primary-key Query/Get path, which
    // resolves the pk to a tag through the doc-store but needs the on-disk attrs.
    void get_nodes_by_tags(const std::vector<TagT> &tags_wanted, NodeOut *out);

    // Held exclusive during the final metadata swap in merge_deletes + reload.
    // Acquired in merge_deletes; released in reload. Callers that invoke reload
    // again (e.g. after copy_index in the double-version path) must re-acquire.
    std::shared_mutex merge_lock;

   private:
    // Background insert I/O commit.
    struct BgTask {
      QueryBuffer *thread_data;
      std::vector<IORequest> writes;
      std::vector<uint64_t> pages_to_unlock;
      std::vector<uint64_t> pages_to_deref;
      bool terminate = false;
    };

    using PageArr = std::vector<uint32_t>;

    static constexpr int kBgIOThreads = 1;

    // Derived/runtime metadata not stored in SSDIndexMetadata.
    uint64_t aligned_dim = 0;
    uint64_t io_size = 0;
    uint64_t io_size_dense = 0;

    // File reader and index file path.
    std::shared_ptr<AlignedFileReader> &reader;
    std::string disk_index_file;

    // Neighbor handler.
    AbstractNeighbor<T> *nbr_handler;

    // Distance comparator.
    pipeann::Metric metric;
    std::shared_ptr<Distance<T>> dist_cmp;

    // Update-related parameters.
    IndexBuildParameters params;

    // Thread-specific scratch buffers.
    ConcurrentQueue<QueryBuffer *> thread_data_queue;
    std::vector<QueryBuffer *> thread_data_bufs;  // pre-allocated thread data

    // Background I/O threads for insert.
    ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
    std::thread *bg_io_thread_[kBgIOThreads]{nullptr};

    // Locking tables for concurrency control.
    pipeann::SparseLockTable<uint64_t> page_lock_table;
    pipeann::SparseLockTable<uint64_t> vec_lock_table;
    pipeann::SparseLockTable<uint64_t> idx_lock_table;

    // ID to location mapping.
    // Concurrency control is done in lock_idx.
    // Only resize should be protected.
    std::vector<uint32_t> id2loc_;
    pipeann::ReaderOptSharedMutex id2loc_resize_mu_;

    // Location to ID mapping.
    // If nnodes_per_sector >= 1, page_layout[i * nnodes_per_sector + j] is the id of the j-th node in the i-th page.
    // ElseIf nnodes_per_sector == 0, page_layout[i] is the id of the i-th node (starting from loc_sector_no(i)).
    std::vector<uint32_t> loc2id_;
    pipeann::ReaderOptSharedMutex loc2id_resize_mu_;
    std::mutex alloc_lock;
    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);

    // Tag support.
    // If ID == tag, then it is not stored.
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;

    // Optional reverse map: tag -> id, kept in sync with `tags` on every
    // mutation. Not persisted (rebuilt from `tags` on load). Enables an
    // O(|wanted|) get_nodes_by_tags instead of a full scan of `tags`; when
    // enable_tag2id is false, get_nodes_by_tags is unsupported and errors out.
    // IDs are never reused, so latest-wins is correct. Note: populated only when
    // a tags file is present (non-equal mapping); equal mapping leaves it empty.
    bool enable_tag2id = false;
    libcuckoo::cuckoohash_map<TagT, uint32_t> tag2id_;

    // Mirror a single id->tag mapping into tag2id_ (no-op unless enabled).
    inline void track_tag2id(uint32_t id, const TagT &tag) {
      if (enable_tag2id)
        tag2id_.insert_or_assign(tag, id);
    }

    // Flags.
    bool load_flag = false;    // already loaded.
    bool enable_tags = false;  // support for tags and dynamic indexing

    void init_buffers(uint64_t nthreads);
    void destroy_buffers();

    // Load id2loc and loc2id (i.e., page_layout), to support index reordering.
    void load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector = 0,
                          const uint64_t num_points = 0);

    void load_tags(const std::string &tag_file, size_t offset = 0);

    // Direct insert related.
    struct InsertContext {
      tsl::robin_map<uint32_t, T *> coord_map;
      T *coord_buf;
      std::vector<uint64_t> hint_pages;
      std::vector<uint64_t> page_ref;

      InsertContext(uint64_t max_nodes, uint64_t aligned_dim) {
        coord_map.reserve(max_nodes);
        pipeann::alloc_aligned((void **) &coord_buf, max_nodes * aligned_dim * sizeof(T), 256);
      }

      ~InsertContext() {
        pipeann::aligned_free(coord_buf);
      }

      InsertContext(const InsertContext &) = delete;
      InsertContext &operator=(const InsertContext &) = delete;
      InsertContext(InsertContext &&) = delete;
      InsertContext &operator=(InsertContext &&) = delete;
    };

    void do_beam_search(const T *query, uint32_t mem_L, uint64_t l_search, const uint64_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, QueryStats *stats = nullptr,
                        InsertContext *insert_ctx = nullptr);

    void do_pipe_search(const T *query, uint64_t k_search, uint32_t mem_L, uint64_t l_search, const uint64_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, QueryStats *stats = nullptr,
                        InsertContext *insert_ctx = nullptr, NodeOut *node_out = nullptr);

    template<typename SpecFn, typename VerifyFn>
    void pipe_search_common(const T *query, uint64_t k_search, uint32_t mem_L, uint64_t l_search, uint64_t l_pool,
                            uint64_t beam_width, bool use_dense_nbrs, SpecFn is_member_approx, VerifyFn is_member,
                            std::vector<Neighbor> &full_retset, QueryStats *stats = nullptr,
                            InsertContext *insert_ctx = nullptr,
                            float range_partial = std::numeric_limits<float>::infinity(), NodeOut *node_out = nullptr);

    // Background I/O thread function.
    void bg_io_thread();

    int get_vector_by_id(const uint32_t &id, T *vector);

    // Deduplicate sorted results and copy top-k to output arrays.
    size_t copy_top_k(const std::vector<Neighbor> &sorted_results, uint64_t k, TagT *res_tags, float *res_dists,
                      float max_dist = std::numeric_limits<float>::infinity());

    uint32_t id2loc(uint32_t id);
    void set_id2loc(uint32_t id, uint32_t loc);
    uint64_t id2page(uint32_t id);

    uint32_t loc2id(uint32_t loc);
    void set_loc2id(uint32_t loc, uint32_t id);
    void erase_loc2id(uint32_t loc);

    PageArr get_page_layout(uint32_t page_no);

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids);
    // Returns <loc, need_read>.
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read);
    void verify_id2loc();

    // lock-related.
    void lock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                  const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                    const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock);
  };
}  // namespace pipeann
