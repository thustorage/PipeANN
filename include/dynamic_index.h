#pragma once

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include "index.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "utils/partition.h"
#include "utils/index_build_utils.h"
#include "ssd_index.h"
#include "utils.h"
#include "filter/dsl_compiler.h"
#include <sys/sysinfo.h>

#ifdef USE_TCMALLOC
#include <gperftools/malloc_extension.h>
#endif

class BaseDynamicIndex {
 public:
  BaseDynamicIndex() = default;
  virtual ~BaseDynamicIndex() = default;
};

inline std::shared_ptr<pipeann::AttrIndex> load_attr_index_from_file(const std::string &filename,
                                                                     const std::string &attr_type, uint64_t n_vectors) {
  return std::shared_ptr<pipeann::AttrIndex>(pipeann::load_attr_index_from_file(filename, attr_type, n_vectors));
}

template<class T>
class DynamicIndex : public BaseDynamicIndex {
  static constexpr float kMemIndexP = 0.01;
  static constexpr uint32_t kBuildThreshold = 100000;
  using TagT = uint32_t;

 public:
  using data_type = T;
  // Per-tag node payload sink (vector coords + filterable attrs), used by the
  // Milvus layer to serve output_fields without an extra disk read.
  using NodeOut = typename pipeann::SSDIndex<T, TagT>::NodeOut;

 public:
  DynamicIndex() = delete;

  explicit DynamicIndex(uint32_t data_dim, pipeann::Metric metric, pipeann::IndexBuildParameters *params = nullptr,
                        uint32_t attr_size = 0, bool enable_tag2id = false)
      : data_dim_(data_dim), metric_(metric), attr_size_(attr_size) {
    if (params != nullptr) {
      params_ = *params;
    }
    reader_.reset(new LinuxAlignedFileReader());
    nbr_handler_ = new pipeann::PQNeighbor<T>(metric_);
    mem_index_.reset(new pipeann::Index<T, TagT>(metric_, data_dim_));
    disk_index_.reset(new pipeann::SSDIndex<T, TagT>(metric_, reader_, nbr_handler_, true, params, enable_tag2id));

    mem_index_params_.set(32, 64, 750, 1.2, params_.num_threads, true);
    cur_index_prefix_ = "./test";
    // rand value for updating in-memory index.
    std::random_device rd;
    gen = std::mt19937(rd());
  }

  float rand_uniform() {
    std::uniform_real_distribution<float> dist(0, 1);
    return dist(gen);
  }

  ~DynamicIndex() {
  }

  // Load an index from disk.
  // If copy_to_shadow is true, the disk index is first copied to a shadow prefix
  // to avoid polluting the original index during in-place inserts.
  void load(const std::string &index_prefix, bool copy_to_shadow = false) {
    auto mem_index_path = index_prefix + "_mem.index";
    if (file_exists(mem_index_path)) {
      use_mem_index_for_disk_index_ = true;
      mem_index_.reset(pipeann::SSDIndex<T, TagT>::load_to_mem(mem_index_path, metric_));
    } else {
      use_mem_index_for_disk_index_ = false;
    }

    auto disk_index_file = index_prefix + "_disk.index";
    if (file_exists(disk_index_file)) {
      use_disk_index_ = true;

      std::string load_prefix = index_prefix;
      if (copy_to_shadow) {
        std::string shadow_prefix = index_prefix + "_shadow";
        disk_index_->copy_index(index_prefix, shadow_prefix);
        LOG(INFO) << "Copy disk index file to " << shadow_prefix << "_disk.index";
        load_prefix = shadow_prefix;
      }

      disk_index_->load(load_prefix.c_str(), true); // force recopy for SPDK.
      if (use_mem_index_for_disk_index_) {
        // mem_index is disk_index's navigation graph.
        disk_index_->mem_index_.reset(mem_index_.get());
      }
      mem_L = use_mem_index_for_disk_index_ ? 10 : 0;
      data_dim_ = disk_index_->meta_.data_dim;
      cur_index_prefix_ = load_prefix;
    } else {
      use_disk_index_ = false;
      cur_index_prefix_ = index_prefix;
    }
  }

  void transform_mem_index_to_disk_index() {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);
    LOG(INFO) << "Transform memory index to disk index.";

    const std::string data_path = cur_index_prefix_ + "_mem_data.bin";
    pipeann::save_bin<T>(data_path, mem_index_->_data.data(), mem_index_->_nd, mem_index_->_dim);
    LOG(INFO) << "Memory index data saved to " << data_path;

    const std::string tags_path = cur_index_prefix_ + "_disk.index.tags";
    mem_index_->save_tags(tags_path);

    // re-build
    this->build(data_path, cur_index_prefix_, tags_path.c_str(), false);
    this->load(cur_index_prefix_, true);  // here sets use_disk_index_ to true.
    LOG(INFO) << "Transform memory index to disk index done.";
  }

  // Use file path to build.
  void build(const std::string &data_path, const std::string &index_prefix, const char *tag_file = nullptr,
             bool build_mem_index = false, uint32_t max_nbrs = 0, uint32_t build_L = 0, uint32_t PQ_bytes = 0,
             uint32_t memory_use_GB = 0, pipeann::AttrWriter *attr_writer = nullptr, uint32_t range_dense = 0,
             const std::string &train_query_path = "", uint32_t R_ood = 0, uint32_t L_ood = 1500) {
    // automatically configure max_nbrs.
    size_t nr = 0, nc = 0;  // nr is number of points, nc is dimension.
    pipeann::get_bin_metadata(data_path, nr, nc);
    int nr_log10 = std::min(9, (int) std::ceil(std::log10(nr)));

    if (max_nbrs == 0) {
      uint32_t nr_nbr_arr[] = {64, 64, 64, 64, 64, 64, /* 1M */ 64, /* 10M */ 64, /* 100M */ 96, /* 1B */ 128};
      max_nbrs = nr_nbr_arr[nr_log10];
      LOG(INFO) << "Dataset contains " << nr << " points. Setting max_nbrs to " << max_nbrs;
    }

    if (PQ_bytes == 0) {
      PQ_bytes = nc / 4;                    // experience value.
      PQ_bytes = std::max(PQ_bytes, 32u);   // at least 32 bytes.
      PQ_bytes = std::min(PQ_bytes, 128u);  // at most 128 bytes.
      LOG(INFO) << "Data dimension is " << nc << ". Setting PQ_bytes to " << PQ_bytes;
    }

    if (build_L == 0) {
      build_L = max_nbrs + 32;
    }

    if (memory_use_GB == 0) {
      struct sysinfo info;
      sysinfo(&info);
      memory_use_GB = info.totalram / (1024 * 1024 * 1024) * 3 / 4;
      LOG(INFO) << "Memory use not specified. Using 75% of total memory: " << memory_use_GB << "GB";
    }

    pipeann::ZeroAttrWriter zero_attr_writer(attr_size_);
    if (attr_writer == nullptr && attr_size_ > 0) {
      attr_writer = &zero_attr_writer;
    }

    // densify by filling one more page of dense neighbors.
    // Only for filtered vector search with >= 1M vectors.
    if (attr_writer != nullptr && nr >= 1000000 && range_dense == 0) {
      uint32_t estimated_node_size = nc * sizeof(T) + attr_writer->attr_size() + (1 + max_nbrs) * sizeof(uint32_t);
      uint32_t gap = ROUND_UP(estimated_node_size, SECTOR_LEN) - estimated_node_size;
      range_dense = gap / sizeof(uint32_t) / 100 * 100;  // make it a multiple of 100 :).
      if (range_dense <= 900) {                          // <= 3.5KB free space, use one more page.
        range_dense += 1000;
      }
    }

    if (train_query_path != "" && R_ood == 0) {
      R_ood = max_nbrs / 2;
      LOG(INFO) << "Automatically setting #neighbors for OOD to " << R_ood;
    }

    pipeann::build_disk_index<T, TagT>(data_path.c_str(), index_prefix.c_str(), max_nbrs, build_L, memory_use_GB,
                                       params_.num_threads, PQ_bytes, metric_, tag_file, nbr_handler_, attr_writer,
                                       range_dense, 0, train_query_path, R_ood, L_ood);

    if (build_mem_index) {
      build_mem(data_path, index_prefix);
    }
  }

  // Canonical on-disk filename for a field's AttrIndex (matches what
  // create_attr_index / build_from_buffer use so reload after build finds them).
  std::string attr_index_file(uint32_t key) const {
    return cur_index_prefix_ + ".label." + std::to_string(key);
  }

  // Bulk-build a disk index from an in-memory buffer of vectors + per-row attributes.
  // This is a SEPARATE path from add()/transform_mem_index_to_disk_index() (which
  // support faiss-style from-scratch incremental construction): the gRPC server
  // buffers all Insert data, then calls this at create_index time to do a single
  // offline build (the fast path) instead of slow per-point insert_in_place.
  //
  //   vectors    : flat row-major buffer of n * data_dim_ values
  //   tags       : one TagT per vector
  //   attrs_vec  : one Attributes per vector (nullptr if no scalar fields)
  //   field_types: key -> attr_type ("label" | "range" | "string"); keys index into Attributes
  //
  // After building, loads the index and wires up the per-field AttrIndexes so the
  // search path (selector pre_filter + get_attr_index) works immediately.
  // build_R / build_L override the graph degree and build search-list size.
  // 0 means keep the offline builder's automatic (dataset-size-based) defaults.
  void build_from_buffer(const T *vectors, const TagT *tags, uint64_t n,
                         const std::vector<pipeann::Attributes> *attrs_vec,
                         const std::map<uint32_t, std::string> &field_types,
                         uint32_t build_R = 0, uint32_t build_L = 0) {
    if (cur_index_prefix_.empty()) {
      throw std::runtime_error("build_from_buffer: index prefix is empty");
    }
    const std::string prefix = cur_index_prefix_;

    // 1. Write the buffered vectors to a temp data .bin for the offline builder.
    const std::string data_path = prefix + "_build_data.bin";
    pipeann::save_bin<T>(data_path, const_cast<T *>(vectors), n, data_dim_);

    // 2. Write the tag file (vector position -> tag).
    const std::string tags_path = prefix + "_disk.index.tags";
    pipeann::save_bin<TagT>(tags_path, const_cast<TagT *>(tags), n, 1);

    // 3. Build each field's AttrIndex from the buffered per-row attributes, save
    //    it, and collect into an AttrIndexWriter so attributes are embedded into
    //    the on-SSD records during the build (for post-filter exact verification).
    std::map<uint32_t, pipeann::AttrIndex *> build_attrs;
    std::vector<std::shared_ptr<pipeann::AttrIndex>> built_indexes;
    for (const auto &[key, type] : field_types) {
      std::vector<pipeann::Attribute> rows(n);
      if (attrs_vec != nullptr) {
        for (uint64_t i = 0; i < n; i++) {
          rows[i] = (*attrs_vec)[i].get(key);
        }
      }
      std::string attr_file = attr_index_file(key);
      auto ai = std::shared_ptr<pipeann::AttrIndex>(pipeann::create_empty_attr_index(attr_file, type, n));
      ai->load_from_rows(rows);
      ai->save();
      build_attrs[key] = ai.get();
      built_indexes.push_back(ai);
    }

    std::unique_ptr<pipeann::AttrIndexWriter> attr_writer;
    if (!build_attrs.empty()) {
      attr_writer = std::make_unique<pipeann::AttrIndexWriter>(build_attrs);
      attr_size_ = attr_writer->attr_size();
    }

    // 4. Offline disk build (fast path), embedding attributes into the records.
    this->build(data_path, prefix, tags_path.c_str(), false, build_R, build_L, 0, 0, attr_writer.get());

    // 5. Load the built index and re-wire the attr indexes for filtered search.
    this->load(prefix, false);
    native_attr_indexes_.clear();
    attr_index_map_.clear();
    for (const auto &[key, type] : field_types) {
      load_attr_index_from_file(key, attr_index_file(key), type);
    }

    std::remove(data_path.c_str());
  }

  void set_index_prefix(const std::string &index_prefix) {
    cur_index_prefix_ = index_prefix;
  }

  void omp_set_num_threads(uint32_t num_threads) {
    ::omp_set_num_threads(std::min(num_threads, std::thread::hardware_concurrency()));
  }

  size_t npoints() const {
    if (use_disk_index_) {
      return disk_index_->cur_id.load();
    }
    return mem_index_->get_num_points();
  }

  std::shared_ptr<pipeann::AttrIndex> load_attr_index_from_file(uint32_t key, const std::string &filename,
                                                                const std::string &attr_type) {
    uint64_t n_vectors = use_disk_index_ ? disk_index_->meta_.npoints : mem_index_->get_num_points();
    auto attr_index = ::load_attr_index_from_file(filename, attr_type, n_vectors);
    native_attr_indexes_.push_back(attr_index);
    attr_index_map_[key] = attr_index.get();
    return attr_index;
  }

  std::shared_ptr<pipeann::AttrIndex> create_attr_index(uint32_t key, const std::string &filename,
                                                       const std::string &attr_type) {
    uint64_t n_vectors = use_disk_index_ ? disk_index_->cur_id.load() : mem_index_->get_num_points();
    auto attr_index = std::shared_ptr<pipeann::AttrIndex>(
        pipeann::create_empty_attr_index(filename, attr_type, n_vectors));
    native_attr_indexes_.push_back(attr_index);
    attr_index_map_[key] = attr_index.get();
    return attr_index;
  }

  // Get a raw AttrIndex pointer by key (for building dsl::Schema externally).
  pipeann::AttrIndex *get_attr_index(uint32_t key) const {
    auto it = attr_index_map_.find(key);
    if (it == attr_index_map_.end()) {
      LOG(ERROR) << "get_attr_index: key " << key << " not found in attr_index_map_";
      return nullptr;
    }
    return it->second;
  }

  // Compile a SQL-like filter string into a CompiledFilter
  // (Selector + attrs_template + slot_map). The schema dict tells
  // the compiler which fields are valid and which AttrIndex backs each one.
  // Caller takes ownership of the returned Selector tree.
  // Use `$$varName` placeholders in the filter for batch/late-binding workflows;
  // when no placeholders are present, `attrs_template` is fully populated and
  // can be passed directly to search.
  pipeann::dsl::CompiledFilter compile_filter(const std::string &json,
                                              const pipeann::dsl::Schema &schema) {
    return pipeann::dsl::compile(json, schema);
  }

  // End-to-end loader for the unified filter config:
  //   {"attr_indexes": [...], "filter": "<SQL expr>", "bindings": {var: spmat_path}}
  // Delegates parsing/compile/binding to pipeann::dsl::load_filter_from_json,
  // then adopts the returned raw AttrIndex* pointers into native_attr_indexes_
  // / attr_index_map_ so subsequent add() calls and the search path can find
  // them. Returns (Selector* owned by caller, one Attributes per query row).
  std::pair<pipeann::Selector *, std::vector<pipeann::Attributes>>
  load_filter_from_json(const std::string &config_path) {
    if (!use_disk_index_) {
      throw std::runtime_error("load_filter_from_json requires a loaded disk index");
    }
    auto [selector, query_attrs, base_stores] =
        pipeann::dsl::load_filter_from_json(config_path, disk_index_->meta_.npoints);
    for (auto &[key, ai] : base_stores) {
      auto shared_ai = std::shared_ptr<pipeann::AttrIndex>(ai);
      native_attr_indexes_.push_back(shared_ai);
      attr_index_map_[key] = shared_ai.get();
    }
    return {selector, std::move(query_attrs)};
  }

  // Pure scalar filter: run pre_filter on the disk-backed attr indexes (using
  // the index's own reader_), translate the first `limit` vector_ids to tags
  // via the disk index. Result order = vector_id ascending (pre_filter output);
  // callers that need tag ordering should sort themselves. limit==0 → no truncation.
  // When `node_out` is non-null, also populates each match's payload (vector +
  // attrs) keyed by tag, for serving output_fields on the Query path; each
  // matched node is read once (before the id→tag translation).
  std::vector<TagT> filter_only(pipeann::Selector *selector, const pipeann::Attributes &query_attrs,
                                size_t limit, NodeOut *node_out = nullptr) {
    if (selector == nullptr) {
      throw std::runtime_error("filter_only: selector is null");
    }
    auto ids = selector->pre_filter(query_attrs, reader_.get(), /*strict=*/true);
    if (limit > 0 && ids.size() > limit) {
      ids.resize(limit);
    }
    std::vector<TagT> out;
    out.reserve(ids.size());
    for (auto vid : ids) {
      out.push_back(disk_index_->id2tag(vid));
    }
    // Batch-read payloads for output_fields when requested (ids are internal
    // vector ids from pre_filter; get_nodes_by_ids keys the output by tag).
    if (node_out != nullptr) {
      disk_index_->get_nodes_by_ids(ids, node_out);
    }
    return out;
  }

  // Fetch node payloads (vector coords + on-disk filterable attrs) addressed by
  // tag. Used by the Milvus primary-key Query/Get path. Disk index only.
  void get_nodes_by_tags(const std::vector<TagT> &tags, NodeOut *node_out) {
    if (use_disk_index_ && node_out != nullptr) {
      disk_index_->get_nodes_by_tags(tags, node_out);
    }
  }

  void build_mem(const std::string &data_path, const std::string &index_prefix) {
    // sample rate 0.01
    std::string sample_prefix = index_prefix + "_mem_sample";
    gen_random_slice<T>(data_path, sample_prefix, kMemIndexP);

    std::string sample_data_bin = sample_prefix + "_data.bin";
    uint64_t data_num, data_dim;
    pipeann::get_bin_metadata(sample_data_bin, data_num, data_dim);

    std::string sample_id_bin = sample_prefix + "_ids.bin";
    std::ifstream reader;
    reader.open(sample_id_bin, std::ios::binary);
    reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
    uint32_t tags_size = data_num * data_dim;
    std::vector<TagT> tags(data_num);
    reader.read((char *) tags.data(), tags_size * sizeof(uint32_t));
    reader.close();

    auto s = std::chrono::high_resolution_clock::now();
    mem_index_.reset(new pipeann::Index<T, TagT>(metric_, data_dim));
    // We should normalize_cosine here, as data is not pre-normalized.
    mem_index_->build(sample_data_bin.c_str(), data_num, mem_index_params_, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    LOG(INFO) << "Finish building memory index, indexing time: " << diff.count() << "\n";
    std::string save_path = index_prefix + "_mem.index";
    pipeann::SSDIndex<T, TagT>::save_from_mem(*mem_index_, save_path);
  }

  // Single-query search. Returns the number of valid results written.
  // When `range` is finite, routes to range_search and pads unused output
  // slots with sentinels (UINT32_MAX / FLT_MAX).
  // When `node_out` is non-null (disk-index path only), it is populated with the
  // matched nodes' vector + attrs keyed by tag, so the Milvus layer can serve
  // output_fields without a second disk read.
  size_t search(const T *query, uint32_t topk, uint32_t L, TagT *out_ids, float *out_dists,
                pipeann::QueryStats *stats = nullptr, pipeann::Selector *selector = nullptr,
                const pipeann::Attributes *query_attrs = nullptr,
                float range = std::numeric_limits<float>::infinity(), NodeOut *node_out = nullptr) {
    std::vector<TagT> tags(L);
    std::vector<float> distances(L);
    const bool use_range = !std::isinf(range);
    size_t n_valid = L;

    if (selector != nullptr) { // filtered search.
      if (use_disk_index_) {
        // spec_filter_search returns the count of true filter matches (non-member
        // sentinels already dropped); honor it so padding isn't surfaced.
        n_valid = disk_index_->spec_filter_search(query, L, L, selector, *query_attrs, tags.data(), distances.data(),
                                                  32, stats, node_out);
      } else {
        auto cand_ids = selector->pre_filter(*query_attrs, reader_.get(), /*strict=*/true);
        n_valid = mem_index_->search_with_tags(query, L, L, tags.data(), distances.data(), &cand_ids);
      }
    } else if (use_range) { // range search (SSD only).
      n_valid = disk_index_->range_search(query, range, tags.data(), distances.data(), 32, mem_L, L, stats);
    } else { // normal ANNS.
      if (use_disk_index_) {
        disk_index_->pipe_search(query, L, mem_L, L, tags.data(), distances.data(), 32, stats, node_out);
      } else {
        mem_index_->search_with_tags(query, L, L, tags.data(), distances.data());
      }
    }

    size_t pos = 0;
    {
      std::shared_lock<std::shared_mutex> deleted_lock(deleted_nodes_mu_);
      for (size_t j = 0; j < n_valid && pos < topk; j++) {
        if (this->deleted_nodes_set_.find(tags[j]) == this->deleted_nodes_set_.end()) {
          out_ids[pos] = tags[j];
          out_dists[pos] = distances[j];
          pos++;
        }
      }
    }
    for (size_t j = pos; j < topk; j++) {
      out_ids[j] = std::numeric_limits<TagT>::max();
      out_dists[j] = std::numeric_limits<float>::max();
    }
    return pos;
  }

  // Batch search (for Python). Writes results to out_ids and out_dists arrays (n_queries * topk each).
  void search(const T *queries, uint32_t n_queries, uint32_t topk, uint32_t L, TagT *out_ids, float *out_dists,
              pipeann::Selector *selector = nullptr,
              const std::vector<pipeann::Attributes> &query_attrs = std::vector<pipeann::Attributes>(),
              float range = std::numeric_limits<float>::infinity()) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);

    if (!std::isinf(range) && !use_disk_index_) {
      LOG(ERROR) << "Range search requires a loaded disk index";
      size_t total = static_cast<size_t>(n_queries) * static_cast<size_t>(topk);
      std::fill_n(out_ids, total, TagT());
      std::fill_n(out_dists, total, 0.0f);
      return;
    }

    if (selector != nullptr && query_attrs.size() != static_cast<size_t>(n_queries)) {
      throw std::runtime_error("Number of query attrs must match number of queries");
    }

#pragma omp parallel for schedule(dynamic) num_threads(std::min((uint32_t)omp_get_max_threads(), (n_queries + 7) / 8))
    for (uint32_t i = 0; i < n_queries; i++) {
      search(queries + i * data_dim_, topk, L, out_ids + i * topk, out_dists + i * topk, nullptr, selector,
             selector != nullptr ? &query_attrs[i] : nullptr, range);
    }
  }

  // Bulk add.
  void add(const T *vectors, const TagT *tags, uint32_t n_vectors,
           const std::vector<pipeann::Attributes> *attrs_vec = nullptr) {
    save_mu_.lock_shared();

#pragma omp parallel for schedule(dynamic) num_threads(std::min((uint32_t)omp_get_max_threads(), (n_vectors + 7) / 8))
    for (uint32_t i = 0; i < n_vectors; i++) {
      const T *vector_p = vectors + i * data_dim_;
      const pipeann::Attributes *attr_p = (attrs_vec != nullptr) ? &(*attrs_vec)[i] : nullptr;
      do_insert(vector_p, tags[i], attr_p);
    }
    save_mu_.unlock_shared();

    if (!use_disk_index_ && mem_index_->get_num_points() > kBuildThreshold) {
      transform_mem_index_to_disk_index();
    }
  }

  // Single-point insert (for C++ benchmarks / streaming updates).
  int insert(const T *point, const TagT &tag, const pipeann::Attributes *attrs = nullptr) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    do_insert(point, tag, attrs);
    return 0;
  }

  // Single-point lazy delete (marks tag for future merge).
  void lazy_delete(const TagT &tag) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    std::unique_lock<std::shared_mutex> deleted_lock(deleted_nodes_mu_);
    if (deleted_nodes_set_.find(tag) == deleted_nodes_set_.end()) {
      deleted_nodes_set_.insert(tag);
      deleted_nodes_.push_back(tag);
    }
    mem_index_->lazy_delete(tag);
  }

  // Bulk remove.
  void remove(const TagT *tags, uint32_t n_tags) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    std::unique_lock<std::shared_mutex> deleted_lock(deleted_nodes_mu_);
    for (uint32_t i = 0; i < n_tags; i++) {
      TagT tag = tags[i];
      if (deleted_nodes_set_.find(tag) == deleted_nodes_set_.end()) {
        deleted_nodes_set_.insert(tag);
        deleted_nodes_.push_back(tag);
      }
      mem_index_->lazy_delete(tag);
    }
  }

  // Merge in-place inserts and deletions into the disk index, then save to index_prefix.
  // If index_prefix == cur_index_prefix_, uses a double-version strategy to avoid corrupting the live index.
  bool save(const std::string &index_prefix, uint32_t nthreads = 0) {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);

    if (cur_index_prefix_.empty()) {
      LOG(ERROR) << "Current index prefix is empty. Cannot save.";
      return false;
    }

    uint32_t merge_threads = (nthreads > 0) ? nthreads : omp_get_max_threads();

    if (use_disk_index_) {
      std::string out_prefix = index_prefix;
      if (cur_index_prefix_ == index_prefix) {
        out_prefix = index_prefix + "_v2";
        LOG(INFO) << "Saving to the same index prefix. Using double-version: " << out_prefix;
      }

      std::vector<TagT> deleted_nodes_snapshot;
      tsl::robin_set<TagT> deleted_nodes_set_snapshot;
      {
        std::shared_lock<std::shared_mutex> deleted_lock(deleted_nodes_mu_);
        deleted_nodes_snapshot = deleted_nodes_;
        deleted_nodes_set_snapshot = deleted_nodes_set_;
      }
      auto id_map = disk_index_->merge_deletes(cur_index_prefix_, out_prefix, deleted_nodes_snapshot,
                                               deleted_nodes_set_snapshot, merge_threads);

      // Merge attr_indexes.
      for (auto &[key, attr_index] : attr_index_map_) {
        attr_index->merge(id_map);
      }

      disk_index_->reload(out_prefix.c_str());

      if (cur_index_prefix_ == index_prefix) {
        LOG(INFO) << "Copying index from " << out_prefix << " to " << index_prefix;
        disk_index_->copy_index(out_prefix, index_prefix);
        // First reload released merge_lock; re-acquire to pair with the unlock in this reload.
        disk_index_->merge_lock.lock();
        disk_index_->reload(index_prefix.c_str());
      }

      cur_index_prefix_ = index_prefix;
      {
        std::unique_lock<std::shared_mutex> deleted_lock(deleted_nodes_mu_);
        deleted_nodes_.clear();
        deleted_nodes_set_.clear();
      }
    }

    if (!use_disk_index_ || use_mem_index_for_disk_index_) {
      pipeann::SSDIndex<T, TagT>::save_from_mem(*mem_index_, index_prefix + "_mem.index");
    }
#ifdef USE_TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
#endif
    return true;
  }

  std::string to_string() const {
    return std::string("DynamicIndex<") + typeid(T).name() + ">";
  }

  const std::string &index_prefix() const {
    return cur_index_prefix_;
  }

  uint32_t attr_size() const {
    return attr_size_;
  }

 private:
  void do_insert(const T *point_p, TagT tag, const pipeann::Attributes *attrs = nullptr) {
    uint32_t attr_vector_id;

    if (use_disk_index_) {
      int target_id = disk_index_->insert_in_place(point_p, tag, attrs);
      attr_vector_id = static_cast<uint32_t>(target_id);

      if (use_mem_index_for_disk_index_ && rand_uniform() <= kMemIndexP) {  // probably insert into the memory index.
        mem_index_->insert_point(point_p, mem_index_params_, target_id);
      }
    } else {
      int target_id = mem_index_->insert_point(point_p, mem_index_params_, tag);
      attr_vector_id = static_cast<uint32_t>(target_id);
    }

    // Insert attributes into attr_indexes (if loaded).
    if (attrs != nullptr) {
      for (const auto &[key, attr] : attrs->attrs_) {
        auto it = attr_index_map_.find(key);
        if (it == attr_index_map_.end()) {
          if (use_disk_index_) {
            LOG(ERROR) << "Attribute key " << key << " not found in attr_index_map_. Skipping attribute insertion.";
          }
          continue;
        }
        it->second->insert(attr_vector_id, attr);
      }
    }
  }

  bool use_disk_index_ = false;
  bool use_mem_index_for_disk_index_ = false;
  std::shared_ptr<AlignedFileReader> reader_;
  pipeann::AbstractNeighbor<T> *nbr_handler_;
  std::shared_mutex save_mu_;  // save mutex.
  std::string data_path_;
  std::string cur_index_prefix_;
  uint32_t data_dim_;
  pipeann::Metric metric_;
  uint32_t attr_size_ = 0;  // Bytes reserved in each on-disk node for serialized scalar attributes. 0 = infer.
  std::vector<TagT> deleted_nodes_;
  tsl::robin_set<TagT> deleted_nodes_set_;  // copy of deleted nodes.
  mutable std::shared_mutex deleted_nodes_mu_;

  // if vectors are less than the threshold, use mem index instead.
  std::mt19937 gen;
  pipeann::IndexBuildParameters mem_index_params_;
  pipeann::IndexBuildParameters params_;
  std::shared_ptr<pipeann::Index<T, TagT>> mem_index_;
  std::shared_ptr<pipeann::SSDIndex<T, TagT>> disk_index_;
  std::vector<std::shared_ptr<pipeann::AttrIndex>> native_attr_indexes_;
  std::map<uint32_t, pipeann::AttrIndex *> attr_index_map_;  // non-owning key -> attr_index mapping
  uint32_t mem_L = 0;                                        // memory search L in pipe search.
};
