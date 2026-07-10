#ifndef ATTRIBUTE_H_
#define ATTRIBUTE_H_

#include "ssd_index_defs.h"
#include "aligned_file_reader.h"
#include "utils.h"
#include "utils/picojson.h"
#include <algorithm>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <map>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include "filter_utils.h"
#include "utils/cityhash.h"
#include "utils/lock_table.h"
#include "utils/libcuckoo/cuckoohash_map.hh"

namespace pipeann {
  // Currently, we only support vector of uint32_t attributes.
  using Attribute = std::vector<uint32_t>;

  inline Attribute pack_string_attr(std::string_view s) {
    size_t pad = (4 - (s.size() % 4)) % 4;
    Attribute out((s.size() + pad) / 4);
    if (!s.empty())
      std::memcpy(out.data(), s.data(), s.size());
    return out;
  }

  inline std::string_view string_attr_view(const Attribute &a) {
    if (a.empty())
      return std::string_view();
    const char *p = reinterpret_cast<const char *>(a.data());
    size_t n = a.size() * sizeof(uint32_t);
    size_t len = 0;
    while (len < n && p[len] != '\0')
      len++;
    return std::string_view(p, len);
  }

  inline std::string string_attr_to_string(const Attribute &a) {
    return std::string(string_attr_view(a));
  }

  inline Attribute reverse_string_attr(const Attribute &a) {
    std::string out(string_attr_view(a));
    std::reverse(out.begin(), out.end());
    return pack_string_attr(out);
  }

  inline Attribute next_string_attr_prefix(const Attribute &prefix) {
    std::string b(string_attr_view(prefix));
    for (int i = (int) b.size() - 1; i >= 0; --i) {
      unsigned char c = (unsigned char) b[i];
      if (c != 0xFF) {
        b[i] = (char) (c + 1);
        b.resize(i + 1);
        return pack_string_attr(b);
      }
    }
    return Attribute();  // empty = +inf sentinel for upper-bound purposes.
  }

  inline bool string_attr_starts_with(const Attribute &s, const Attribute &prefix) {
    std::string_view sv = string_attr_view(s);
    std::string_view pv = string_attr_view(prefix);
    return sv.size() >= pv.size() && sv.substr(0, pv.size()) == pv;
  }

  inline bool string_attr_ends_with(const Attribute &s, const Attribute &suffix) {
    std::string_view sv = string_attr_view(s);
    std::string_view pv = string_attr_view(suffix);
    return sv.size() >= pv.size() && sv.substr(sv.size() - pv.size()) == pv;
  }

  inline bool str_attr_less(const Attribute &a, const Attribute &b) {
    const char *pa = a.empty() ? "" : reinterpret_cast<const char *>(a.data());
    const char *pb = b.empty() ? "" : reinterpret_cast<const char *>(b.data());
    return std::strcmp(pa, pb) < 0;
  }

  struct StrAttrLess {
    bool operator()(const Attribute &a, const Attribute &b) const {
      return str_attr_less(a, b);
    }
  };

  inline std::tuple<int64_t, std::vector<int64_t>, std::vector<int32_t>, std::vector<float>> load_spmat(
      const std::string &spmat_filename) {
    std::ifstream reader(spmat_filename, std::ios::binary);
    if (!reader.is_open()) {
      LOG(ERROR) << "Failed to open spmat file: " << spmat_filename;
      crash();
    }

    // Read header: 3 int64 values (nrow, ncol, nnz)
    int64_t nrow, ncol, nnz;
    reader.read((char *) &nrow, sizeof(int64_t));
    reader.read((char *) &ncol, sizeof(int64_t));
    reader.read((char *) &nnz, sizeof(int64_t));

    LOG(INFO) << "Loading spmat: nrow=" << nrow << ", ncol=" << ncol << ", nnz=" << nnz;

    // Read indptr: nrow+1 int64 values
    std::vector<int64_t> indptr(nrow + 1);
    reader.read((char *) indptr.data(), (nrow + 1) * sizeof(int64_t));

    // Read indices: nnz int32 values
    std::vector<int32_t> indices(nnz);
    reader.read((char *) indices.data(), nnz * sizeof(int32_t));

    // Read data: nnz float32 values
    std::vector<float> data(nnz);
    reader.read((char *) data.data(), nnz * sizeof(float));

    return std::make_tuple(nrow, indptr, indices, data);
  }

  // Multiple attributes are stored as KV pairs.
  struct Attributes {
    std::unordered_map<uint32_t, Attribute> attrs_;

    bool find(uint32_t key) const {
      return attrs_.find(key) != attrs_.end();
    }

    const Attribute &get(uint32_t key) const {
      return attrs_.at(key);
    }

    void set(uint32_t key, const Attribute &attr) {
      attrs_[key] = attr;
    }

    void remove(uint32_t key) {
      attrs_.erase(key);
    }

    void clear() {
      attrs_.clear();
    }

    size_t serialized_size() const {
      size_t cnt = 1;
      for (const auto &[_, attr] : attrs_) {
        cnt += 2 + attr.size();
      }
      return sizeof(uint32_t) * cnt;
    }

    // serialize: [n_keys]
    // For each key, [key, n_attrs, attr_0, ..., attr_n]
    void serialize(char *buffer) const {
      uint32_t size = attrs_.size();
      memcpy(buffer, &size, sizeof(uint32_t));
      buffer += sizeof(uint32_t);
      for (const auto &[key, attr] : attrs_) {
        memcpy(buffer, &key, sizeof(uint32_t));
        buffer += sizeof(uint32_t);
        uint32_t n_attrs = attr.size();
        memcpy(buffer, &n_attrs, sizeof(uint32_t));
        buffer += sizeof(uint32_t);
        for (uint32_t j = 0; j < n_attrs; j++) {
          memcpy(buffer, &attr[j], sizeof(uint32_t));
          buffer += sizeof(uint32_t);
        }
      }
    }

    static Attributes deserialize(const char *buffer) {
      Attributes ret;
      uint32_t size, key, n_attrs;
      memcpy(&size, buffer, sizeof(uint32_t));
      buffer += sizeof(uint32_t);
      for (uint32_t i = 0; i < size; i++) {
        memcpy(&key, buffer, sizeof(uint32_t));
        buffer += sizeof(uint32_t);
        memcpy(&n_attrs, buffer, sizeof(uint32_t));
        buffer += sizeof(uint32_t);
        Attribute attr(n_attrs);
        memcpy(attr.data(), buffer, n_attrs * sizeof(uint32_t));
        buffer += n_attrs * sizeof(uint32_t);
        ret.set(key, std::move(attr));
      }
      return ret;
    }

    void print() const {
      for (const auto &[key, attr] : attrs_) {
        std::string attrs_str = "";
        for (const auto &attr_value : attr) {
          attrs_str += std::to_string(attr_value) + " ";
        }
        LOG(INFO) << "Key: " << key << " Attrs: " << attrs_str;
      }
      LOG(INFO) << "Total attr keys: " << attrs_.size();
    }
  };

  // AttrIndex: Abstract base class for on-SSD attribute indexes.
  // Including on-SSD index and in-memory probabilistic structures for speculative filtering.
  struct AttrIndex {
    std::string filename;
    std::atomic<uint64_t> n_vectors;  // Current vector ID upper bound, including unmerged delta attrs.
    uint64_t base_n_vectors_;         // Number of vectors stored in the attr index file.
    int fd;

    std::vector<Attribute *> delta_attrs_;  // Default insert storage keyed by vector_id - base_n_vectors_.
    ReaderOptSharedMutex delta_attrs_mu_;   // Extremely read-heavy (each in-filter acquires), so ReaderOpt.

    AttrIndex(const std::string &filename, uint64_t n_vectors) {
      this->filename = filename;
      this->fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
      if (fd == -1) {
        LOG(ERROR) << "Failed to open AttrIndex file: " << filename;
        crash();
      }
      this->n_vectors = n_vectors;
      this->base_n_vectors_ = n_vectors;
    }

    virtual ~AttrIndex() {
      release_delta_attrs();
      close(fd);
    }

    // Load the full attrribute data from file to memory so get(id) can be used.
    virtual void load_attrs() = 0;
    // Load the full attribute data from in-memory rows (for Python helpers).
    virtual void load_from_rows(const std::vector<Attribute> &rows) = 0;
    // Load only statistics & approx filters to memory (for search).
    virtual void load_approx() = 0;
    // Get the attribute value(s) for a vector (requires load_attrs()).
    virtual Attribute get(uint32_t vector_id) = 0;
    // Save the attribute index to file.
    virtual void save() = 0;
    // Maximum number of attribute values per vector.
    virtual size_t max_attrs() = 0;

    // Insert or overwrite the attribute for an inserted vector.
    // Delta attrs are keyed by vector_id - base_n_vectors_; base_n_vectors_
    // stays fixed until save/merge materializes the delta into the index file.
    // For concurrently-inserting vectors, their attributes may not be read.
    // However, this is safe because it is ANNS.
    // Insert in serial, but it is not the performance bottleneck.
    virtual void insert(uint32_t vector_id, const Attribute &attr) {
      uint32_t delta_id = vector_id - base_n_vectors_;
      Attribute *new_attr = new Attribute(attr);

      // delta_attrs_.size() is the capacity.
      if (unlikely(delta_id >= delta_attrs_.size())) {
        delta_attrs_mu_.lock();
        if (likely(delta_id >= delta_attrs_.size())) {
          delta_attrs_.resize(std::max<size_t>(1.5 * delta_id, 100));
        }
        delta_attrs_mu_.unlock();
      }

      delta_attrs_mu_.lock_shared();
      delta_attrs_[delta_id] = new_attr;
      atomic_max(n_vectors, (uint64_t) vector_id + 1);
      delta_attrs_mu_.unlock_shared();
      return;
    }

    // Merge delta attributes with on-disk index.
    // id_map: old_id -> new_id, used to reorder attributes during merge.
    virtual void merge(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) = 0;

    // --- Cost estimation ---

    // Estimate the number of vectors satisfying the query constraint.
    virtual uint32_t estimate_count(const Attribute &query) = 0;
    // Estimate the precision of is_member_approx: TP / (TP + FP). 1.0 = exact (no false positives).
    virtual double estimate_precision(const Attribute &query) = 0;
    // Estimate the number of SSD pages read during speculative pre-filtering.
    virtual uint32_t estimate_prefilter_reads(const Attribute &query) = 0;

    // --- Speculative pre-filtering ---
    // Scan the on-SSD attribute index to return a superset of valid vector IDs.
    // May skip high-selectivity branches to reduce I/O, deferring to exact verification.
    virtual VectorIDList pre_filter(const Attribute &query, AlignedFileReader *reader) = 0;

    // --- Speculative in-filtering ---
    // Estimate the number of SSD pages read during in-filter preparation.
    virtual uint32_t estimate_infilter_reads(const Attribute &query) = 0;
    // Pre-scan cold/rare attribute entries from SSD before graph traversal.
    // Returns a VectorIDList of cold-attribute matches for use in is_member_approx().
    virtual VectorIDList prepare_in_filter(const Attribute &query, AlignedFileReader *reader) = 0;
    // Fast in-memory approximate membership check. Guarantees no false negatives:
    // if it returns false, the vector is definitely invalid.
    // Uses Bloom filters (for label attrs) or quantized values (for range attrs).
    // The `list` parameter contains pre-scanned cold-attribute matches from prepare_in_filter().
    virtual bool is_member_approx(uint32_t target_id, const Attribute &query, const VectorIDList &list) = 0;

    // Prefetch the cacheline is_member_approx(target_id) will read (per-vector
    // bucket id / bloom filter). Default no-op; overridden by indexes whose check
    // is a random probe into a dataset-sized array.
    virtual void prefetch_approx(uint32_t /*target_id*/, const Attribute & /*query*/, const VectorIDList & /*list*/) {}

    // In-filter search pins this for the whole graph traversal (see
    // Selector::lock_shared) instead of locking inside every is_member_approx
    // call; per-call lock/unlock pairs were ~24% of in-filter CPU. Excludes
    // concurrent merge (exclusive); a merge waits for in-flight queries (~ms).
    void lock_shared() {
      delta_attrs_mu_.lock_shared();
    }
    void unlock_shared() {
      delta_attrs_mu_.unlock_shared();
    }

   protected:
    static uint64_t atomic_max(std::atomic<uint64_t> &target, uint64_t value) {
      uint64_t old_value = target;
      while (old_value < value && !target.compare_exchange_weak(old_value, value, std::memory_order_relaxed)) {
      }
      return std::max(old_value, value);
    }

    void release_delta_attrs() {
      for (auto *attr : delta_attrs_) {
        delete attr;
      }
      delta_attrs_.clear();
    }
  };

  // BloomFilter: A non-owning slice over a byte buffer representing a single bloom filter.
  // Supports arbitrary byte sizes (1, 2, 3, 4, 8, 16, ...) via per-bit operations.
  struct BloomFilter {
    uint8_t *data;
    uint32_t size;  // bytes

    BloomFilter(uint8_t *ptr, uint32_t size) : data(ptr), size(size) {
    }
    BloomFilter(const uint8_t *ptr, uint32_t size) : data(const_cast<uint8_t *>(ptr)), size(size) {
    }

    // Number of hash functions, scaling with filter size.
    static uint32_t num_hash_functions(uint32_t filter_bytes) {
      uint32_t bits = filter_bytes * 8;
      return std::max(2u, std::min(8u, bits / 8));
    }

    uint32_t num_hash_functions() const {
      return num_hash_functions(size);
    }

    // Add a label to the bloom filter (set the corresponding bits).
    void add(uint32_t label) {
      uint32_t bits = size * 8;
      uint32_t h1 = mix_hash(label);
      uint32_t h2 = mix_hash(label ^ 0x9e3779b9);
      uint32_t k = num_hash_functions();
      for (uint32_t i = 0; i < k; i++) {
        uint32_t bit = (h1 + i * h2) % bits;
        data[bit / 8] |= (1u << (bit % 8));
      }
    }

    // Test if a label is possibly in the bloom filter (may have false positives).
    bool contains(uint32_t label) const {
      uint32_t bits = size * 8;
      uint32_t h1 = mix_hash(label);
      uint32_t h2 = mix_hash(label ^ 0x9e3779b9);
      uint32_t k = num_hash_functions();
      for (uint32_t i = 0; i < k; i++) {
        uint32_t bit = (h1 + i * h2) % bits;
        if (!(data[bit / 8] & (1u << (bit % 8))))
          return false;
      }
      return true;
    }

    // Estimated FPR for n items inserted.
    // FPR = (1 - e^{-kn/m})^k
    static double estimated_fpr(uint32_t filter_bytes, double n_items = 4.0) {
      uint32_t m = filter_bytes * 8;
      uint32_t k = num_hash_functions(filter_bytes);
      return std::pow(1.0 - std::exp(-1.0 * k * n_items / m), k);
    }

    static inline uint32_t mix_hash(uint32_t k) {
      k ^= k >> 16;
      k *= 0x85ebca6b;
      k ^= k >> 13;
      k *= 0xc2b2ae35;
      k ^= k >> 16;
      return k;
    }
  };

  // A label attribute index tailored for label-filtered ANNS:
  // In this format, labels are stored column-wise as an inverted index:
  // First is (2N + 1) uint64_t, [n_labels, start_of_label_0, end_of_label_0, ..., end_of_label_n] (page-aligned)
  // Then, for each label, [vector_id_0, ..., vector_id_n] (the start of each label is page-aligned)
  struct InvertedLabelAttrIndex : public AttrIndex {
    std::vector<std::pair<uint64_t, uint64_t>> label_loc_;     // histogram: 2 uint64_t per label
    std::vector<Attribute> labels_;                            // data: 1 Attribute per vector
    size_t max_attrs_ = 0;                                     // maximum number of labels per vector.
    static constexpr uint32_t kApproxFilterThreshold = 65536;  // 256KB.
    std::vector<uint8_t> bloom_filters_;  // bloom filter for hot labels (bloom_bytes_per_point_ bytes per vector).
    uint32_t bloom_bytes_per_point_ = 4;  // bytes per vector for bloom filter (e.g. 2, 4, 8).
    std::map<uint32_t, std::vector<uint32_t>> delta_;
    uint64_t delta_bytes_ = 0;

    InvertedLabelAttrIndex(const std::string &filename, uint64_t n_vectors, uint32_t bloom_bytes = 4)
        : AttrIndex(filename, n_vectors), bloom_bytes_per_point_(bloom_bytes) {
    }

    // Only load labels_, not load label_loc_. For index building.
    void load_from_spmat(const std::string &spmat_filename) {
      auto [nrow, indptr, indices, data] = load_spmat(spmat_filename);

      this->n_vectors = nrow;
      this->base_n_vectors_ = nrow;
      // Process the sparse matrix: for each row i, collect labels where data != 0
      labels_.resize(nrow);
      max_attrs_ = 0;
      for (int64_t i = 0; i < nrow; i++) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        for (int64_t j = start; j < end; j++) {
          // x[i][j] != 0 means vector i contains label indices[j]
          if (data[j] != 0.0f) {
            labels_[i].push_back(static_cast<uint32_t>(indices[j]));
          }
        }
        max_attrs_ = std::max(max_attrs_, labels_[i].size());
      }
    }

    void load_from_rows(const std::vector<Attribute> &rows) override {
      labels_ = rows;
      n_vectors = rows.size();
      base_n_vectors_ = rows.size();
      max_attrs_ = 0;
      for (const auto &row : rows) {
        max_attrs_ = std::max(max_attrs_, row.size());
      }
    }

    size_t max_attrs() override {
      return max_attrs_;
    }

    void load_attrs() override {
      read_label_locs();
      std::ifstream reader(filename, std::ios::binary);
      if (!reader.is_open()) {
        LOG(ERROR) << "Failed to open inverted index file: " << filename;
        crash();
      }

      labels_.assign(base_n_vectors_, Attribute());
      for (uint32_t i = 0; i < label_loc_.size(); i++) {
        // read the ith label's vectors.
        auto [st, ed] = label_loc_[i];
        std::vector<uint32_t> buf((ed - st) / sizeof(uint32_t));
        reader.seekg(st, std::ios::beg);
        reader.read((char *) buf.data(), (ed - st));
        for (uint32_t j = 0; j < buf.size(); j++) {
          labels_[buf[j]].push_back(i);
        }
      }
    }

    void read_label_locs() {
      std::ifstream reader(filename, std::ios::binary);
      if (!reader.is_open()) {
        LOG(ERROR) << "Failed to open inverted index file: " << filename;
        crash();
      }
      uint64_t n_labels;
      reader.read((char *) &n_labels, sizeof(uint64_t));

      this->label_loc_.resize(n_labels);
      for (size_t i = 0; i < n_labels; i++) {
        reader.read((char *) &label_loc_[i].first, sizeof(uint64_t));
        reader.read((char *) &label_loc_[i].second, sizeof(uint64_t));
      }
    }

    // Load label_loc_ and bloom filters.
    void load_approx() override {
      read_label_locs();

      // The .filter file stores uint8_t with ndim = bloom_bytes_per_point_.
      size_t npts, dim;
      pipeann::load_bin<uint8_t>(filename + ".filter", bloom_filters_, npts, dim);
      bloom_bytes_per_point_ = dim;
    }

    Attribute get(uint32_t vector_id) override {
      return labels_[vector_id];
    }

    // Get a BloomFilter slice for a given vector.
    BloomFilter get_bloom_filter(uint32_t vector_id) {
      return BloomFilter(&bloom_filters_[vector_id * bloom_bytes_per_point_], bloom_bytes_per_point_);
    }

    void save_approx() {
      pipeann::save_bin<uint8_t>(filename + ".filter", bloom_filters_.data(), n_vectors, bloom_bytes_per_point_);
    }

    void remap_approx(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) {
      uint64_t new_n_vectors = id_map.empty() ? (uint64_t) n_vectors : id_map.size();
      std::vector<uint8_t> remapped(new_n_vectors * bloom_bytes_per_point_, 0);
      uint32_t old_n_vectors = std::min<uint64_t>(bloom_filters_.size() / bloom_bytes_per_point_, n_vectors);
      for (uint32_t old_id = 0; old_id < old_n_vectors; old_id++) {
        uint32_t new_id;
        if (id_map.empty()) {
          new_id = old_id;
        } else if (!id_map.find(old_id, new_id)) {
          continue;
        }
        memcpy(remapped.data() + new_id * bloom_bytes_per_point_,
               bloom_filters_.data() + old_id * bloom_bytes_per_point_, bloom_bytes_per_point_);
      }
      bloom_filters_ = std::move(remapped);
    }

    void save_inverted(const std::vector<std::vector<uint32_t>> &vector_ids) {
      // Compute offsets.
      size_t n_labels = vector_ids.size();
      // 1 n_labels + 2 * n_labels uint64_t for label_loc_.
      uint64_t cur_offset = ROUND_UP(sizeof(uint64_t) + (2 * n_labels) * sizeof(uint64_t), SECTOR_LEN);
      label_loc_.resize(n_labels);

      for (size_t i = 0; i < n_labels; i++) {
        label_loc_[i].first = cur_offset;
        label_loc_[i].second = cur_offset + vector_ids[i].size() * sizeof(uint32_t);
        cur_offset += ROUND_UP(vector_ids[i].size() * sizeof(uint32_t), SECTOR_LEN);
      }

      std::ofstream writer(filename, std::ios::binary);
      if (!writer.is_open()) {
        LOG(ERROR) << "Failed to open inverted index file: " << filename;
        crash();
      }

      // Write [n_labels, start_of_label_0, end_of_label_0, ..., end_of_label_n]
      writer.write((char *) &n_labels, sizeof(uint64_t));
      for (size_t i = 0; i < n_labels; i++) {
        writer.write((char *) &label_loc_[i].first, sizeof(uint64_t));
        writer.write((char *) &label_loc_[i].second, sizeof(uint64_t));
      }

      // for each label, write vector IDs.
      for (size_t i = 0; i < n_labels; i++) {
        writer.seekp(label_loc_[i].first, std::ios::beg);
        for (uint32_t j = 0; j < vector_ids[i].size(); j++) {
          writer.write((char *) &vector_ids[i][j], sizeof(uint32_t));
        }
        assert(writer.tellp() == (off_t) label_loc_[i].second);
      }
      writer.close();
    }

    void save() override {
      // We assume that only labels_ is correct.
      uint32_t max_label_id = 0;
      for (size_t i = 0; i < labels_.size(); i++) {
        for (auto &label : labels_[i]) {
          if (label > max_label_id) {
            max_label_id = label;
          }
        }
      }

      std::vector<std::vector<uint32_t>> vector_ids(max_label_id + 1);
      for (size_t i = 0; i < labels_.size(); i++) {
        for (auto &label : labels_[i]) {
          vector_ids[label].push_back(i);
        }
      }

      bloom_filters_.assign(n_vectors * bloom_bytes_per_point_, 0);
      for (size_t label = 0; label < vector_ids.size(); label++) {
        if (vector_ids[label].size() < kApproxFilterThreshold) {
          continue;
        }
        for (auto vector_id : vector_ids[label]) {
          get_bloom_filter(vector_id).add(label);
        }
      }

      save_inverted(vector_ids);

      base_n_vectors_ = n_vectors;
      save_approx();
    }

    void insert(uint32_t vector_id, const Attribute &attr) override {
      // delta_attrs_mu_ is repurposed to protect delta indexes and bloom filters.
      delta_attrs_mu_.lock();
      for (uint32_t label : attr) {
        // New label, very rare, so do not overprovision.
        // This ensures label_loc_.size() == n_labels.
        if (unlikely(label >= label_loc_.size())) {
          label_loc_.resize(label + 1);
        }
        delta_[label].push_back(vector_id);
        delta_bytes_ += 2 * sizeof(uint32_t);  // a conservative estimation.
      }

      atomic_max(n_vectors, (uint64_t) vector_id + 1);
      max_attrs_ = std::max(max_attrs_, attr.size());

      size_t req_size = bloom_bytes_per_point_ * (vector_id + 1);
      if (unlikely(req_size >= bloom_filters_.size())) {
        // resize.
        bloom_filters_.resize(std::max(1.5 * req_size, 1024.0));
      }

      BloomFilter bf = get_bloom_filter(vector_id);
      for (auto label : attr) {
        bf.add(label);
      }

      if (delta_bytes_ >= std::max<uint64_t>(4 * 1024 * 1024, bloom_bytes_per_point_ * base_n_vectors_ / 8)) {
        do_merge(libcuckoo::cuckoohash_map<uint32_t, uint32_t>());
      }
      delta_attrs_mu_.unlock();
    }

    uint32_t estimate_label_cnt(uint32_t label) {
      // for label, its count is (end_of_label - start_of_label) / sizeof(uint32_t)
      // A query label unseen in the base data (>= label_loc_ size) has zero matches.
      if (unlikely(label >= label_loc_.size())) {
        return 0;
      }
      auto [st, ed] = label_loc_[label];
      return (ed - st) / sizeof(uint32_t);
    }

    // Functions below handle the OR case, namely,
    // vectors that satisfy at least one given label in the dataset labels.
    uint32_t estimate_count(const Attribute &query) override {
      uint32_t count = 0;
      for (auto &label : query) {
        count += estimate_label_cnt(label);
      }
      return std::min((uint64_t) count, (uint64_t) n_vectors);
    }

    // Estimate the precision of bloom-filter-based approximate membership test.
    // Precision = TP / (TP + FP).
    double estimate_precision(const Attribute &query) override {
      uint32_t tp = estimate_count(query);
      uint32_t n_hot_labels = 0;
      for (auto &label : query) {
        if (!is_cold_label(label)) {
          n_hot_labels++;
        }
      }
      if (n_hot_labels == 0) {
        return 1.0;  // All cold labels use exact inverted index.
      }
      double fpr_per_label = BloomFilter::estimated_fpr(bloom_bytes_per_point_);
      // For OR: FP rate = 1 - (1 - fpr_per_label)^n_hot_labels
      double fpr = 1.0 - std::pow(1.0 - fpr_per_label, n_hot_labels);
      double fp = fpr * (n_vectors - tp);
      if (tp + fp < 1.0)
        return 1.0;
      return tp / (tp + fp);
    }

    uint32_t estimate_prefilter_reads(const Attribute &query) override {
      return estimate_count(query) * sizeof(uint32_t) / SECTOR_LEN;
    }

    // Prefilter by loading the inverted lists from SSD.
    // Each label is read with one request, then merge all lists.
    VectorIDList pre_filter(const Attribute &query, AlignedFileReader *reader) override {
      auto ctx = reader->get_ctx();

      if (unlikely(query.size() == 0)) {
        return VectorIDList();
      }

      // Read each label's inverted list into its own VectorIDList
      // [buf, n_items, delta_buf, n_delta_items]
      std::vector<std::tuple<uint32_t *, size_t, uint32_t *, size_t>> lists;
      std::vector<IORequest> reqs;

      delta_attrs_mu_.lock_shared();
      for (auto &label : query) {
        // A query label unseen in the base data has no inverted list; skip it.
        if (unlikely(label >= label_loc_.size())) {
          continue;
        }
        auto [range_st, range_ed] = label_loc_[label];
        uint64_t len = range_ed - range_st;
        auto delta_it = delta_.find(label);
        auto delta_ptr = delta_it == delta_.end() ? nullptr : delta_it->second.data();
        uint64_t n_delta = delta_it == delta_.end() ? 0 : delta_it->second.size();

        if (unlikely(len + n_delta * sizeof(uint32_t) == 0)) {
          continue;
        }

        uint32_t *buf = nullptr;
        pipeann::alloc_aligned((void **) &buf, ROUND_UP(len + n_delta * sizeof(uint32_t), SECTOR_LEN), SECTOR_LEN);
        if (len > 0) {
          reqs.push_back(IORequest(range_st, ROUND_UP(len, SECTOR_LEN), buf, range_st, len));
        }
        lists.push_back({buf, len / sizeof(uint32_t) + n_delta, delta_ptr, n_delta});
      }

      if (!reqs.empty()) {
        reader->read_fd(fd, reqs, ctx);
      }
      for (size_t i = 0; i < lists.size(); i++) {
        auto &[buf, n_items, delta_ptr, n_delta] = lists[i];
        if (delta_ptr != nullptr) {
          memcpy(buf + n_items - n_delta, delta_ptr, n_delta * sizeof(uint32_t));
        }
      }
      delta_attrs_mu_.unlock_shared();

      // sort by size, small first for merging.
      std::sort(lists.begin(), lists.end(),
                [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); });

      VectorIDList result;
      for (auto &list : lists) {
        result = or_sorted_unique(result.data(), result.size(), std::get<0>(list), std::get<1>(list));
        pipeann::aligned_free(std::get<0>(list));
      }
      return result;
    }

    uint32_t estimate_infilter_reads(const Attribute &query) override {
      uint32_t count = 0;
      for (auto &label : query) {
        if (is_cold_label(label)) {
          count += estimate_label_cnt(label);
        }
      }
      return count * sizeof(uint32_t) / SECTOR_LEN;
    }

    VectorIDList prepare_in_filter(const Attribute &query, AlignedFileReader *reader) override {
      Attribute cold_labels;
      for (auto &label : query) {
        if (is_cold_label(label)) {
          cold_labels.push_back(label);
        }
      }
      return pre_filter(cold_labels, reader);
    }

    bool is_member_approx(uint32_t target_id, const Attribute &query, const VectorIDList &cold_list) override {
      // First, check if the target_id is in the list.
      if (std::binary_search(cold_list.begin(), cold_list.end(), target_id)) {
        return true;
      }

      // For hot labels, use bloom filter to check if query in target_id's labels.
      // No locking here: the caller holds lock_shared() across the traversal.
      auto bf = get_bloom_filter(target_id);
      for (auto &label : query) {
        if (is_cold_label(label)) {
          continue;
        }
        if (bf.contains(label)) {
          return true;
        }
      }
      return false;
    }

    // Prefetch the per-vector bloom filter bytes (random probe into bloom_filters_,
    // sized bloom_bytes_per_point_ * n_vectors) that is_member_approx will read.
    void prefetch_approx(uint32_t target_id, const Attribute & /*query*/, const VectorIDList & /*cold*/) override {
      pipeann::cpu_prefetch_t0(&bloom_filters_[(size_t) target_id * bloom_bytes_per_point_]);
    }

    // ---- AND semantics: all query labels must be present ----

    uint32_t estimate_count_and(const Attribute &query) {
      if (query.empty())
        return 0;
      uint32_t min_count = estimate_label_cnt(query[0]);
      for (size_t i = 1; i < query.size(); i++) {
        min_count = std::min(min_count, estimate_label_cnt(query[i]));
      }
      return min_count;
    }

    double estimate_precision_and(const Attribute &query) {
      uint32_t n_hot_labels = 0;
      for (auto &label : query) {
        if (!is_cold_label(label)) {
          n_hot_labels++;
        }
      }
      if (n_hot_labels == 0) {
        return 1.0;
      }
      // For AND: FP requires all bloom filter checks to pass simultaneously.
      // FPR ≈ fpr_per_label^n_hot_labels (much lower than OR).
      double fpr_per_label = BloomFilter::estimated_fpr(bloom_bytes_per_point_);
      double fpr = std::pow(fpr_per_label, n_hot_labels);
      uint32_t tp = estimate_count_and(query);
      double fp = fpr * (n_vectors - tp);
      if (tp + fp < 1.0)
        return 1.0;
      return tp / (tp + fp);
    }

    uint32_t estimate_prefilter_reads_and(const Attribute &query) {
      return estimate_count(query) * sizeof(uint32_t) / SECTOR_LEN;
    }

    VectorIDList pre_filter_and(const Attribute &query, AlignedFileReader *reader) {
      auto ctx = reader->get_ctx();

      if (unlikely(query.size() == 0)) {
        return VectorIDList();
      }

      // [buf, n_items, delta_buf, n_delta_items]
      std::vector<std::tuple<uint32_t *, size_t, uint32_t *, size_t>> lists;
      std::vector<IORequest> reqs;
      delta_attrs_mu_.lock_shared();
      for (auto &label : query) {
        // For AND, a query label unseen in the base data (>= label_loc_ size)
        // has an empty inverted list, so the intersection is empty.
        bool out_of_range = label >= label_loc_.size();
        auto [range_st, range_ed] = out_of_range ? std::make_pair<uint64_t, uint64_t>(0, 0) : label_loc_[label];
        uint64_t len = range_ed - range_st;
        auto delta_it = out_of_range ? delta_.end() : delta_.find(label);
        auto delta_ptr = delta_it == delta_.end() ? nullptr : delta_it->second.data();
        uint64_t n_delta = delta_it == delta_.end() ? 0 : delta_it->second.size();
        if (unlikely(len + n_delta * sizeof(uint32_t) == 0)) {
          delta_attrs_mu_.unlock_shared();
          for (auto &list : lists) {
            pipeann::aligned_free(std::get<0>(list));
          }
          return VectorIDList();
        }

        uint32_t *buf = nullptr;
        pipeann::alloc_aligned((void **) &buf, ROUND_UP(len + n_delta * sizeof(uint32_t), SECTOR_LEN), SECTOR_LEN);
        if (len > 0) {
          reqs.push_back(IORequest(range_st, ROUND_UP(len, SECTOR_LEN), buf, range_st, len));
        }
        lists.push_back({buf, len / sizeof(uint32_t) + n_delta, delta_ptr, n_delta});
      }

      if (!reqs.empty()) {
        reader->read_fd(fd, reqs, ctx);
      }
      for (size_t i = 0; i < lists.size(); i++) {
        auto &[buf, n_items, delta_ptr, n_delta] = lists[i];
        if (delta_ptr != nullptr) {
          memcpy(buf + n_items - n_delta, delta_ptr, n_delta * sizeof(uint32_t));
        }
      }
      delta_attrs_mu_.unlock_shared();

      // sort by size, small first for efficient intersection.
      std::sort(lists.begin(), lists.end(),
                [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); });

      VectorIDList result(std::get<0>(lists[0]), std::get<0>(lists[0]) + std::get<1>(lists[0]));
      pipeann::aligned_free(std::get<0>(lists[0]));
      for (size_t i = 1; i < lists.size(); i++) {
        result = and_sorted_unique(result.data(), result.size(), std::get<0>(lists[i]), std::get<1>(lists[i]));
        pipeann::aligned_free(std::get<0>(lists[i]));
      }
      return result;
    }

    uint32_t estimate_infilter_reads_and(const Attribute &query) {
      uint32_t count = 0;
      for (auto &label : query) {
        if (is_cold_label(label)) {
          count += estimate_label_cnt(label);
        }
      }
      return count * sizeof(uint32_t) / SECTOR_LEN;
    }

    VectorIDList prepare_in_filter_and(const Attribute &query, AlignedFileReader *reader) {
      Attribute cold_labels;
      for (auto &label : query) {
        if (is_cold_label(label)) {
          cold_labels.push_back(label);
        }
      }
      if (cold_labels.empty()) {
        return VectorIDList();
      }
      return pre_filter_and(cold_labels, reader);
    }

    bool is_member_approx_and(uint32_t target_id, const Attribute &query, const VectorIDList &cold_list) {
      bool has_cold = false;
      for (auto &label : query) {
        if (is_cold_label(label)) {
          has_cold = true;
          break;
        }
      }

      // If there are cold labels, target must be in the intersection (cold_list).
      if (has_cold && !std::binary_search(cold_list.begin(), cold_list.end(), target_id)) {
        return false;
      }

      // All hot labels must pass bloom filter.
      // No locking here: the caller holds lock_shared() across the traversal.
      auto bf = get_bloom_filter(target_id);
      for (auto &label : query) {
        if (is_cold_label(label)) {
          continue;
        }
        if (!bf.contains(label)) {
          return false;
        }
      }
      return true;
    }

    // Merge delta attributes with on-disk index.
    // id_map: old_id -> new_id for reordering during merge_deletes.
    void merge(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) override {
      delta_attrs_mu_.lock();
      do_merge(id_map);
      delta_attrs_mu_.unlock();
    }

   private:
    bool is_cold_label(uint32_t label) {
      return estimate_label_cnt(label) < kApproxFilterThreshold;
    }

    void do_merge(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) {
      std::vector<std::vector<uint32_t>> vector_ids(label_loc_.size());
      uint64_t merged_size = id_map.empty() ? (uint64_t) n_vectors : id_map.size();

      std::ifstream reader(filename, std::ios::binary);
      if (!reader.is_open()) {
        LOG(ERROR) << "Failed to open inverted index file: " << filename;
        crash();
      }

      for (uint32_t label = 0; label < label_loc_.size(); label++) {
        auto [st, ed] = label_loc_[label];
        if (st == ed) {
          continue;
        }
        std::vector<uint32_t> old_ids((ed - st) / sizeof(uint32_t));
        reader.seekg(st, std::ios::beg);
        reader.read((char *) old_ids.data(), ed - st);
        for (uint32_t old_id : old_ids) {
          uint32_t new_id = 0;
          if (id_map.empty()) {
            new_id = old_id;
            vector_ids[label].push_back(new_id);
          } else if (id_map.find(old_id, new_id)) {
            vector_ids[label].push_back(new_id);
          }
        }
      }

      for (const auto &[label, ids] : delta_) {
        if (label >= vector_ids.size()) {
          vector_ids.resize(label + 1);
        }
        for (uint32_t old_id : ids) {
          uint32_t new_id = 0;
          if (id_map.empty()) {
            new_id = old_id;
            vector_ids[label].push_back(new_id);
          } else if (id_map.find(old_id, new_id)) {
            vector_ids[label].push_back(new_id);
          }
        }
      }

      max_attrs_ = 0;
      std::vector<uint32_t> attr_counts(merged_size, 0);
      for (auto &ids : vector_ids) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (uint32_t id : ids) {
          attr_counts[id]++;
          max_attrs_ = std::max(max_attrs_, (size_t) attr_counts[id]);
        }
      }

      n_vectors = merged_size;
      base_n_vectors_ = merged_size;

      save_inverted(vector_ids);
      remap_approx(id_map);
      save_approx();

      delta_.clear();
      delta_bytes_ = 0;
    }
  };

  // A scalar attr index tailored for sorted range filters:
  // In this format, attrs are stored in Structure of Arrays (SoA):
  // [attr1, attr2, ..., attrN][ID1, ID2, ..., IDN]
  // Both attrs and IDs are sorted by attr value.
  // This allows direct I/O read into VectorIDList without extraction.
  struct SortedRangeAttrIndex : public AttrIndex {
    static constexpr uint32_t kHistogramBuckets = 1000;
    static constexpr uint32_t kQuantizeBuckets = 256;
    // Histogram: Xp Value: X =[0/kHistogramBuckets, 1/kHistogramBuckets, ..., kHistogramBuckets-1/kHistogramBuckets]
    std::vector<uint32_t> histogram_;  // <attr>
    std::vector<uint32_t> attrs_;      // attrs_[ID] = attr (only loaded with load())
    size_t stride;                     // stride of the histogram (4KB aligned).
    uint64_t ids_offset;               // offset to IDs block in file.
    std::map<uint32_t, std::vector<uint32_t>> delta_;
    uint64_t delta_bytes_ = 0;

    // Quantized in-memory representation for is_member_approx.
    // bucket_ids_[vector_id] = bucket index (0..kQuantizeBuckets-1)
    std::vector<uint8_t> bucket_ids_;
    // bucket_boundaries_[i] = minimum attr value for bucket i; bucket_boundaries_[kQuantizeBuckets] = max_val+1
    std::vector<uint32_t> bucket_boundaries_;

    SortedRangeAttrIndex(const std::string &filename, uint64_t n_vectors) : AttrIndex(filename, n_vectors) {
      ids_offset = ROUND_UP(base_n_vectors_ * sizeof(uint32_t), SECTOR_LEN);
    }

    size_t max_attrs() override {
      return 1;
    }

    void load_attrs() override {
      std::ifstream reader(filename, std::ios::binary);

      // Read attrs block (offset 0).
      std::vector<uint32_t> sorted_attrs(base_n_vectors_);
      reader.read((char *) sorted_attrs.data(), base_n_vectors_ * sizeof(uint32_t));

      // Read IDs block (offset ids_offset, page-aligned).
      std::vector<uint32_t> sorted_ids(base_n_vectors_);
      reader.seekg(ids_offset, std::ios::beg);
      reader.read((char *) sorted_ids.data(), base_n_vectors_ * sizeof(uint32_t));

      // Initialize attrs_ (mapping ID -> attr).
      attrs_.resize(base_n_vectors_);
      for (size_t i = 0; i < base_n_vectors_; i++) {
        attrs_[sorted_ids[i]] = sorted_attrs[i];
      }
    }

    void load_histogram() {
      std::ifstream reader(filename, std::ios::binary);
      histogram_.clear();
      stride = std::max<uint64_t>(1, ROUND_UP(base_n_vectors_ / kHistogramBuckets, SECTOR_LEN / sizeof(uint32_t)));
      for (size_t i = 0; i < base_n_vectors_; i += stride) {
        uint32_t attr;
        reader.seekg(i * sizeof(uint32_t), std::ios::beg);
        reader.read((char *) &attr, sizeof(uint32_t));
        histogram_.push_back(attr);
      }
    }

    void load_approx() override {
      load_histogram();
      load_quantized_buckets();
    }

    Attribute get(uint32_t vector_id) override {
      return Attribute({attrs_[vector_id]});
    }

    // The bin stores one scalar attr per vector.
    void load_from_bin(const std::string &bin_filename) {
      uint64_t npts, ndim;
      pipeann::load_bin(bin_filename, this->attrs_, npts, ndim);
      this->n_vectors = this->attrs_.size();
      this->base_n_vectors_ = this->attrs_.size();
      this->ids_offset = ROUND_UP(this->base_n_vectors_ * sizeof(uint32_t), SECTOR_LEN);
    }

    void load_from_spmat(const std::string &spmat_filename) {
      auto [nrow, indptr, indices, data] = load_spmat(spmat_filename);

      this->n_vectors = nrow;
      this->base_n_vectors_ = nrow;
      this->ids_offset = ROUND_UP(this->base_n_vectors_ * sizeof(uint32_t), SECTOR_LEN);
      this->attrs_.resize(nrow);
      for (int64_t i = 0; i < nrow; i++) {
        int64_t start = indptr[i];
        attrs_[i] = data[start];
      }
    }

    void load_from_rows(const std::vector<Attribute> &rows) override {
      n_vectors = rows.size();
      base_n_vectors_ = rows.size();
      ids_offset = ROUND_UP(base_n_vectors_ * sizeof(uint32_t), SECTOR_LEN);
      attrs_.resize(base_n_vectors_);
      for (size_t i = 0; i < base_n_vectors_; i++) {
        attrs_[i] = rows[i][0];
      }
    }

    void save() override {
      delta_attrs_mu_.lock();
      ids_offset = ROUND_UP(n_vectors * sizeof(uint32_t), SECTOR_LEN);

      std::vector<std::pair<uint32_t, uint32_t>> data(n_vectors);
      for (uint32_t i = 0; i < n_vectors; i++) {
        data[i] = {attrs_[i], i};
      }
      std::sort(data.begin(), data.end());

      std::vector<uint32_t> sorted_attrs(n_vectors), sorted_ids(n_vectors);
      for (size_t i = 0; i < n_vectors; i++) {
        sorted_attrs[i] = data[i].first;
        sorted_ids[i] = data[i].second;
      }

      write_sorted_index(sorted_attrs, sorted_ids);
      // Build and save quantized bucket representation.
      build_quantized_buckets(sorted_attrs, sorted_ids);
      save_quantized_buckets();
      delta_attrs_mu_.unlock();
    }

    void insert(uint32_t vector_id, const Attribute &attr) override {
      delta_attrs_mu_.lock();
      uint32_t value = attr[0];
      delta_[value].push_back(vector_id);
      delta_bytes_ += 2 * sizeof(uint32_t);
      atomic_max(n_vectors, (uint64_t) vector_id + 1);

      if (unlikely(vector_id >= bucket_ids_.size())) {
        bucket_ids_.resize(std::max(1.5 * vector_id, 1024.0));
      }
      auto it = std::upper_bound(bucket_boundaries_.begin(), bucket_boundaries_.end(), value);
      uint32_t bid = it == bucket_boundaries_.begin() ? 0 : (uint32_t) (it - bucket_boundaries_.begin() - 1);
      bucket_ids_[vector_id] = (uint8_t) std::min(bid, kQuantizeBuckets - 1);

      if (delta_bytes_ >= std::max<uint64_t>(4 * 1024 * 1024, 2 * sizeof(uint32_t) * base_n_vectors_ / 8)) {
        do_merge(libcuckoo::cuckoohash_map<uint32_t, uint32_t>());
      }
      delta_attrs_mu_.unlock();
    }

    void write_sorted_index(const std::vector<uint32_t> &sorted_attrs, const std::vector<uint32_t> &sorted_ids) {
      ids_offset = ROUND_UP(sorted_attrs.size() * sizeof(uint32_t), SECTOR_LEN);
      std::ofstream writer(filename, std::ios::binary);
      if (!writer.is_open()) {
        LOG(ERROR) << "Failed to open sorted range attr file: " << filename;
        crash();
      }

      writer.write((char *) sorted_attrs.data(), sorted_attrs.size() * sizeof(uint32_t));
      writer.seekp(ids_offset, std::ios::beg);
      writer.write((char *) sorted_ids.data(), sorted_ids.size() * sizeof(uint32_t));
      writer.close();
    }

    auto get_range(const Attribute &query) {
      uint32_t l = query[0], r = query.size() > 1 ? query[1] : l + 1;
      auto it1 = std::lower_bound(histogram_.begin(), histogram_.end(), l);
      if (it1 != histogram_.begin()) {
        it1 = std::prev(it1);
      }
      auto it2 = std::lower_bound(histogram_.begin(), histogram_.end(), r);
      return std::make_pair(it1, it2);
    }

    // Range attrs are encoded as [l, r).
    // estimate the upper bound.
    uint32_t estimate_count(const Attribute &query) override {
      auto [it1, it2] = get_range(query);
      return std::min((uint64_t) (std::distance(it1, it2) * stride), base_n_vectors_);
    }

    // Estimate the precision of quantized approximate membership test.
    // The quantized value covers a bucket range [bucket_boundaries_[i], bucket_boundaries_[i+1]).
    // True positives: vectors whose exact value is in [l, r).
    // False positives: vectors in boundary buckets whose exact value is outside [l, r).
    // We estimate precision using the histogram for TP and the bucket boundaries for total positives.
    double estimate_precision(const Attribute &query) override {
      if (bucket_boundaries_.empty()) {
        return 1.0;  // No quantization loaded, assume exact.
      }
      uint32_t l = query[0], r = query.size() > 1 ? query[1] : l + 1;

      // Find the bucket range that overlaps with [l, r).
      // A bucket i covers [bucket_boundaries_[i], bucket_boundaries_[i+1]).
      // Bucket i overlaps [l,r) iff bucket_boundaries_[i] < r && bucket_boundaries_[i+1] > l.
      uint32_t tp = estimate_count(query);  // Approximate true positive count from histogram.

      // Count total positives from quantized buckets (all vectors in overlapping buckets).
      uint32_t total_positive = 0;
      for (uint32_t i = 0; i < kQuantizeBuckets; i++) {
        uint32_t b_lo = bucket_boundaries_[i];
        uint32_t b_hi = bucket_boundaries_[i + 1];
        if (b_lo < r && b_hi > l) {
          // This bucket overlaps with [l, r).
          // Count vectors in this bucket using bucket_boundaries_ stride.
          uint32_t bucket_start = (uint64_t) i * base_n_vectors_ / kQuantizeBuckets;
          uint32_t bucket_end = (uint64_t) (i + 1) * base_n_vectors_ / kQuantizeBuckets;
          total_positive += bucket_end - bucket_start;
        }
      }

      if (total_positive == 0)
        return 1.0;
      double precision = (double) tp / total_positive;
      return std::min(precision, 1.0);
    }

    uint32_t estimate_prefilter_reads(const Attribute &query) override {
      auto [it1, it2] = get_range(query);
      size_t idx_l = (it1 - histogram_.begin()) * stride;
      size_t idx_r = std::min((it2 - histogram_.begin()) * stride, (size_t) base_n_vectors_);
      // Need to read both attr and ID segments.
      return 2 * DIV_ROUND_UP((idx_r - idx_l) * sizeof(uint32_t), SECTOR_LEN);
    }

    VectorIDList pre_filter(const Attribute &query, AlignedFileReader *reader) override {
      auto [it1, it2] = get_range(query);
      size_t idx_l = (it1 - histogram_.begin()) * stride;
      size_t idx_r = std::min((it2 - histogram_.begin()) * stride, (size_t) base_n_vectors_);
      size_t n_elems = idx_r - idx_l;

      auto ctx = reader->get_ctx();
      uint64_t attr_file_offset = idx_l * sizeof(uint32_t);
      uint64_t id_file_offset = ids_offset + idx_l * sizeof(uint32_t);
      uint64_t read_len = ROUND_UP(n_elems * sizeof(uint32_t), SECTOR_LEN);

      uint32_t *attr_buf = nullptr;
      uint32_t *id_buf = nullptr;
      VectorIDList result;

      if (n_elems > 0) {
        pipeann::alloc_aligned((void **) &attr_buf, ROUND_UP(read_len, SECTOR_LEN), SECTOR_LEN);
        pipeann::alloc_aligned((void **) &id_buf, ROUND_UP(read_len, SECTOR_LEN), SECTOR_LEN);
        std::vector<IORequest> reqs = {
            IORequest(attr_file_offset, read_len, attr_buf, attr_file_offset, n_elems * sizeof(uint32_t)),
            IORequest(id_file_offset, read_len, id_buf, id_file_offset, n_elems * sizeof(uint32_t))};
        reader->read_fd(fd, reqs, ctx);

        // Attrs are sorted. Binary search on them to find exact [l, r) boundaries.
        uint32_t l = query[0], r = query.size() > 1 ? query[1] : l + 1;
        auto begin_it = std::lower_bound(attr_buf, attr_buf + n_elems, l);
        auto end_it = std::lower_bound(attr_buf, attr_buf + n_elems, r);
        size_t start_idx = begin_it - attr_buf;
        size_t end_idx = end_it - attr_buf;

        // Extract matching IDs and sort them (IDs are ordered by attr, not by ID value).
        result = VectorIDList(id_buf + start_idx, id_buf + end_idx);

        pipeann::aligned_free(attr_buf);
        pipeann::aligned_free(id_buf);
      }

      uint32_t l = query[0], r = query.size() > 1 ? query[1] : l + 1;
      delta_attrs_mu_.lock_shared();
      auto delta_it = delta_.lower_bound(l);
      while (delta_it != delta_.end() && delta_it->first < r) {
        for (auto id : delta_it->second) {
          result.push_back(id);
        }
        delta_it++;
      }
      delta_attrs_mu_.unlock_shared();
      std::sort(result.begin(), result.end());
      return result;
    }

    // In-filter does not need to read from SSD if quantized buckets are loaded in memory.
    uint32_t estimate_infilter_reads(const Attribute &query) override {
      return 0;
    }

    VectorIDList prepare_in_filter(const Attribute &query, AlignedFileReader *reader) override {
      return VectorIDList();
    }

    // Approximate membership test using quantized bucket IDs.
    // A vector is considered a member if its bucket range overlaps with the query range.
    bool is_member_approx(uint32_t target_id, const Attribute &query, const VectorIDList &list) override {
      // No locking here: the caller holds lock_shared() across the traversal.
      uint32_t l = query[0], r = query.size() > 1 ? query[1] : l + 1;
      // Use quantized bucket check if available.
      uint8_t bid = bucket_ids_[target_id];

      // Bucket bid covers [bucket_boundaries_[bid], bucket_boundaries_[bid+1]).
      // Check if this range overlaps with [l, r).
      return bucket_boundaries_[bid] < r && bucket_boundaries_[bid + 1] > l;
    }

    // The dominant cost of is_member_approx is this random probe into bucket_ids_
    // (uint8_t per vector; ~1GB at 1e9 pts). Prefetch its cacheline so the check
    // a few neighbors later hits L1/L2 instead of stalling on a memory read.
    void prefetch_approx(uint32_t target_id, const Attribute & /*query*/, const VectorIDList & /*list*/) override {
      pipeann::cpu_prefetch_t0(&bucket_ids_[target_id]);
    }

   private:
    // Build quantized bucket IDs from sorted attrs and sorted IDs.
    // sorted_attrs[i] and sorted_ids[i] are sorted by attr value.
    void build_quantized_buckets(const std::vector<uint32_t> &sorted_attrs, const std::vector<uint32_t> &sorted_ids) {
      bucket_boundaries_.resize(kQuantizeBuckets + 1);
      bucket_ids_.resize(n_vectors);

      // Divide sorted vectors into kQuantizeBuckets equal-depth buckets.

      for (uint32_t b = 0; b < kQuantizeBuckets; b++) {
        size_t idx = (uint64_t) b * n_vectors / kQuantizeBuckets;
        bucket_boundaries_[b] = sorted_attrs[idx];
      }
      // sentinel: upper bound of the last bucket.
      bucket_boundaries_[kQuantizeBuckets] = sorted_attrs[n_vectors - 1] + 1;

      // Assign bucket IDs to each vector by its original ID.
      for (size_t i = 0; i < n_vectors; i++) {
        uint8_t bid = (uint8_t) std::min((uint32_t) (i * kQuantizeBuckets / n_vectors), kQuantizeBuckets - 1);
        bucket_ids_[sorted_ids[i]] = bid;
      }
    }

    void save_quantized_buckets() {
      std::string quant_file = filename + ".quantize";
      std::ofstream writer(quant_file, std::ios::binary);
      if (!writer.is_open()) {
        LOG(ERROR) << "Failed to open quantize file: " << quant_file;
        return;
      }
      // Write bucket boundaries: (kQuantizeBuckets + 1) uint32_t values.
      uint32_t n_buckets = kQuantizeBuckets;
      writer.write((char *) &n_buckets, sizeof(uint32_t));
      writer.write((char *) bucket_boundaries_.data(), (kQuantizeBuckets + 1) * sizeof(uint32_t));
      // Write bucket IDs: n_vectors uint8_t values.
      uint32_t n_pts = bucket_ids_.size();
      writer.write((char *) &n_pts, sizeof(uint32_t));
      writer.write((char *) bucket_ids_.data(), n_pts * sizeof(uint8_t));
      writer.close();
      LOG(INFO) << "Saved quantized buckets to " << quant_file;
    }

    void load_quantized_buckets() {
      std::string quant_file = filename + ".quantize";
      std::ifstream reader(quant_file, std::ios::binary);
      if (!reader.is_open()) {
        LOG(WARNING) << "Quantize file not found: " << quant_file;
        return;
      }
      uint32_t n_buckets;
      reader.read((char *) &n_buckets, sizeof(uint32_t));
      bucket_boundaries_.resize(n_buckets + 1);
      reader.read((char *) bucket_boundaries_.data(), (n_buckets + 1) * sizeof(uint32_t));

      uint32_t n_pts;
      reader.read((char *) &n_pts, sizeof(uint32_t));
      bucket_ids_.resize(n_pts);
      reader.read((char *) bucket_ids_.data(), n_pts * sizeof(uint8_t));
      reader.close();
      LOG(INFO) << "Loaded quantized buckets from " << quant_file << ": " << n_buckets << " buckets, " << n_pts
                << " vectors";
    }

    void remap_approx(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) {
      std::vector<uint8_t> remapped(id_map.empty() ? bucket_ids_.size() : id_map.size());
      for (uint32_t old_id = 0; old_id < bucket_ids_.size(); old_id++) {
        uint32_t new_id = 0;
        if (id_map.empty()) {
          new_id = old_id;
        } else if (!id_map.find(old_id, new_id)) {
          continue;
        }
        remapped[new_id] = bucket_ids_[old_id];
      }
      bucket_ids_ = std::move(remapped);
    }

    void merge(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) override {
      delta_attrs_mu_.lock();
      do_merge(id_map);
      delta_attrs_mu_.unlock();
    }

   private:
    // Merge delta attributes with on-disk index.
    // id_map: old_id -> new_id for reordering during merge_deletes.
    void do_merge(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) {
      ids_offset = ROUND_UP(base_n_vectors_ * sizeof(uint32_t), SECTOR_LEN);

      std::vector<uint32_t> old_attrs(base_n_vectors_);
      std::vector<uint32_t> old_ids(base_n_vectors_);
      std::ifstream reader(filename, std::ios::binary);
      if (!reader.is_open()) {
        LOG(ERROR) << "Failed to open sorted range attr file: " << filename;
        crash();
      }
      reader.read((char *) old_attrs.data(), base_n_vectors_ * sizeof(uint32_t));
      reader.seekg(ids_offset, std::ios::beg);
      reader.read((char *) old_ids.data(), base_n_vectors_ * sizeof(uint32_t));

      size_t delta_size = 0;
      for (const auto &[_, ids] : delta_) {
        delta_size += ids.size();
      }

      std::vector<std::pair<uint32_t, uint32_t>> data;
      data.reserve(base_n_vectors_ + delta_size);
      for (uint32_t i = 0; i < base_n_vectors_; i++) {
        uint32_t new_id = 0;
        if (id_map.empty()) {
          new_id = old_ids[i];
          data.push_back({old_attrs[i], new_id});
        } else if (id_map.find(old_ids[i], new_id)) {
          data.push_back({old_attrs[i], new_id});
        }
      }

      for (const auto &[value, ids] : delta_) {
        for (uint32_t old_id : ids) {
          uint32_t new_id = 0;
          if (id_map.empty()) {
            new_id = old_id;
            data.push_back({value, new_id});
          } else if (id_map.find(old_id, new_id)) {
            data.push_back({value, new_id});
          }
        }
      }

      std::sort(data.begin(), data.end());
      std::vector<uint32_t> sorted_attrs(data.size()), sorted_ids(data.size());
      for (size_t i = 0; i < data.size(); i++) {
        sorted_attrs[i] = data[i].first;
        sorted_ids[i] = data[i].second;
      }

      uint64_t merged_size = id_map.empty() ? (uint64_t) n_vectors : id_map.size();
      n_vectors = merged_size;
      base_n_vectors_ = merged_size;
      histogram_.clear();
      stride = std::max<uint64_t>(1, ROUND_UP(base_n_vectors_ / kHistogramBuckets, SECTOR_LEN / sizeof(uint32_t)));
      for (size_t i = 0; i < base_n_vectors_; i += stride) {
        histogram_.push_back(sorted_attrs[i]);
      }

      remap_approx(id_map);
      write_sorted_index(sorted_attrs, sorted_ids);
      save_quantized_buckets();

      delta_.clear();
      delta_bytes_ = 0;
    }
  };

  // Sorted string attr index. Supports equality, prefix, suffix, and LIKE queries.
  // Two sorted files: prefix (sorted by string) and suffix (sorted by reversed string).
  //
  // On-disk layout (same for both files):
  //   [u32 n_entries][u32 n_sampled][u64 × n_sampled: sampled_offsets]
  //   [entry_0][entry_1]...[entry_{n-1}]
  //   Each entry: [u32 vector_id][u32 key_len][u32 × key_len: key_data]
  //
  // In-memory (loaded by load_approx):
  //   - sampled_offsets[i] = byte offset of entry at index i*stride
  //   - histogram[i] = key of entry at index i*stride (for in-memory binary search)
  //   - hashes[vector_id] = CityHash32(key) for exact-eq is_member_approx
  //   - bucket_boundaries + bucket_ids for directional range is_member_approx
  struct StringAttrIndex : public AttrIndex {
    enum Direction : uint32_t { PREFIX = 0, SUFFIX = 1, N_DIRECTIONS = 2 };
    static constexpr uint32_t kBuckets = 256;
    static constexpr uint32_t kSampleRate = 1000;

    int suffix_fd_ = -1;

    // Per-direction in-memory state (loaded from file header).
    uint32_t n_entries_[N_DIRECTIONS] = {};
    std::vector<uint64_t> sampled_offsets_[N_DIRECTIONS];
    std::vector<Attribute> histogram_[N_DIRECTIONS];
    uint64_t file_size_[N_DIRECTIONS] = {};

    // Approx filters.
    std::vector<uint32_t> hashes_;                            // per vector_id, CityHash32
    std::vector<uint32_t> bucket_ids_[N_DIRECTIONS];          // per vector_id, bucket index
    std::vector<Attribute> bucket_boundaries_[N_DIRECTIONS];  // kBuckets+1 boundary keys

    // Full data (only when load_attrs() called).
    std::vector<Attribute> strings_;
    size_t max_attrs_ = 0;

    // Delta (pending inserts, merged periodically).
    std::map<Attribute, std::vector<uint32_t>, StrAttrLess> delta_[N_DIRECTIONS];
    uint64_t delta_bytes_ = 0;

    StringAttrIndex(const std::string &filename, uint64_t n_vectors) : AttrIndex(filename, n_vectors) {
      suffix_fd_ = open((filename + ".rev").c_str(), O_RDWR | O_CREAT, 0666);
      if (suffix_fd_ == -1) {
        LOG(ERROR) << "Failed to open suffix file: " << filename << ".rev";
        crash();
      }
    }

    ~StringAttrIndex() override {
      if (suffix_fd_ != -1)
        close(suffix_fd_);
    }

    int fd_for(Direction dir) const {
      return dir == PREFIX ? fd : suffix_fd_;
    }

    std::string approx_filename(Direction dir) const {
      return dir == PREFIX ? filename + ".prefix" : filename + ".suffix";
    }

    static uint32_t hash_packed(const Attribute &a) {
      return CityHash32(reinterpret_cast<const char *>(a.data()), a.size() * sizeof(uint32_t));
    }

    static bool like_match(std::string_view s, std::string_view p) {
      size_t i = 0, j = 0, star = std::string::npos, mark = 0;
      while (i < s.size()) {
        if (j < p.size() && (p[j] == '_' || p[j] == s[i])) {
          i++;
          j++;
        } else if (j < p.size() && p[j] == '%') {
          star = j++;
          mark = i;
        } else if (star != std::string::npos) {
          j = star + 1;
          i = ++mark;
        } else
          return false;
      }
      while (j < p.size() && p[j] == '%')
        j++;
      return j == p.size();
    }

    static Attribute like_fixed_prefix(const Attribute &pattern) {
      std::string_view p = string_attr_view(pattern);
      size_t len = 0;
      for (char c : p) {
        if (c == '%' || c == '_')
          break;
        len++;
      }
      return pack_string_attr(p.substr(0, len));
    }

    static Attribute like_fixed_suffix(const Attribute &pattern) {
      std::string_view p = string_attr_view(pattern);
      size_t start = p.size();
      for (int i = (int) p.size() - 1; i >= 0; --i) {
        if (p[i] == '%' || p[i] == '_')
          break;
        start = (size_t) i;
      }
      return pack_string_attr(p.substr(start));
    }

    // --- Load / Save ---

    size_t max_attrs() override {
      return max_attrs_;
    }

    void load_from_rows(const std::vector<Attribute> &rows) override {
      strings_ = rows;
      n_vectors = rows.size();
      base_n_vectors_ = rows.size();
      max_attrs_ = 0;
      for (const auto &r : rows)
        max_attrs_ = std::max(max_attrs_, r.size());
    }

    Attribute get(uint32_t vector_id) override {
      if (vector_id >= strings_.size())
        return Attribute();
      return strings_[vector_id];
    }

    // Read file header: n_entries, sampled_offsets, histogram keys.
    void load_header(Direction dir) {
      int f = fd_for(dir);
      off_t sz = lseek(f, 0, SEEK_END);
      file_size_[dir] = sz < 0 ? 0 : (uint64_t) sz;
      if (file_size_[dir] < 2 * sizeof(uint32_t)) {
        n_entries_[dir] = 0;
        return;
      }
      lseek(f, 0, SEEK_SET);
      uint32_t n = 0, ns = 0;
      std::ignore = read(f, &n, sizeof(uint32_t));
      std::ignore = read(f, &ns, sizeof(uint32_t));
      n_entries_[dir] = n;
      sampled_offsets_[dir].resize(ns);
      if (ns)
        std::ignore = read(f, sampled_offsets_[dir].data(), ns * sizeof(uint64_t));

      // Build histogram by reading the key at each sampled offset.
      histogram_[dir].clear();
      histogram_[dir].reserve(ns);
      for (uint32_t i = 0; i < ns; i++) {
        lseek(f, sampled_offsets_[dir][i], SEEK_SET);
        uint32_t id = 0, klen = 0;
        std::ignore = read(f, &id, sizeof(uint32_t));
        std::ignore = read(f, &klen, sizeof(uint32_t));
        Attribute key(klen);
        if (klen)
          std::ignore = read(f, key.data(), klen * sizeof(uint32_t));
        histogram_[dir].push_back(std::move(key));
      }
    }

    void load_attrs() override {
      load_header(PREFIX);
      strings_.assign(base_n_vectors_, Attribute());
      max_attrs_ = 0;
      // Scan all entries sequentially.
      if (n_entries_[PREFIX] == 0)
        return;
      uint64_t data_offset = 2 * sizeof(uint32_t) + sampled_offsets_[PREFIX].size() * sizeof(uint64_t);
      std::ifstream reader(filename, std::ios::binary);
      reader.seekg(data_offset);
      for (uint32_t i = 0; i < n_entries_[PREFIX]; i++) {
        uint32_t id = 0, klen = 0;
        reader.read((char *) &id, sizeof(uint32_t));
        reader.read((char *) &klen, sizeof(uint32_t));
        Attribute key(klen);
        if (klen)
          reader.read((char *) key.data(), klen * sizeof(uint32_t));
        if (id >= strings_.size())
          strings_.resize(id + 1);
        max_attrs_ = std::max(max_attrs_, (size_t) klen);
        strings_[id] = std::move(key);
      }
    }

    void load_approx() override {
      for (Direction dir : {PREFIX, SUFFIX})
        load_header(dir);
      size_t npts, dim;
      if (file_exists(filename + ".filter"))
        pipeann::load_bin<uint32_t>(filename + ".filter", hashes_, npts, dim);
      for (Direction dir : {PREFIX, SUFFIX})
        load_bucket_approx(dir);
    }

    void save() override {
      std::vector<std::pair<Attribute, uint32_t>> entries;
      entries.reserve(strings_.size());
      for (uint32_t i = 0; i < (uint32_t) strings_.size(); i++)
        entries.push_back({strings_[i], i});
      save_both(entries);
    }

    // --- Estimation (pure in-memory) ---

    // Coarse range from histogram: returns [lo_sample, hi_sample) indices into sampled_offsets.
    std::pair<size_t, size_t> get_range(Direction dir, const Attribute &lo, const Attribute &hi) const {
      auto &h = histogram_[dir];
      auto l_it = std::lower_bound(h.begin(), h.end(), lo, str_attr_less);
      if (l_it != h.begin())
        l_it = std::prev(l_it);
      auto r_it = hi.empty() ? h.end() : std::lower_bound(h.begin(), h.end(), hi, str_attr_less);
      return {(size_t) (l_it - h.begin()), (size_t) (r_it - h.begin())};
    }

    uint32_t estimate_count(const Attribute &query) override {
      return estimate_count(PREFIX, query);
    }

    uint32_t estimate_count(Direction dir, const Attribute &query) {
      if (histogram_[dir].empty())
        return 0;
      Attribute lo = dir == PREFIX ? query : reverse_string_attr(query);
      Attribute hi = next_string_attr_prefix(lo);
      auto [l, r] = get_range(dir, lo, hi);
      uint64_t cnt = (r - l) * kSampleRate;
      // Add delta contribution.
      for (auto it = delta_[dir].lower_bound(lo);
           it != delta_[dir].end() && (hi.empty() || str_attr_less(it->first, hi)); ++it)
        cnt += it->second.size();
      return (uint32_t) std::min<uint64_t>(cnt, n_vectors.load());
    }

    uint32_t estimate_count_like(const Attribute &pattern) {
      Attribute prefix = like_fixed_prefix(pattern);
      Attribute suffix = like_fixed_suffix(pattern);
      uint32_t best = (uint32_t) n_vectors.load();
      if (!prefix.empty())
        best = std::min(best, estimate_count(PREFIX, prefix));
      if (!suffix.empty())
        best = std::min(best, estimate_count(SUFFIX, suffix));
      return best;
    }

    double estimate_precision(const Attribute &) override {
      return 1.0;
    }

    double estimate_precision(Direction dir, const Attribute &query) {
      uint32_t tp = estimate_count(dir, query);
      uint32_t total = estimate_bucket_total(dir, query);
      if (total == 0)
        return 1.0;
      return std::min((double) tp / total, 1.0);
    }

    uint32_t estimate_bucket_total(Direction dir, const Attribute &query) {
      if (bucket_boundaries_[dir].empty() || base_n_vectors_ == 0)
        return 0;
      Attribute lo = dir == PREFIX ? query : reverse_string_attr(query);
      Attribute hi = next_string_attr_prefix(lo);
      auto l = std::lower_bound(bucket_boundaries_[dir].begin(), bucket_boundaries_[dir].end(), lo, str_attr_less);
      auto r = hi.empty() ? bucket_boundaries_[dir].end()
                          : std::lower_bound(bucket_boundaries_[dir].begin(), bucket_boundaries_[dir].end(), hi,
                                             str_attr_less);
      int lb = std::max<int>(0, (int) (l - bucket_boundaries_[dir].begin()) - 1);
      int rb = std::min<int>((int) kBuckets - 1, (int) (r - bucket_boundaries_[dir].begin()) + 1);
      if (rb < lb)
        return 0;
      return (uint32_t) std::min<uint64_t>((uint64_t) (rb - lb + 1) * base_n_vectors_ / kBuckets, n_vectors.load());
    }

    uint32_t estimate_prefilter_reads(const Attribute &query) override {
      return estimate_prefilter_reads(PREFIX, query);
    }

    uint32_t estimate_prefilter_reads(Direction dir, const Attribute &query) {
      uint32_t cnt = estimate_count(dir, query);
      uint32_t avg_entry = 2 * sizeof(uint32_t) + std::max<size_t>(max_attrs_, 1) * sizeof(uint32_t);
      return DIV_ROUND_UP((uint64_t) cnt * avg_entry, SECTOR_LEN);
    }

    uint32_t estimate_prefilter_reads_like(const Attribute &pattern) {
      Attribute prefix = like_fixed_prefix(pattern);
      Attribute suffix = like_fixed_suffix(pattern);
      uint32_t best = std::numeric_limits<uint32_t>::max() / 2;
      if (!prefix.empty())
        best = std::min(best, estimate_prefilter_reads(PREFIX, prefix));
      if (!suffix.empty())
        best = std::min(best, estimate_prefilter_reads(SUFFIX, suffix));
      return best;
    }

    uint32_t estimate_infilter_reads(const Attribute &) override {
      return 0;
    }

    // --- Pre-filter (SSD reads via AlignedFileReader) ---

    // Scan a byte range of entries, calling fn(key, id) for each.
    template<typename Fn>
    void scan_range(Direction dir, uint64_t byte_begin, uint64_t byte_end, AlignedFileReader *reader, Fn fn) {
      if (byte_end <= byte_begin)
        return;
      uint64_t aligned_off = byte_begin / SECTOR_LEN * SECTOR_LEN;
      uint64_t read_len = ROUND_UP(byte_end - aligned_off, SECTOR_LEN);
      char *buf = nullptr;
      pipeann::alloc_aligned((void **) &buf, read_len, SECTOR_LEN);
      std::vector<IORequest> reqs = {IORequest(aligned_off, read_len, buf, aligned_off, byte_end - aligned_off)};
      auto ctx = reader->get_ctx();
      reader->read_fd(fd_for(dir), reqs, ctx);

      size_t off = byte_begin - aligned_off;
      size_t limit = byte_end - aligned_off;
      while (off + 2 * sizeof(uint32_t) <= limit) {
        uint32_t id = 0, klen = 0;
        std::memcpy(&id, buf + off, sizeof(uint32_t));
        std::memcpy(&klen, buf + off + sizeof(uint32_t), sizeof(uint32_t));
        size_t entry_bytes = 2 * sizeof(uint32_t) + (size_t) klen * sizeof(uint32_t);
        if (off + entry_bytes > limit)
          break;
        const uint32_t *key_ptr = reinterpret_cast<const uint32_t *>(buf + off + 2 * sizeof(uint32_t));
        fn(key_ptr, klen, id);
        off += entry_bytes;
      }
      pipeann::aligned_free(buf);
    }

    // Get byte range [begin, end) covering entries in [lo_key, hi_key).
    std::pair<uint64_t, uint64_t> byte_range_for(Direction dir, const Attribute &lo, const Attribute &hi) const {
      auto &h = histogram_[dir];
      auto &so = sampled_offsets_[dir];
      if (h.empty() || so.empty())
        return {0, 0};
      // Find first sample >= lo.
      auto l_it = std::lower_bound(h.begin(), h.end(), lo, str_attr_less);
      size_t l_idx = l_it == h.begin() ? 0 : (size_t) (l_it - h.begin() - 1);
      // Find first sample >= hi.
      size_t r_idx;
      if (hi.empty()) {
        r_idx = so.size();  // to end of file
      } else {
        auto r_it = std::lower_bound(h.begin(), h.end(), hi, str_attr_less);
        r_idx = (size_t) (r_it - h.begin());
        if (r_idx < so.size())
          r_idx++;  // include one past to be safe
      }
      uint64_t begin = so[l_idx];
      uint64_t end = r_idx >= so.size() ? file_size_[dir] : so[r_idx];
      return {begin, end};
    }

    VectorIDList pre_filter(const Attribute &query, AlignedFileReader *reader) override {
      return pre_filter(PREFIX, query, reader);
    }

    VectorIDList pre_filter(Direction dir, const Attribute &query, AlignedFileReader *reader) {
      Attribute lo = dir == PREFIX ? query : reverse_string_attr(query);
      Attribute hi = next_string_attr_prefix(lo);
      auto [begin, end] = byte_range_for(dir, lo, hi);

      VectorIDList result;
      if (begin < end) {
        scan_range(dir, begin, end, reader, [&](const uint32_t *key_ptr, uint32_t klen, uint32_t id) {
          // Check if key is in [lo, hi).
          Attribute key(key_ptr, key_ptr + klen);
          if (!str_attr_less(key, lo) && (hi.empty() || str_attr_less(key, hi)))
            result.push_back(id);
        });
      }

      // Merge delta.
      delta_attrs_mu_.lock_shared();
      for (auto it = delta_[dir].lower_bound(lo);
           it != delta_[dir].end() && (hi.empty() || str_attr_less(it->first, hi)); ++it) {
        for (uint32_t id : it->second)
          result.push_back(id);
      }
      delta_attrs_mu_.unlock_shared();

      std::sort(result.begin(), result.end());
      result.erase(std::unique(result.begin(), result.end()), result.end());
      return result;
    }

    VectorIDList pre_filter_like(const Attribute &pattern, AlignedFileReader *reader) {
      std::string_view pat = string_attr_view(pattern);
      Direction dir = PREFIX;
      Attribute range_key;
      if (!pat.empty() && pat.front() != '%' && pat.front() != '_') {
        range_key = like_fixed_prefix(pattern);
        dir = PREFIX;
      } else if (!pat.empty() && pat.back() != '%' && pat.back() != '_') {
        range_key = reverse_string_attr(like_fixed_suffix(pattern));
        dir = SUFFIX;
      }

      Attribute hi = range_key.empty() ? Attribute() : next_string_attr_prefix(range_key);
      auto [begin, end] =
          range_key.empty()
              ? std::make_pair((uint64_t) (2 * sizeof(uint32_t) + sampled_offsets_[dir].size() * sizeof(uint64_t)),
                               file_size_[dir])
              : byte_range_for(dir, range_key, hi);

      VectorIDList result;
      scan_range(dir, begin, end, reader, [&](const uint32_t *key_ptr, uint32_t klen, uint32_t id) {
        Attribute key(key_ptr, key_ptr + klen);
        Attribute original = dir == PREFIX ? key : reverse_string_attr(key);
        if (like_match(string_attr_view(original), pat))
          result.push_back(id);
      });

      // Delta scan (bounded).
      delta_attrs_mu_.lock_shared();
      auto d_begin = range_key.empty() ? delta_[PREFIX].begin() : delta_[PREFIX].lower_bound(range_key);
      auto d_end = (range_key.empty() || hi.empty()) ? delta_[PREFIX].end() : delta_[PREFIX].lower_bound(hi);
      for (auto it = d_begin; it != d_end; ++it) {
        if (like_match(string_attr_view(it->first), pat))
          result.insert(result.end(), it->second.begin(), it->second.end());
      }
      delta_attrs_mu_.unlock_shared();

      std::sort(result.begin(), result.end());
      result.erase(std::unique(result.begin(), result.end()), result.end());
      return result;
    }

    VectorIDList prepare_in_filter(const Attribute &, AlignedFileReader *) override {
      return VectorIDList();
    }

    // --- is_member_approx ---

    bool is_member_approx(uint32_t target_id, const Attribute &query, const VectorIDList &) override {
      // No locking here: the caller holds lock_shared() across the traversal.
      return target_id < hashes_.size() && hashes_[target_id] == hash_packed(query);
    }

    bool is_member_approx(uint32_t target_id, const Attribute &query, const VectorIDList &, Direction dir) {
      if (target_id >= bucket_ids_[dir].size() || bucket_boundaries_[dir].empty())
        return true;
      Attribute lo = dir == PREFIX ? query : reverse_string_attr(query);
      Attribute hi = next_string_attr_prefix(lo);
      auto l = std::lower_bound(bucket_boundaries_[dir].begin(), bucket_boundaries_[dir].end(), lo, str_attr_less);
      auto r = hi.empty() ? bucket_boundaries_[dir].end()
                          : std::lower_bound(bucket_boundaries_[dir].begin(), bucket_boundaries_[dir].end(), hi,
                                             str_attr_less);
      int lb = std::max<int>(0, (int) (l - bucket_boundaries_[dir].begin()) - 1);
      int rb = std::min<int>((int) kBuckets - 1, (int) (r - bucket_boundaries_[dir].begin()) + 1);
      uint32_t bid = bucket_ids_[dir][target_id];
      return (int) bid >= lb && (int) bid <= rb;
    }

    // --- Insert / Merge ---

    void insert(uint32_t vector_id, const Attribute &attr) override {
      delta_attrs_mu_.lock();
      delta_[PREFIX][attr].push_back(vector_id);
      delta_[SUFFIX][reverse_string_attr(attr)].push_back(vector_id);
      delta_bytes_ += 2 * sizeof(uint32_t) + attr.size() * sizeof(uint32_t);
      atomic_max(n_vectors, (uint64_t) vector_id + 1);
      max_attrs_ = std::max(max_attrs_, attr.size());
      if (vector_id >= hashes_.size())
        hashes_.resize(std::max<size_t>(1024, (size_t) (1.5 * (vector_id + 1))));
      hashes_[vector_id] = hash_packed(attr);
      ensure_bucket_id(PREFIX, vector_id, attr);
      if (unlikely(delta_bytes_ >= 4 * 1024 * 1024))
        do_merge(libcuckoo::cuckoohash_map<uint32_t, uint32_t>());
      delta_attrs_mu_.unlock();
    }

    void merge(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) override {
      delta_attrs_mu_.lock();
      do_merge(id_map);
      delta_attrs_mu_.unlock();
    }

   private:
    void ensure_bucket_id(Direction dir, uint32_t vector_id, const Attribute &key) {
      if (vector_id >= bucket_ids_[dir].size())
        bucket_ids_[dir].resize(std::max<size_t>(1024, (size_t) (1.5 * (vector_id + 1))));
      if (bucket_boundaries_[dir].empty())
        return;
      auto it = std::upper_bound(bucket_boundaries_[dir].begin(), bucket_boundaries_[dir].end(), key, str_attr_less);
      bucket_ids_[dir][vector_id] = (uint32_t) std::min<size_t>(
          it == bucket_boundaries_[dir].begin() ? 0 : (it - bucket_boundaries_[dir].begin() - 1), kBuckets - 1);
    }

    // Write sorted entries to one direction file.
    void write_sorted_file(Direction dir, std::vector<std::pair<Attribute, uint32_t>> &entries) {
      std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
        if (str_attr_less(a.first, b.first))
          return true;
        if (str_attr_less(b.first, a.first))
          return false;
        return a.second < b.second;
      });

      uint32_t n = entries.size();
      uint32_t ns = (n + kSampleRate - 1) / kSampleRate;  // number of sampled offsets
      uint64_t header_size = 2 * sizeof(uint32_t) + ns * sizeof(uint64_t);

      // Compute byte offsets of each entry.
      std::vector<uint64_t> entry_offsets(n);
      uint64_t cur = header_size;
      for (uint32_t i = 0; i < n; i++) {
        entry_offsets[i] = cur;
        cur += 2 * sizeof(uint32_t) + entries[i].first.size() * sizeof(uint32_t);
      }

      // Build sampled offsets (every kSampleRate-th entry).
      sampled_offsets_[dir].resize(ns);
      for (uint32_t i = 0; i < ns; i++)
        sampled_offsets_[dir][i] = entry_offsets[(uint64_t) i * kSampleRate];

      // Write via fd (so that the fd stays valid for subsequent reads).
      int f = fd_for(dir);
      std::ignore = ftruncate(f, 0);
      lseek(f, 0, SEEK_SET);
      std::ignore = write(f, &n, sizeof(uint32_t));
      std::ignore = write(f, &ns, sizeof(uint32_t));
      if (ns)
        std::ignore = write(f, sampled_offsets_[dir].data(), ns * sizeof(uint64_t));
      for (auto &[key, id] : entries) {
        uint32_t klen = key.size();
        std::ignore = write(f, &id, sizeof(uint32_t));
        std::ignore = write(f, &klen, sizeof(uint32_t));
        if (klen)
          std::ignore = write(f, key.data(), klen * sizeof(uint32_t));
        if (dir == PREFIX)
          max_attrs_ = std::max(max_attrs_, (size_t) klen);
      }
      fsync(f);

      n_entries_[dir] = n;
      file_size_[dir] = cur;

      // Build histogram.
      histogram_[dir].clear();
      histogram_[dir].reserve(ns);
      for (uint32_t i = 0; i < ns; i++)
        histogram_[dir].push_back(entries[(uint64_t) i * kSampleRate].first);
    }

    void rebuild_buckets(Direction dir, const std::vector<std::pair<Attribute, uint32_t>> &entries) {
      bucket_ids_[dir].assign(n_vectors, 0);
      bucket_boundaries_[dir].clear();
      if (entries.empty())
        return;
      for (uint32_t b = 0; b <= kBuckets; b++) {
        size_t idx = std::min<size_t>((uint64_t) b * entries.size() / kBuckets, entries.size() - 1);
        bucket_boundaries_[dir].push_back(entries[idx].first);
      }
      for (size_t i = 0; i < entries.size(); i++) {
        uint32_t bid = (uint32_t) std::min<size_t>((uint64_t) i * kBuckets / entries.size(), kBuckets - 1);
        bucket_ids_[dir][entries[i].second] = bid;
      }
    }

    void save_bucket_approx(Direction dir) {
      std::ofstream writer(approx_filename(dir), std::ios::binary | std::ios::trunc);
      uint32_t nb = bucket_boundaries_[dir].size();
      writer.write((char *) &nb, sizeof(uint32_t));
      for (auto &b : bucket_boundaries_[dir]) {
        uint32_t len = b.size();
        writer.write((char *) &len, sizeof(uint32_t));
        if (len)
          writer.write((char *) b.data(), len * sizeof(uint32_t));
      }
      uint32_t n = bucket_ids_[dir].size();
      writer.write((char *) &n, sizeof(uint32_t));
      if (n)
        writer.write((char *) bucket_ids_[dir].data(), n * sizeof(uint32_t));
      writer.close();
    }

    void load_bucket_approx(Direction dir) {
      std::ifstream reader(approx_filename(dir), std::ios::binary);
      if (!reader.is_open())
        return;
      uint32_t nb = 0;
      reader.read((char *) &nb, sizeof(uint32_t));
      bucket_boundaries_[dir].resize(nb);
      for (uint32_t i = 0; i < nb; i++) {
        uint32_t len = 0;
        reader.read((char *) &len, sizeof(uint32_t));
        bucket_boundaries_[dir][i].resize(len);
        if (len)
          reader.read((char *) bucket_boundaries_[dir][i].data(), len * sizeof(uint32_t));
      }
      uint32_t n = 0;
      reader.read((char *) &n, sizeof(uint32_t));
      bucket_ids_[dir].resize(n);
      if (n)
        reader.read((char *) bucket_ids_[dir].data(), n * sizeof(uint32_t));
    }

    void save_both(std::vector<std::pair<Attribute, uint32_t>> &entries) {
      write_sorted_file(PREFIX, entries);

      std::vector<std::pair<Attribute, uint32_t>> rev;
      rev.reserve(entries.size());
      for (auto &[k, id] : entries)
        rev.push_back({reverse_string_attr(k), id});
      write_sorted_file(SUFFIX, rev);

      // Build hashes and bucket approx.
      hashes_.assign(n_vectors, 0);
      for (auto &[k, id] : entries)
        hashes_[id] = hash_packed(k);
      pipeann::save_bin<uint32_t>(filename + ".filter", hashes_.data(), n_vectors, 1);

      rebuild_buckets(PREFIX, entries);
      rebuild_buckets(SUFFIX, rev);
      for (Direction dir : {PREFIX, SUFFIX})
        save_bucket_approx(dir);

      base_n_vectors_ = n_vectors;
    }

    void do_merge(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &id_map) {
      // Re-read existing entries from prefix file.
      std::vector<std::pair<Attribute, uint32_t>> entries;
      if (n_entries_[PREFIX] > 0) {
        uint64_t data_offset = 2 * sizeof(uint32_t) + sampled_offsets_[PREFIX].size() * sizeof(uint64_t);
        std::ifstream reader(filename, std::ios::binary);
        reader.seekg(data_offset);
        for (uint32_t i = 0; i < n_entries_[PREFIX]; i++) {
          uint32_t id = 0, klen = 0;
          reader.read((char *) &id, sizeof(uint32_t));
          reader.read((char *) &klen, sizeof(uint32_t));
          Attribute key(klen);
          if (klen)
            reader.read((char *) key.data(), klen * sizeof(uint32_t));
          uint32_t nid = 0;
          if (id_map.empty()) {
            entries.push_back({std::move(key), id});
          } else if (id_map.find(id, nid)) {
            entries.push_back({std::move(key), nid});
          }
        }
      }
      // Merge delta.
      for (auto &[key, ids] : delta_[PREFIX]) {
        for (uint32_t id : ids) {
          uint32_t nid = 0;
          if (id_map.empty()) {
            entries.push_back({key, id});
          } else if (id_map.find(id, nid)) {
            entries.push_back({key, nid});
          }
        }
      }
      n_vectors = id_map.empty() ? (uint64_t) n_vectors : id_map.size();
      save_both(entries);
      for (auto &d : delta_)
        d.clear();
      delta_bytes_ = 0;
    }
  };

  // AttrWriter: Serializes per-vector attributes into the on-SSD graph index layout.
  // During index building, each vector's attributes from all registered AttrIndexes
  // are serialized into an Attributes KV-map and written into the vector's on-SSD record.
  // These embedded attributes are used for exact post-filtering verification (is_member).
  struct AttrWriter {
    AttrWriter() = default;
    virtual ~AttrWriter() = default;
    // Serialize attributes of a vector with given ID into the provided buffer according to the on-SSD record layout.
    virtual void write(uint32_t id, void *buffer) = 0;
    // Maximum size of serialized attributes for one vector, used for pre-allocating buffer during index building.
    virtual size_t attr_size() = 0;
  };

  // AttrIndexWriter: Attributes are stored as indexes.
  struct AttrIndexWriter : public AttrWriter {
    std::map<uint32_t, AttrIndex *> attrs_;
    size_t attr_size_ = 0;         // 0 = infer a tight per-row bound (see attr_size()).
    size_t computed_attr_size_ = 0;  // cached result of the inferred bound (0 = not yet computed).

    AttrIndexWriter() = default;
    AttrIndexWriter(const std::map<uint32_t, AttrIndex *> &attrs, size_t attr_size = 0)
        : attrs_(attrs), attr_size_(attr_size) {
    }
    virtual ~AttrIndexWriter() = default;

    virtual void write(uint32_t id, void *buffer) override {
      memset(buffer, 0, attr_size());
      Attributes id_attrs;
      for (auto &[key, attr] : attrs_) {
        id_attrs.set(key, attr->get(id));
      }
      if (unlikely(id_attrs.serialized_size() > attr_size())) {
        LOG(ERROR) << "Attribute size " << attr_size() << " is smaller than serialized attribute size "
                   << id_attrs.serialized_size();
        return;
      }
      id_attrs.serialize((char *) buffer);
    }

    virtual size_t attr_size() override {
      if (attr_size_ != 0) {
        return attr_size_;
      }
      size_t cnt = 1;  // n_keys
      for (auto &[key, attr] : attrs_) {
        cnt += 2 + attr->max_attrs();  // 1 for key, 1 for attr count, max_attrs() for attrs.
      }
      return sizeof(uint32_t) * cnt;
    }
  };

  // AttrVecWriter: Attributes are stored as a vector of Attributes (one per query), directly serialized into the on-SSD
  // record.
  struct AttrVecWriter : AttrWriter {
    const std::vector<Attributes> &attrs_vec;
    size_t attr_size_;

    explicit AttrVecWriter(const std::vector<Attributes> &v, size_t attr_size = 0) : attrs_vec(v) {
      attr_size_ = attr_size;
      if (attr_size_ == 0) {
        attr_size_ = 4;
        for (const auto &attrs : attrs_vec) {
          attr_size_ = std::max(attr_size_, attrs.serialized_size());
        }
      }
    }

    void write(uint32_t id, void *buffer) override {
      memset(buffer, 0, attr_size_);
      if (unlikely(attrs_vec[id].serialized_size() > attr_size_)) {
        LOG(ERROR) << "Attribute size " << attr_size_ << " is smaller than serialized attribute size "
                   << attrs_vec[id].serialized_size();
        return;
      }
      attrs_vec[id].serialize(static_cast<char *>(buffer));
    }

    size_t attr_size() override {
      return attr_size_;
    }
  };

  struct ZeroAttrWriter : AttrWriter {
    size_t attr_size_;
    explicit ZeroAttrWriter(size_t attr_size) : attr_size_(attr_size) {
    }
    void write(uint32_t, void *buffer) override {
      memset(buffer, 0, attr_size_);
    }
    size_t attr_size() override {
      return attr_size_;
    }
  };

  // Helper: check if filename ends with ".spmat"
  inline bool is_suffix(const std::string &filename, const std::string &suffix) {
    if (filename.size() < suffix.size())
      return false;
    return filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

  inline AttrIndex *create_empty_attr_index(const std::string &filename, const std::string &attr_type,
                                            uint64_t n_vectors) {
    if (attr_type == "label") {
      return new InvertedLabelAttrIndex(filename, n_vectors);
    }
    if (attr_type == "range") {
      return new SortedRangeAttrIndex(filename, n_vectors);
    }
    if (attr_type == "string") {
      return new StringAttrIndex(filename, n_vectors);
    }
    throw std::runtime_error("attr_type must be 'label', 'range', or 'string'");
  }

  inline std::string native_attr_type(const AttrIndex *attr_index) {
    if (dynamic_cast<const StringAttrIndex *>(attr_index) != nullptr) {
      return "string";
    }
    if (dynamic_cast<const InvertedLabelAttrIndex *>(attr_index) != nullptr) {
      return "label";
    }
    if (dynamic_cast<const SortedRangeAttrIndex *>(attr_index) != nullptr) {
      return "range";
    }
    throw std::runtime_error("Unknown native attr index type");
  }

  // Field descriptor extracted from a unified schema config.
  // The actual AttrIndex pointer is owned by the caller (stored in base_stores).
  struct FieldDescriptor {
    std::string name;
    uint32_t key = 0;
    std::string type;  // "label" | "range" | "string"
    AttrIndex *attr_index = nullptr;
  };

  // Decode a sparse-matrix file into one Attribute per row, applying the row
  // decoding convention for the given attribute type:
  //   - "label": collect indices[j] where data[j] != 0
  //   - "range": collect data[j] values verbatim (typically [lo, hi))
  // Throws on unsupported type or non-.spmat suffix.
  inline std::vector<Attribute> decode_spmat_rows(const std::string &file, const std::string &type) {
    if (type != "label" && type != "range") {
      throw std::runtime_error("decode_spmat_rows: unsupported type '" + type + "'");
    }
    if (!is_suffix(file, ".spmat")) {
      throw std::runtime_error("decode_spmat_rows: only .spmat files are supported, got: " + file);
    }
    auto [nrow, indptr, indices, data] = load_spmat(file);
    std::vector<Attribute> rows(nrow);
    for (int64_t i = 0; i < nrow; i++) {
      Attribute row_attr;
      int64_t start = indptr[i];
      int64_t end = indptr[i + 1];
      for (int64_t j = start; j < end; j++) {
        if (type == "label") {
          if (data[j] != 0.0f) {
            row_attr.push_back(static_cast<uint32_t>(indices[j]));
          }
        } else {  // range
          row_attr.push_back(static_cast<uint32_t>(data[j]));
        }
      }
      rows[i] = std::move(row_attr);
    }
    return rows;
  }

  // Load query attrs from file into a vector (one Attribute per query)
  inline void load_query_attrs_from_file(const std::string &file, const std::string &type, uint32_t key,
                                         std::vector<Attributes> &query_attrs) {
    if (!query_attrs.empty() && query_attrs[0].find(key)) {
      return;
    }
    auto rows = decode_spmat_rows(file, type);
    int64_t nrow = static_cast<int64_t>(rows.size());
    if (query_attrs.empty()) {
      query_attrs.resize(nrow);
    }
    for (int64_t i = 0; i < nrow; i++) {
      query_attrs[i].set(key, std::move(rows[i]));
    }
    LOG(INFO) << "Loaded query attributes from " << file << " (type: " << type << "): " << nrow << " queries";
  }

  inline AttrIndex *load_attr_index_from_file(const std::string &filename, const std::string &attr_type,
                                              uint64_t n_vectors) {
    auto *attr_index = create_empty_attr_index(filename, attr_type, n_vectors);
    attr_index->load_approx();
    return attr_index;
  }

  inline void save_attr_index_from_rows(const std::vector<Attribute> &rows, const std::string &file_out,
                                        const std::string &attr_type) {
    AttrIndex *attr_index = create_empty_attr_index(file_out, attr_type, rows.size());
    attr_index->load_from_rows(rows);
    attr_index->save();
    delete attr_index;
  }

}  // namespace pipeann

#endif  // ATTRIBUTE_H_
