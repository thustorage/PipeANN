#ifndef SELECTOR_H_
#define SELECTOR_H_

#include "attribute.h"
#include <stdexcept>

namespace pipeann {

  // Selector: Abstract base class for attribute filtering in filtered ANNS.
  // Defines how to filter vectors based on attribute constraints, supporting
  // speculative pre-filtering, speculative in-filtering, and post-filtering.
  // Composite selectors (AndSelector, OrSelector, NotSelector) combine multiple
  // Selectors via Boolean logic for complex multi-attribute queries.
  struct Selector {
    virtual ~Selector() = default;
    // copy() always returns a deep copy of the selector tree. Callers either
    // borrow an existing selector or take ownership of a copied selector.
    virtual Selector *copy() const {
      throw std::runtime_error("copy() is only implemented for native selectors");
    }

    // Estimate the fraction of dataset vectors satisfying this constraint.
    virtual double estimate_selectivity(const Attributes &query_attrs) = 0;
    // Estimate the precision of is_member_approx: TP / (TP + FP). 1.0 = strict filter.
    virtual double estimate_precision(const Attributes &query_attrs) = 0;
    // Estimate the SSD pages read during speculative pre-filtering.
    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) = 0;

    // Scan on-SSD attribute indexes to return a superset of valid vector IDs.
    // May speculatively skip high-selectivity branches (e.g., in AndSelector),
    // deferring exact verification to is_member() during re-ranking.
    //
    // strict=true: never skip branches, return the precise matching set.
    // Used by filter_only / filter_query (no graph traversal to verify with).
    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader, bool strict = false) = 0;

    // Exact membership check using full attributes stored in the on-SSD record.
    // Used during re-ranking to verify candidates from speculative filtering.
    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) = 0;

    // Estimate the SSD pages read during in-filter preparation (prepare_in_filter).
    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) = 0;

    // Pre-scan rare/cold attribute entries from SSD before graph traversal starts.
    // The prepared state is stored on this selector instance and consumed by
    // is_member_approx() during traversal.
    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) = 0;

    // Fast in-memory approximate membership check during graph traversal.
    // No false negatives: returns false only if the vector is definitely invalid.
    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) = 0;

    // Prefetch the memory that is_member_approx(target_id) will touch (e.g. the
    // per-vector bucket-id / bloom-filter cacheline), so a later call hits cache.
    // The membership check over a node's neighbors is dominated by this random
    // access into a per-vector array sized with the dataset (~1GB for 1e9 pts),
    // so software-prefetching a few neighbors ahead hides the miss latency.
    // Default no-op: selectors whose check is already cache-resident skip it.
    virtual void prefetch_approx(uint32_t /*target_id*/, const Attributes & /*query_attrs*/) {}

    // Hold the underlying attr indexes' shared locks across a whole in-filter
    // traversal, so per-neighbor is_member_approx calls run lock-free (the
    // per-call lock/unlock pair was ~24% of in-filter CPU). Excludes concurrent
    // merge (exclusive lock); a merge now waits for in-flight queries (~ms),
    // same order as before. Call AFTER prepare_in_filter (which takes the same
    // locks internally). attr_indexes_ is populated at construction (see below).
    void lock_shared() {
      for (auto *index : attr_indexes_) {
        index->lock_shared();
      }
    }

    void unlock_shared() {
      for (auto *index : attr_indexes_) {
        index->unlock_shared();
      }
    }

   protected:
    // The AttrIndexes whose in-memory structures is_member_approx reads. Leaf
    // selectors seed this with their own index at construction; composites merge
    // their children's via merge_attr_indexes(). NotSelector reads only its own
    // prepared list, so it contributes nothing. Deduplicated: several leaves may
    // share one AttrIndex, and re-locking a shared_mutex on the same thread is UB.
    std::vector<AttrIndex *> attr_indexes_;

    void merge_attr_indexes(const Selector *child) {
      for (auto *index : child->attr_indexes_) {
        if (std::find(attr_indexes_.begin(), attr_indexes_.end(), index) == attr_indexes_.end()) {
          attr_indexes_.push_back(index);
        }
      }
    }
  };

  struct LabelOrSelector : public Selector {
    uint32_t key_;
    uint32_t base_key_;
    AttrIndex *attr_index_;
    VectorIDList cold_list_;
    LabelOrSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : key_(key), base_key_(base_key), attr_index_(attr_index) {
      attr_indexes_.push_back(attr_index_);
    }

    // Leaf selectors only need a shallow copy because AttrIndex ownership is
    // managed outside the selector tree.
    Selector *copy() const override {
      return new LabelOrSelector(key_, base_key_, attr_index_);
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return (double) attr_index_->estimate_count(query_attrs.get(key_)) / attr_index_->n_vectors;
    }

    virtual double estimate_precision(const Attributes &query_attrs) override {
      return attr_index_->estimate_precision(query_attrs.get(key_));
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_prefilter_reads(query_attrs.get(key_));
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                                    bool /*strict*/ = false) override {
      return attr_index_->pre_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!target_attrs.find(base_key_)) {
        return false;
      }
      Attribute query_attr = query_attrs.get(key_);
      Attribute target_attr = target_attrs.get(base_key_);
      for (auto &label : query_attr) {
        if (std::find(target_attr.begin(), target_attr.end(), label) != target_attr.end()) {
          return true;
        }
      }
      return false;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_infilter_reads(query_attrs.get(key_));
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      cold_list_ = attr_index_->prepare_in_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      return attr_index_->is_member_approx(target_id, query_attrs.get(key_), cold_list_);
    }

    virtual void prefetch_approx(uint32_t target_id, const Attributes &query_attrs) override {
      attr_index_->prefetch_approx(target_id, query_attrs.get(key_), cold_list_);
    }
  };

  struct LabelAndSelector : public Selector {
    uint32_t key_;
    uint32_t base_key_;
    AttrIndex *attr_index_;
    VectorIDList cold_list_;
    LabelAndSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : key_(key), base_key_(base_key), attr_index_(attr_index) {
      attr_indexes_.push_back(attr_index_);
    }

    // Leaf selectors only need a shallow copy because AttrIndex ownership is
    // managed outside the selector tree.
    Selector *copy() const override {
      return new LabelAndSelector(key_, base_key_, attr_index_);
    }

    InvertedLabelAttrIndex *inv_store() {
      return static_cast<InvertedLabelAttrIndex *>(attr_index_);
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return (double) inv_store()->estimate_count_and(query_attrs.get(key_)) / attr_index_->n_vectors;
    }

    virtual double estimate_precision(const Attributes &query_attrs) override {
      return inv_store()->estimate_precision_and(query_attrs.get(key_));
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return inv_store()->estimate_prefilter_reads_and(query_attrs.get(key_));
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                                    bool /*strict*/ = false) override {
      return inv_store()->pre_filter_and(query_attrs.get(key_), reader);
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!target_attrs.find(base_key_)) {
        return false;
      }
      Attribute query_attr = query_attrs.get(key_);
      Attribute target_attr = target_attrs.get(base_key_);
      for (auto &label : query_attr) {
        if (std::find(target_attr.begin(), target_attr.end(), label) == target_attr.end()) {
          return false;
        }
      }
      return true;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return inv_store()->estimate_infilter_reads_and(query_attrs.get(key_));
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      cold_list_ = inv_store()->prepare_in_filter_and(query_attrs.get(key_), reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      return inv_store()->is_member_approx_and(target_id, query_attrs.get(key_), cold_list_);
    }

    virtual void prefetch_approx(uint32_t target_id, const Attributes &query_attrs) override {
      attr_index_->prefetch_approx(target_id, query_attrs.get(key_), cold_list_);
    }
  };

  struct RangeSelector : public Selector {
    uint32_t key_;
    uint32_t base_key_;
    AttrIndex *attr_index_;
    VectorIDList prepared_list_;

    RangeSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : key_(key), base_key_(base_key), attr_index_(attr_index) {
      attr_indexes_.push_back(attr_index_);
    }

    // Leaf selectors only need a shallow copy because AttrIndex ownership is
    // managed outside the selector tree.
    Selector *copy() const override {
      return new RangeSelector(key_, base_key_, attr_index_);
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return (double) attr_index_->estimate_count(query_attrs.get(key_)) / attr_index_->n_vectors;
    }

    virtual double estimate_precision(const Attributes &query_attrs) override {
      return attr_index_->estimate_precision(query_attrs.get(key_));
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_prefilter_reads(query_attrs.get(key_));
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                                    bool /*strict*/ = false) override {
      return attr_index_->pre_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!target_attrs.find(base_key_)) {
        return false;
      }
      Attribute query_attr = query_attrs.get(key_);
      Attribute target_attr = target_attrs.get(base_key_);
      uint32_t l = query_attr[0], r = query_attr.size() > 1 ? query_attr[1] : l + 1;
      return target_attr[0] >= l && target_attr[0] < r;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_infilter_reads(query_attrs.get(key_));
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      prepared_list_ = attr_index_->prepare_in_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      return attr_index_->is_member_approx(target_id, query_attrs.get(key_), prepared_list_);
    }

    virtual void prefetch_approx(uint32_t target_id, const Attributes &query_attrs) override {
      attr_index_->prefetch_approx(target_id, query_attrs.get(key_), prepared_list_);
    }
  };

  struct AndSelector : public Selector {
    std::vector<std::pair<Selector *, double>> selectors_;    // selector + selectivity
    static constexpr double kHighSelectivityThreshold = 0.1;  // Skip branches with selectivity > this in pre_filter.

    AndSelector(std::vector<Selector *> selectors) {
      for (auto &selector : selectors) {
        merge_attr_indexes(selector);
        selectors_.push_back(std::make_pair(selector, 1.0));
      }
    }

    // The parent AndSelector owns and deletes its children, so we deep-copy
    // the child selector tree before handing ownership to the new parent.
    Selector *copy() const override {
      std::vector<Selector *> selectors;
      selectors.reserve(selectors_.size());
      for (const auto &selector : selectors_) {
        selectors.push_back(selector.first->copy());
      }
      return new AndSelector(std::move(selectors));
    }

    ~AndSelector() override {
      for (auto &selector : selectors_) {
        delete selector.first;
      }
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      double selectivity = 1.0;
      for (auto &ss : selectors_) {
        ss.second = ss.first->estimate_selectivity(query_attrs);
        selectivity *= ss.second;
      }
      // from low selectivity to high selectivity.
      std::sort(selectors_.begin(), selectors_.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
      return selectivity;
    }

    // For AND, precision is the product of individual precisions.
    virtual double estimate_precision(const Attributes &query_attrs) override {
      double precision = 1.0;
      for (auto &ss : selectors_) {
        precision *= ss.first->estimate_precision(query_attrs);
      }
      return precision;
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (size_t i = 0; i < selectors_.size() && selectors_[i].second <= kHighSelectivityThreshold; i++) {
        reads += selectors_[i].first->estimate_prefilter_reads(query_attrs);
      }
      return reads;
    }

    // For pre_filter, skip high-selectivity branches to avoid expensive scans.
    // High-selectivity branches will be verified during exact is_member check later.
    // strict=true forces every branch to be intersected (no skipping); used when
    // there's no graph-traversal verification stage to recover precision.
    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                                    bool strict = false) override {
      VectorIDList ret = selectors_[0].first->pre_filter(query_attrs, reader, strict);
      for (size_t i = 1; i < selectors_.size(); i++) {
        if (strict || selectors_[i].second <= kHighSelectivityThreshold) {
          auto cur_set = selectors_[i].first->pre_filter(query_attrs, reader, strict);
          ret = and_sorted_unique(ret, cur_set);
        } else {
          LOG(INFO) << "Skip high-selectivity branch: " << selectors_[i].second;
        }
      }
      return ret;
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      for (auto &ss : selectors_) {
        if (!ss.first->is_member(target_id, query_attrs, target_attrs)) {
          return false;
        }
      }
      return true;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (auto &ss : selectors_) {
        reads += ss.first->estimate_infilter_reads(query_attrs);
      }
      return reads;
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      for (auto &ss : selectors_) {
        ss.first->prepare_in_filter(query_attrs, reader);
      }
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      for (auto &ss : selectors_) {
        if (!ss.first->is_member_approx(target_id, query_attrs)) {
          return false;
        }
      }
      return true;
    }

    virtual void prefetch_approx(uint32_t target_id, const Attributes &query_attrs) override {
      for (auto &ss : selectors_) {
        ss.first->prefetch_approx(target_id, query_attrs);
      }
    }
  };

  struct OrSelector : public Selector {
    std::vector<Selector *> selectors_;
    OrSelector(std::vector<Selector *> selectors) : selectors_(std::move(selectors)) {
      for (auto *selector : selectors_) {
        merge_attr_indexes(selector);
      }
    }

    // The parent OrSelector owns and deletes its children, so we deep-copy
    // the child selector tree before handing ownership to the new parent.
    Selector *copy() const override {
      std::vector<Selector *> selectors;
      selectors.reserve(selectors_.size());
      for (const auto &selector : selectors_) {
        selectors.push_back(selector->copy());
      }
      return new OrSelector(std::move(selectors));
    }

    ~OrSelector() override {
      for (auto *selector : selectors_) {
        delete selector;
      }
    }

    // The selectors are mostly independent. We should use IEP to estimate the selectivity.
    // We simplify IEP by using the union of the selectivities.
    // This is because we only care about low-selectivity cases, where sel1 * sel2 is close to 0.
    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      double selectivity = 0.0;
      for (auto &selector : selectors_) {
        selectivity += selector->estimate_selectivity(query_attrs);
      }
      return selectivity;
    }

    // For OR, precision = union(TP) / union(TP + FP).
    // union(TP) ≈ sum(sel_i * N) (simplified IEP for low selectivity).
    // union(TP + FP) ≈ sum(sel_i * N / p_i).
    virtual double estimate_precision(const Attributes &query_attrs) override {
      double tp_sum = 0.0, total_sum = 0.0;
      for (auto &selector : selectors_) {
        double sel = selector->estimate_selectivity(query_attrs);
        double p = selector->estimate_precision(query_attrs);
        tp_sum += sel;
        total_sum += sel / p;
      }
      return std::min(tp_sum / total_sum, 1.0);
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (auto &selector : selectors_) {
        reads += selector->estimate_prefilter_reads(query_attrs);
      }
      return reads;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (auto &selector : selectors_) {
        reads += selector->estimate_infilter_reads(query_attrs);
      }
      return reads;
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                                    bool strict = false) override {
      VectorIDList vector_id_set;
      for (auto &selector : selectors_) {
        auto ret = selector->pre_filter(query_attrs, reader, strict);
        vector_id_set = or_sorted_unique(vector_id_set, ret);
      }
      return vector_id_set;
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      for (auto &selector : selectors_) {
        if (selector->is_member(target_id, query_attrs, target_attrs)) {
          return true;
        }
      }
      return false;
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      for (auto &selector : selectors_) {
        selector->prepare_in_filter(query_attrs, reader);
      }
    }

    // OR semantics: any branch returning true means the vector is a member.
    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      for (auto &selector : selectors_) {
        if (selector->is_member_approx(target_id, query_attrs)) {
          return true;
        }
      }
      return false;
    }

    virtual void prefetch_approx(uint32_t target_id, const Attributes &query_attrs) override {
      for (auto &selector : selectors_) {
        selector->prefetch_approx(target_id, query_attrs);
      }
    }
  };

  struct NotSelector : public Selector {
    Selector *selector_;
    uint32_t n_vectors_;
    VectorIDList vector_id_set_;
    NotSelector(Selector *selector, uint32_t n_vectors) : selector_(selector), n_vectors_(n_vectors) {
    }

    // Deep-copy the child tree so each query-local selector has its own state.
    Selector *copy() const override {
      return new NotSelector(selector_->copy(), n_vectors_);
    }

    ~NotSelector() override {
      delete selector_;
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return 1.0 - selector_->estimate_selectivity(query_attrs);
    }

    // NOT degrades to pre-filter (exact), so precision is 1.0.
    virtual double estimate_precision(const Attributes &query_attrs) override {
      return 1.0;
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return selector_->estimate_prefilter_reads(query_attrs);
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                                    bool strict = false) override {
      VectorIDList not_vector_ids = selector_->pre_filter(query_attrs, reader, strict);
      VectorIDList vector_ids;
      vector_ids.reserve(n_vectors_ - not_vector_ids.size());

      uint32_t start_id = 0;

      for (uint32_t exclude_id : not_vector_ids) {
        for (uint32_t id = start_id; id < exclude_id; ++id) {
          vector_ids.push_back(id);
        }
        start_id = exclude_id + 1;
      }

      for (uint32_t id = start_id; id < n_vectors_; ++id) {
        vector_ids.push_back(id);
      }
      return vector_ids;
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      return !selector_->is_member(target_id, query_attrs, target_attrs);
    }

    // degrade to pre-filter (as NOT-supersets have false negatives).
    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return this->estimate_prefilter_reads(query_attrs);
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      vector_id_set_ = this->pre_filter(query_attrs, reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &) override {
      return std::binary_search(vector_id_set_.begin(), vector_id_set_.end(), target_id);
    }
  };

  // String-equality selector. Backed by StringAttrIndex.
  //
  // Query Attribute format (multi-record, supports $eq and $in uniformly):
  //   [u32 rec0_len_u32][rec0_bytes (rec0_len_u32 words)]
  //   [u32 rec1_len_u32][rec1_bytes ...]
  //   ...
  // Each record is one alternative string. For $eq there is exactly one record;
  // for $in there are N. Matching is OR across records.
  //
  // Target Attribute format (in the on-disk vector record): bare packed bytes
  // for the single string this vector carries (no length prefix). is_member
  // exact-compares each query record's bytes against the target bytes.
  struct StringEqSelector : public Selector {
    uint32_t key_;
    uint32_t base_key_;
    StringAttrIndex *attr_index_;
    VectorIDList cold_list_;

    StringEqSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : key_(key), base_key_(base_key), attr_index_(dynamic_cast<StringAttrIndex *>(attr_index)) {
      if (attr_index_ == nullptr) {
        throw std::runtime_error("StringEqSelector requires a StringAttrIndex");
      }
      attr_indexes_.push_back(attr_index_);
    }

    Selector *copy() const override {
      return new StringEqSelector(key_, base_key_, attr_index_);
    }

    // Walk multi-record query Attribute, invoking `fn(data_ptr, len_u32)` for
    // each record. Zero-copy: passes a pointer into q's storage directly.
    // Callers that need an Attribute can construct one from the pointer.
    template<typename Fn>
    static void for_each_record(const Attribute &q, Fn fn) {
      size_t pos = 0;
      while (pos < q.size()) {
        uint32_t rec_len = q[pos];
        pos++;
        if (pos + rec_len > q.size())
          return;
        fn(q.data() + pos, rec_len);
        pos += rec_len;
      }
    }

    // Convenience: wrap data pointer into an Attribute for callers that need one.
    static Attribute make_record(const uint32_t *data, uint32_t len) {
      return Attribute(data, data + len);
    }

    double estimate_selectivity(const Attributes &query_attrs) override {
      const Attribute &q = query_attrs.get(key_);
      double total = 0.0;
      for_each_record(q, [&](const uint32_t *data, uint32_t len) {
        auto rec = make_record(data, len);
        total += (double) attr_index_->estimate_count(rec) / attr_index_->n_vectors;
      });
      return std::min(total, 1.0);
    }

    double estimate_precision(const Attributes &query_attrs) override {
      const Attribute &q = query_attrs.get(key_);
      double tp_sum = 0.0, total_sum = 0.0;
      for_each_record(q, [&](const uint32_t *data, uint32_t len) {
        auto rec = make_record(data, len);
        double cnt = attr_index_->estimate_count(rec);
        double p = attr_index_->estimate_precision(rec);
        tp_sum += p * cnt;
        total_sum += cnt;
      });
      return tp_sum / total_sum;
    }

    uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint32_t reads = 0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        reads += attr_index_->estimate_prefilter_reads(make_record(data, len));
      });
      return reads;
    }

    VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                            bool /*strict*/ = false) override {
      VectorIDList result;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        auto sub = attr_index_->pre_filter(make_record(data, len), reader);
        result = or_sorted_unique(result, sub);
      });
      return result;
    }

    bool is_member(uint32_t /*target_id*/, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!target_attrs.find(base_key_))
        return false;
      const Attribute &q = query_attrs.get(key_);
      const Attribute &t = target_attrs.get(base_key_);
      bool match = false;
      for_each_record(q, [&](const uint32_t *data, uint32_t len) {
        if (!match && len == t.size() && memcmp(data, t.data(), len * sizeof(uint32_t)) == 0)
          match = true;
      });
      return match;
    }

    uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      uint32_t reads = 0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        reads += attr_index_->estimate_infilter_reads(make_record(data, len));
      });
      return reads;
    }

    void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      cold_list_.clear();
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        auto sub = attr_index_->prepare_in_filter(make_record(data, len), reader);
        cold_list_ = or_sorted_unique(cold_list_, sub);
      });
    }

    bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      if (std::binary_search(cold_list_.begin(), cold_list_.end(), target_id))
        return true;
      const Attribute &q = query_attrs.get(key_);
      bool any = false;
      for_each_record(q, [&](const uint32_t *data, uint32_t len) {
        if (!any && attr_index_->is_member_approx(target_id, make_record(data, len), VectorIDList()))
          any = true;
      });
      return any;
    }
  };

  struct StringPrefixSelector : public StringEqSelector {
    StringPrefixSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : StringEqSelector(key, base_key, attr_index) {
    }

    Selector *copy() const override {
      return new StringPrefixSelector(key_, base_key_, attr_index_);
    }

    double estimate_selectivity(const Attributes &query_attrs) override {
      double total = 0.0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        total += (double) attr_index_->estimate_count(StringAttrIndex::PREFIX, make_record(data, len)) /
                 attr_index_->n_vectors;
      });
      return std::min(total, 1.0);
    }

    double estimate_precision(const Attributes &query_attrs) override {
      double tp_sum = 0.0, total_sum = 0.0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        auto rec = make_record(data, len);
        double cnt = attr_index_->estimate_count(StringAttrIndex::PREFIX, rec);
        double p = attr_index_->estimate_precision(StringAttrIndex::PREFIX, rec);
        tp_sum += cnt;
        total_sum += p > 0.0 ? cnt / p : cnt;
      });
      return std::min(tp_sum / total_sum, 1.0);
    }

    uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint32_t reads = 0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        reads += attr_index_->estimate_prefilter_reads(StringAttrIndex::PREFIX, make_record(data, len));
      });
      return reads;
    }

    VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader,
                            bool /*strict*/ = false) override {
      VectorIDList result;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        auto sub = attr_index_->pre_filter(StringAttrIndex::PREFIX, make_record(data, len), reader);
        result = or_sorted_unique(result, sub);
      });
      return result;
    }

    bool is_member(uint32_t, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!target_attrs.find(base_key_))
        return false;
      const Attribute &t = target_attrs.get(base_key_);
      bool match = false;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        if (!match)
          match = string_attr_starts_with(t, make_record(data, len));
      });
      return match;
    }

    bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      bool any = false;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        if (!any)
          any = attr_index_->is_member_approx(target_id, make_record(data, len), VectorIDList(),
                                              StringAttrIndex::PREFIX);
      });
      return any;
    }
  };

  struct StringSuffixSelector : public StringEqSelector {
    StringSuffixSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : StringEqSelector(key, base_key, attr_index) {
    }

    Selector *copy() const override {
      return new StringSuffixSelector(key_, base_key_, attr_index_);
    }

    double estimate_selectivity(const Attributes &query_attrs) override {
      double total = 0.0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        total += (double) attr_index_->estimate_count(StringAttrIndex::SUFFIX, make_record(data, len)) /
                 attr_index_->n_vectors;
      });
      return std::min(total, 1.0);
    }

    double estimate_precision(const Attributes &query_attrs) override {
      double tp_sum = 0.0, total_sum = 0.0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        auto rec = make_record(data, len);
        double cnt = attr_index_->estimate_count(StringAttrIndex::SUFFIX, rec);
        double p = attr_index_->estimate_precision(StringAttrIndex::SUFFIX, rec);
        tp_sum += cnt;
        total_sum += p > 0.0 ? cnt / p : cnt;
      });
      return std::min(tp_sum / total_sum, 1.0);
    }

    uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint32_t reads = 0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        reads += attr_index_->estimate_prefilter_reads(StringAttrIndex::SUFFIX, make_record(data, len));
      });
      return reads;
    }

    VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader, bool = false) override {
      VectorIDList result;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        result = or_sorted_unique(result, attr_index_->pre_filter(StringAttrIndex::SUFFIX, make_record(data, len), reader));
      });
      return result;
    }

    bool is_member(uint32_t, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!target_attrs.find(base_key_))
        return false;
      const Attribute &t = target_attrs.get(base_key_);
      bool match = false;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        if (!match)
          match = string_attr_ends_with(t, make_record(data, len));
      });
      return match;
    }

    uint32_t estimate_infilter_reads(const Attributes &) override {
      return 0;
    }

    bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      bool any = false;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        if (!any)
          any = attr_index_->is_member_approx(target_id, make_record(data, len), VectorIDList(),
                                              StringAttrIndex::SUFFIX);
      });
      return any;
    }
  };

  struct StringLikeSelector : public StringEqSelector {
    // Generic LIKE fallback for complex patterns. The query record stores the raw SQL LIKE pattern.
    StringLikeSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : StringEqSelector(key, base_key, attr_index) {
    }

    Selector *copy() const override {
      return new StringLikeSelector(key_, base_key_, attr_index_);
    }

    double estimate_selectivity(const Attributes &query_attrs) override {
      double total = 0.0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        total += (double) attr_index_->estimate_count_like(make_record(data, len)) / attr_index_->n_vectors;
      });
      return std::min(total, 1.0);
    }

    double estimate_precision(const Attributes &) override {
      return 1.0;
    }

    uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        reads += attr_index_->estimate_prefilter_reads_like(make_record(data, len));
      });
      return static_cast<uint32_t>(std::min<uint64_t>(reads, std::numeric_limits<uint32_t>::max() / 2));
    }

    VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader, bool = false) override {
      VectorIDList result;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        auto sub = attr_index_->pre_filter_like(make_record(data, len), reader);
        result = or_sorted_unique(result, sub);
      });
      return result;
    }

    // disable in-filtering.
    uint32_t estimate_infilter_reads(const Attributes &) override {
      return std::numeric_limits<uint32_t>::max() / 2;
    }

    bool is_member_approx(uint32_t, const Attributes &) override {
      return true;
    }

    bool is_member(uint32_t, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!target_attrs.find(base_key_))
        return false;
      std::string_view t = string_attr_view(target_attrs.get(base_key_));
      bool match = false;
      for_each_record(query_attrs.get(key_), [&](const uint32_t *data, uint32_t len) {
        if (!match) {
          Attribute rec(data, data + len);
          match = StringAttrIndex::like_match(t, string_attr_view(rec));
        }
      });
      return match;
    }
  };
}  // namespace pipeann

#endif  // SELECTOR_H_
