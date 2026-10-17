#pragma once

#include "utils.h"
#include "utils/aligned_allocator.h"
#include "distance.h"
#include <vector>
#include <memory>
#include <fstream>
#include <stdexcept>
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index_defs.h"

namespace pipeann {
  template<typename T>
  class AbstractNeighbor {
   public:
    static constexpr uint32_t MAX_BYTES_PER_NBR = 256;

    pipeann::Metric metric;

    AbstractNeighbor(pipeann::Metric metric) : metric(metric) {
    }

    virtual ~AbstractNeighbor() = default;

    // Representation.
    virtual std::string get_name() const {
      return "AbstractNeighbor";
    }

    virtual uint64_t dim() const {
      return 0;
    }

    // Training and construction.
    // Node payloads and neighbor codes are independent instances of this interface.
    virtual void train(const std::string &) {
      throw std::logic_error("Training unsupported");
    }
    // Input vectors are prepared (including normalization) by the caller.
    // Call load after build to load the neighbors.
    virtual void build(const std::string &index_prefix, const std::string &data_bin, uint32_t bytes_per_nbr) {
    }

    // Persistence.
    // Metadata excludes codes; RAW has no separate metadata files.
    virtual void load_metadata(const char *prefix) {
    }
    virtual void save_metadata(const char *prefix) {
    }

    // Load metadata and records from the same prefix (or the RAW bin filename).
    virtual void load(const char *index_prefix) {
    }

    // Save the neighbor data (e.g., PQ) to disk.
    virtual void save(const char *index_prefix) {
    }

    // Encoding and query preparation.
    // Memory graphs use the stored-record representation on both sides.
    virtual void encode(const T *, void *) {
      throw std::logic_error("Record encoding unsupported");
    }

    virtual void decode(const void *, T *) const {
      throw std::logic_error("Record decoding unsupported");
    }

    // max size of context needed for a single query.
    virtual uint64_t query_ctx_size() {
      return 0;
    }

    virtual void initialize_query(const T *, void *) {
      throw std::logic_error("Query initialization unavailable");
    }

    // Distance computation.
    // A prepared query against stored records addressed by ID.
    virtual void compute_dists(const void *query, const uint32_t *ids, size_t n, float *out) {
      throw std::logic_error("Prepared query scoring unsupported");
    }
    // One stored record against other stored records, all addressed by ID.
    virtual void compute_dists(uint32_t source, const uint32_t *ids, size_t n, float *out) {
      throw std::logic_error("Stored record scoring unsupported");
    }

    virtual void compute_dists_sym(const uint8_t *, const uint32_t *, size_t, float *) {
      throw std::logic_error("Symmetric record scoring unsupported");
    }

    // Record access.
    // Unchecked access to one stored record.
    uint8_t *at(size_t id) {
      return data_.data() + id * data_size;
    }
    size_t size() const {
      return data_size ? data_.size() / data_size : 0;
    }
    uint8_t *data() {
      return data_.data();
    }

    // Caller-side prefetch of the entry compute_dists() will gather for `id`:
    // issued during neighbor collection, the random DRAM access is long resolved
    // by the time the distance kernel runs. Reading data_.data()/data_size while a
    // concurrent insert() resizes `data_` is architecturally harmless (a prefetch of
    // a stale/invalid address just no-ops); the real access still runs under lock.
    inline void prefetch(uint32_t id) const {
      const uint8_t *p = data_.data() + (uint64_t) id * data_size;
      pipeann::cpu_prefetch_t0(p);
      pipeann::cpu_prefetch_t0(p + data_size - 1);  // second line only if the entry straddles
    }

    // Record updates.
    // Allocate addressable slots without adding points.
    void reserve(size_t capacity) {
      if (capacity > size())
        data_.resize(capacity * data_size);
    }

    void resize(size_t count) {
      data_.resize(count * data_size);
      npoints = count;
    }

    virtual void insert(const T *point, uint32_t loc) {
    }

    // rev_id_map: new_id -> old_id.
    virtual AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map,
                                         uint64_t new_npoints, uint32_t nthreads) {
      return this;
    }

    // Release heavy in-memory data (e.g. compressed vectors) after build.
    // The object stays valid; call load() to re-populate from disk.
    virtual void clear() {
      data_.clear();
      data_.shrink_to_fit();
      npoints = 0;
    }

   protected:
    // Contiguous RAW, PQ or RaBitQ records.
    std::vector<uint8_t, AlignedAllocator<uint8_t>> data_;

   public:
    uint64_t data_size = 0;  // Bytes per vector record.
    uint64_t npoints = 0;
  };
}  // namespace pipeann
