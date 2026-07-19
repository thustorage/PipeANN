#pragma once

#include "utils.h"
#include <vector>
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index_defs.h"

namespace pipeann {
  template<typename T>
  class AbstractNeighbor {
   public:
    static constexpr size_t MAX_TRAINING_SET_SIZE = 256000;
    static constexpr double TRAINING_SET_FRACTION = 0.1;
    static constexpr uint32_t MAX_BYTES_PER_NBR = 256;

    pipeann::Metric metric;

    AbstractNeighbor(pipeann::Metric metric) : metric(metric) {
    }

    virtual ~AbstractNeighbor() = default;

    // max size of context needed for a single query.
    virtual uint64_t query_ctx_size() {
      return 0;
    }

    virtual double get_sample_p() {
      if (unlikely(this->npoints == 0)) {
        LOG(ERROR) << "npoints is 0, cannot compute sample p";
        exit(-1);
      }
      auto training_set_size = TRAINING_SET_FRACTION * npoints > MAX_TRAINING_SET_SIZE
                                   ? MAX_TRAINING_SET_SIZE
                                   : (uint32_t) std::round(TRAINING_SET_FRACTION * npoints);
      training_set_size = (training_set_size == 0) ? 1 : training_set_size;
      double p_val = ((double) training_set_size / (double) npoints);
      return p_val;
    }

    virtual std::string get_name() {
      return "AbstractNeighbor";
    }

    // rev_id_map: new_id -> old_id.
    virtual AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map,
                                         uint64_t new_npoints, uint32_t nthreads) {
      return this;
    }

    virtual void initialize_query(const T *query, QueryBuffer *query_buf) {
    }
    
    // Compressed-vector store owned here so the whole hierarchy shares one
    // handle: PQ codes (uint8) or RaBitQ codes (bytes). data_size is the
    // per-entry stride. The concrete handler fills `data` and sets `data_size`
    // in load()/build(); prefetch(id) then warms the entry compute_dists() will
    // gather. Non-virtual and inlined at the call site, so no per-neighbor dispatch.
    std::vector<uint8_t> data;
    uint64_t data_size = 0;

    // Caller-side prefetch of the entry compute_dists() will gather for `id`:
    // issued during neighbor collection, the random DRAM access is long resolved
    // by the time the distance kernel runs. Reading data.data()/data_size while a
    // concurrent insert() resizes `data` is architecturally harmless (a prefetch of
    // a stale/invalid address just no-ops); the real access still runs under lock.
    inline void prefetch(uint32_t id) const {
      const uint8_t *p = data.data() + (uint64_t) id * data_size;
      pipeann::cpu_prefetch_t0(p);
      pipeann::cpu_prefetch_t0(p + data_size - 1);  // second line only if the entry straddles
    }

    // Compute dists using assymetric distance computation.
    virtual void compute_dists(QueryBuffer *query_buf, const uint32_t *ids, const uint64_t n_ids) {
    }

    // Compute dists using PQ all-to-all.
    virtual void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                               uint8_t *aligned_scratch) {
    }

    // Load the neighbor data (e.g., PQ) from disk.
    virtual void load(const char *index_prefix) {
    }

    // Save the neighbor data (e.g., PQ) to disk.
    virtual void save(const char *index_prefix) {
    }

    // Call load after build to load the neighbors.
    virtual void build(const std::string &index_prefix, const std::string &data_bin, uint32_t bytes_per_nbr) {
    }

    virtual void insert(T *point, uint32_t loc) {
    }

    // Release heavy in-memory data (e.g. compressed vectors) after build.
    // The object stays valid; call load() to re-populate from disk.
    virtual void clear() {
    }

    uint64_t npoints = 0;
  };
}  // namespace pipeann
