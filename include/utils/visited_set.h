#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "utils.h"

namespace pipeann {

  // Insert-only id set for per-query visited tracking: open addressing, linear
  // probing, power-of-two capacity. Exists because the visited check runs once
  // per scanned neighbor (~100k/query at large L) and dominates when the probe
  // misses cache: unlike a generic hash set, the probe target address is a pure
  // function of the key, so the search loop can prefetch() a neighbor's slot a
  // few iterations ahead of insert() and hide the miss entirely.
  //
  // The empty slot marker doubles as kInvalidID, which can never be a real
  // vector id; a one-bit side flag keeps the set correct even if a caller
  // inserts it. clear() keeps capacity, so a pooled QueryBuffer reaches a
  // steady state with no rehash inside queries.
  class FlatVisitedSet {
   public:
    explicit FlatVisitedSet(size_t min_capacity = 1 << 13) {
      size_t cap = 8;
      while (cap < min_capacity) {
        cap <<= 1;
      }
      slots_.assign(cap, kEmpty);
      mask_ = cap - 1;
    }

    void clear() {
      if (size_ != 0) {
        std::memset(slots_.data(), 0xFF, slots_.size() * sizeof(uint32_t));
      }
      size_ = 0;
      has_empty_key_ = false;
    }

    // Prefetch the probe line for key, so a later insert(key) hits cache.
    inline void prefetch(uint32_t key) const {
      pipeann::cpu_prefetch_t0((const char *) &slots_[slot_of(key)]);
    }

    // Returns true iff key was newly inserted.
    inline bool insert(uint32_t key) {
      if (unlikely(key == kEmpty)) {
        bool fresh = !has_empty_key_;
        has_empty_key_ = true;
        return fresh;
      }
      size_t i = slot_of(key);
      while (true) {
        const uint32_t v = slots_[i];
        if (v == key) {
          return false;
        }
        if (v == kEmpty) {
          slots_[i] = key;
          // Grow at load factor 1/2 to keep probe chains short.
          if (unlikely(2 * ++size_ > slots_.size())) {
            grow();
          }
          return true;
        }
        i = (i + 1) & mask_;
      }
    }

   private:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;

    inline size_t slot_of(uint32_t key) const {
      // Fibonacci hashing: graph ids arrive clustered, identity would clump
      // linear-probe chains.
      return (key * 0x9E3779B1u) & mask_;
    }

    void grow() {
      std::vector<uint32_t> old = std::move(slots_);
      slots_.assign(old.size() * 2, kEmpty);
      mask_ = slots_.size() - 1;
      for (const uint32_t v : old) {
        if (v == kEmpty) {
          continue;
        }
        size_t i = slot_of(v);
        while (slots_[i] != kEmpty) {
          i = (i + 1) & mask_;
        }
        slots_[i] = v;
      }
    }

    std::vector<uint32_t> slots_;
    size_t mask_ = 0;
    size_t size_ = 0;
    bool has_empty_key_ = false;
  };

}  // namespace pipeann
