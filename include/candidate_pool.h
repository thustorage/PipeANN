#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "utils.h"

namespace pipeann {

  // Priority-ordered candidate pool shared by every graph search.
  //
  // Owns the retset array, the per-candidate `visited`/`flag` state, and the
  // markers that walk it. Two frontiers ride the same sorted storage:
  //
  //   * The calc/visited frontier (first_unvisited_): the closest candidate not
  //     yet expanded. Every search uses it. insert() lowers first_unvisited_ to
  //     the smallest insertion position, so the frontier resumes at the closest
  //     unexpanded candidate. The synchronous searches (beam/coro/page and the
  //     in-memory index) drive it directly via next_unvisited()/all_visited(),
  //     marking `visited` at pick time. pipe_search drives it via
  //     next_calc_nbr()/frontier(), gated by `ready_` (page landed), and
  //     terminates on first_unvisited_ >= size.
  //
  //   * The read frontier (send_marker_/`flag`): the closest candidate whose
  //     read has not been issued. Owned by pipe_search alone (next_read_nbr),
  //     because only it issues reads deeply ahead of, and out of order with,
  //     expansion. The synchronous searches never touch `flag`.
  //
  // It does NOT own the caller's dedup hash set; whether a candidate's page has
  // arrived is asked through `ready_`, the one unavoidable seam with the async
  // I/O layer.
  //
  // ReadyFn: (unsigned id) -> bool, "is this id's page already staged/landed".
  template<class ReadyFn = AlwaysTrue>
  class CandidatePool {
   public:
    // Default-constructible so it can live as a struct member (coro); call
    // reset() before use.
    CandidatePool() = default;

    // l_pool: max pool size (candidates beyond this are evicted).
    // capacity: backing storage (headroom for the transient overflow slot and
    //           pool growth; caller sizes it as mem_L + R + l_pool * factor, or
    //           l_pool + 1 for the synchronous searches).
    CandidatePool(size_t l_pool, size_t capacity, ReadyFn ready = ReadyFn()) : ready_(std::move(ready)) {
      reset(l_pool, capacity);
    }

    // (Re)initialize for a fresh query, reusing the backing allocation. Unfilled
    // slots are set to +inf so callers that read a fixed [0, l_pool) window
    // (in-memory index output) see sorted-last placeholders, matching the
    // pre-refactor best_L_nodes behavior.
    void reset(size_t l_pool, size_t capacity) {
      l_pool_ = (unsigned) l_pool;
      cur_list_size_ = 0;
      first_unvisited_ = 0;
      send_marker_ = 0;
      retset_.assign(std::max(capacity, l_pool + 1), Neighbor(0, std::numeric_limits<float>::max()));
    }

    size_t size() const {
      return cur_list_size_;
    }

    bool full() const {
      return cur_list_size_ == l_pool_;
    }

    unsigned first_unvisited() const {
      return first_unvisited_;
    }

    const Neighbor &worst() const {
      return retset_[cur_list_size_ - 1];
    }

    // Read access to the backing array (valid up to l_pool for the fixed-window
    // in-memory output; [0, size()) otherwise).
    const Neighbor &operator[](size_t i) const {
      return retset_[i];
    }
    // Mutable access for callers that read/patch candidates in place (e.g.
    // page_search's page-level dedup).
    Neighbor &operator[](size_t i) {
      return retset_[i];
    }

    const Neighbor *begin() const {
      return retset_.data();
    }

    const Neighbor *end() const {
      return retset_.data() + cur_list_size_;
    }

    // Insert a candidate, keeping the pool sorted and the markers current.
    // Evicts the worst element when full. On rejection returns a sentinel index
    // past the live window: l_pool_ when the candidate is no better than the
    // worst while full, or cur_list_size_ + 1 for a duplicate id. Both are
    // >= any real insertion position, so callers comparing the result against a
    // marker (e.g. r < nk) treat a rejection as "no update".
    unsigned insert(Neighbor nn) {
      // Empty pool: insert_into_pool underflows at K == 0 (right = K - 1), so
      // seed the first slot directly.
      if (cur_list_size_ == 0) {
        retset_[0] = nn;
        cur_list_size_ = 1;
        first_unvisited_ = 0;
        send_marker_ = 0;
        return 0;
      }

      // Early cutoff: full pool and this candidate is no better than the worst.
      if (nn >= retset_[cur_list_size_ - 1] && cur_list_size_ == l_pool_) {
        return l_pool_;
      }

      unsigned r = insert_into_pool(retset_.data(), cur_list_size_, nn);
      if (cur_list_size_ < l_pool_) {
        ++cur_list_size_;
        if (unlikely(cur_list_size_ >= retset_.size())) {
          retset_.resize(2 * cur_list_size_, Neighbor(0, std::numeric_limits<float>::max()));
        }
      }

      if (r < first_unvisited_) {
        first_unvisited_ = r;
      }
      send_marker_ = std::min(send_marker_, r);
      return r;
    }

    // The frontier walks below rest on one invariant: a candidate's `flag` is
    // set only at insert() and cleared the instant its read is issued (async) or
    // it is handed out for expansion (sync). So a set flag implies the read has
    // not gone out yet, hence the page is not staged (!ready_) and the node is
    // not visited. That collapses "the unread frontier" to a single flag test,
    // shared by advance_to_flagged().

    // --- Asynchronous (pipe_search) frontier ---

    // Next candidate to issue a read for, or nullptr. Does not clear the flag:
    // the caller clears it when it issues the read, which advances the marker
    // past it on the next call.
    Neighbor *next_read_nbr() {
      unsigned i = advance_to_flagged();
      return i < cur_list_size_ ? &retset_[i] : nullptr;
    }

    // Select the next unvisited candidate whose page has landed, mark it visited
    // and clear its flag, and return it (or nullptr if none is ready). The
    // returned pointer is valid only until the next insert().
    Neighbor *next_calc_nbr() {
      for (unsigned marker = first_unvisited_; marker < cur_list_size_; ++marker) {
        if (!retset_[marker].visited && ready_(retset_[marker].id)) {
          retset_[marker].flag = false;
          retset_[marker].visited = true;
          return &retset_[marker];
        }
      }
      return nullptr;
    }

    // Advance the calc marker across the settled (visited) prefix, then return
    // the read frontier: the closest candidate still flagged for a read
    // (cur_list_size_ if none). Call after expanding the node returned by
    // next_calc_nbr, so freshly inserted neighbors are accounted for.
    unsigned frontier() {
      while (first_unvisited_ < cur_list_size_ && retset_[first_unvisited_].visited) {
        ++first_unvisited_;
      }
      return advance_to_flagged();
    }

    // --- Synchronous greedy frontier (beam/coro/page/in-memory index) ---
    //
    // These walk the `visited` frontier, not the flag frontier: the synchronous
    // searches don't pipeline reads, so "picked for expansion" and "settled"
    // coincide and a single cursor (first_unvisited_) suffices. `visited` is set
    // at pick time; because that happens before any insert() of the expanded
    // node's neighbors, the memmove inside insert() carries the bit along and no
    // index needs to be re-found. flag is left untouched here — it is the read
    // frontier, owned by the asynchronous (pipe) search alone.

    // Return the closest not-yet-visited candidate, marking it visited, or
    // nullptr when all are visited. Since nothing gates readiness in the
    // synchronous case, the closest unvisited is always at first_unvisited_;
    // insert() lowers first_unvisited_ when a closer candidate appears, so the
    // next call resumes there. Amortized O(1): first_unvisited_ walks the
    // visited prefix once. Extract the id before calling insert() during
    // expansion, as insert() may invalidate the pointer.
    Neighbor *next_unvisited() {
      while (first_unvisited_ < cur_list_size_ && retset_[first_unvisited_].visited) {
        ++first_unvisited_;
      }
      if (first_unvisited_ >= cur_list_size_) {
        return nullptr;
      }
      retset_[first_unvisited_].visited = true;
      return &retset_[first_unvisited_];
    }

    // True once every candidate has been visited. Advances first_unvisited_
    // across the settled prefix (idempotent). The termination check for the
    // synchronous searches.
    bool all_visited() {
      while (first_unvisited_ < cur_list_size_ && retset_[first_unvisited_].visited) {
        ++first_unvisited_;
      }
      return first_unvisited_ >= cur_list_size_;
    }

    // Apply fn to each live candidate [0, size()) in sorted order; fn:
    // (const Neighbor&) -> bool, returning true stops the scan early. Callers
    // accumulate through captures. Two call sites layer their own stop condition:
    //   * termination stops at the first !n.visited -- since first_unvisited_ is
    //     the first unvisited index and [0, first_unvisited_) is entirely
    //     visited, that walks exactly the settled prefix.
    //   * the non-member cand_q drain stops at the first out-of-band distance
    //     (its entries are never visited, so it ignores visited).
    template<class Fn>
    void scan(Fn fn) const {
      for (unsigned i = 0; i < cur_list_size_; ++i) {
        if (fn(retset_[i])) {
          return;
        }
      }
    }

    // Drop the closest n candidates. Used by the non-member cand_q after a drain
    // promotes its in-band prefix into the main pool. cand_q never uses the
    // visited/flag frontiers, so just reset the markers.
    void drop_front(unsigned n) {
      cur_list_size_ -= n;
      memmove(retset_.data(), retset_.data() + n, cur_list_size_ * sizeof(Neighbor));
      first_unvisited_ = 0;
      send_marker_ = 0;
    }

   private:
    // Advance send_marker_ past flag-cleared candidates (reads already issued /
    // nodes already expanded) and return the index of the closest still-flagged
    // candidate, or cur_list_size_ if none remain. The one frontier walk shared
    // by the async read marker and the synchronous k/nk cursor; insert() lowers
    // send_marker_ when a closer candidate appears, so the walk resumes there.
    unsigned advance_to_flagged() {
      while (send_marker_ < cur_list_size_ && !retset_[send_marker_].flag) {
        ++send_marker_;
      }
      return send_marker_;
    }

    // Insert nn into the sorted array addr[0..K); returns the insertion index,
    // or K+1 if a duplicate id is already present. (Formerly the free function
    // InsertIntoPool in utils.h.)
    static unsigned insert_into_pool(Neighbor *addr, unsigned K, Neighbor nn) {
      // find the location to insert
      unsigned left = 0, right = K - 1;
      if (addr[left] > nn) {
        memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
      }
      if (addr[right] < nn) {
        addr[K] = nn;
        return K;
      }
      while (right > 1 && left < right - 1) {
        unsigned mid = (left + right) / 2;
        if (addr[mid] > nn)
          right = mid;
        else
          left = mid;
      }
      // check equal ID
      while (left > 0) {
        if (addr[left] < nn)
          break;
        if (addr[left].id == nn.id)
          return K + 1;
        left--;
      }
      if (addr[left].id == nn.id || addr[right].id == nn.id)
        return K + 1;
      memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
      addr[right] = nn;
      return right;
    }

    std::vector<Neighbor> retset_;
    unsigned cur_list_size_ = 0;
    unsigned l_pool_ = 0;
    // Smallest index with !retset_[i].visited: cheap to maintain, lets calc and
    // termination skip the already-settled prefix (async mode).
    unsigned first_unvisited_ = 0;
    // Smallest index that may still have its flag set: the read marker (async)
    // and the k/nk frontier cursor (synchronous).
    unsigned send_marker_ = 0;
    ReadyFn ready_{};
  };

  // Deduction guides: 2-arg form (synchronous) picks AlwaysTrue; 3-arg form
  // deduces the predicate type.
  CandidatePool(size_t, size_t) -> CandidatePool<AlwaysTrue>;
  template<class ReadyFn>
  CandidatePool(size_t, size_t, ReadyFn) -> CandidatePool<ReadyFn>;

}  // namespace pipeann
