#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace pipeann {
  // Cache-line alignment for small buffers; whole, aligned 2 MiB regions for THP.
  // Advice is issued before vector initialization touches the allocation.
  template<typename T>
  class AlignedAllocator {
   public:
    using value_type = T;
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    static constexpr size_t CACHE_LINE = 64;
    static constexpr size_t HUGE_PAGE = 2 * 1024 * 1024;

    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

    T *allocate(size_t n) {
      if (n == 0) return nullptr;
      if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
      const size_t bytes = n * sizeof(T);
      const size_t alignment = std::max(alignof(T), bytes >= HUGE_PAGE ? HUGE_PAGE : CACHE_LINE);
      if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) throw std::bad_alloc();
      const size_t allocation_bytes = (bytes + alignment - 1) & ~(alignment - 1);
      void *p = nullptr;
      if (posix_memalign(&p, alignment, allocation_bytes) != 0) throw std::bad_alloc();
#if defined(__linux__) && defined(MADV_HUGEPAGE)
      // Best effort: ordinary pages remain usable when THP is unavailable.
      if (bytes >= HUGE_PAGE) (void) madvise(p, allocation_bytes, MADV_HUGEPAGE);
#endif
      return static_cast<T *>(p);
    }

    void deallocate(T *p, size_t) noexcept { std::free(p); }
  };

  template<typename T, typename U>
  bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) noexcept { return true; }
  template<typename T, typename U>
  bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) noexcept { return false; }
}  // namespace pipeann
