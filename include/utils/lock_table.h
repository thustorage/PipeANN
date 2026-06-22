#ifndef LOCK_TABLE_H_
#define LOCK_TABLE_H_
#include <atomic>
#include <chrono>
#include <cstddef>
#include <omp.h>
#include <shared_mutex>
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils/log.h"

// Returns a unique per-thread ID that works both inside OMP parallel regions
// and from plain std::thread / gRPC worker threads.  The ID is assigned on
// first call and stays constant for the thread's lifetime.
inline size_t portable_thread_id() {
  static std::atomic<size_t> next_id{0};
  thread_local size_t my_id = next_id.fetch_add(1);
  return my_id;
}

inline void thread_pause() {
  // Use pause instruction to reduce contention in tight loops.
#ifdef __x86_64__
  asm volatile("pause" ::: "memory");
#endif
}

namespace pipeann {
  template<class K, class HashFunction = std::hash<K>>
  class SparseLockTable {
   public:
    SparseLockTable() {
      locks_ = new libcuckoo::cuckoohash_map<K, std::pair<pthread_rwlock_t *, int>, HashFunction>();
    }

    int tryrdlock(const K &key) {
      int ret = 0;
      locks_->upsert(key, [&](std::pair<pthread_rwlock_t *, int> &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          v = std::make_pair(new pthread_rwlock_t, 0);
          pthread_rwlock_init(v.first, nullptr);
        }
        ret = pthread_rwlock_tryrdlock(v.first);
        if (ret == 0) {
          v.second++;
        }
      });
      return ret;
    }

    int trywrlock(const K &key) {
      int ret = 0;
      locks_->upsert(key, [&](std::pair<pthread_rwlock_t *, int> &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          v = std::make_pair(new pthread_rwlock_t, 0);
          pthread_rwlock_init(v.first, nullptr);
        }
        ret = pthread_rwlock_trywrlock(v.first);
        if (ret == 0) {
          v.second++;
        }
      });
      return ret;
    }

    void rdlock(const K &key) {
      while (tryrdlock(key) != 0) {
        thread_pause();
      }
    }

    void wrlock(const K &key) {
      while (trywrlock(key) != 0) {
        thread_pause();
      }
    }

    inline void unlock(const K &key) {
      locks_->erase_fn(key, [&](std::pair<pthread_rwlock_t *, int> &v) {
        if (v.second == 0) {
          LOG(ERROR) << "SparseLockTable: unlock a non-locked key: " << key;
          __builtin_trap();
        }
        pthread_rwlock_unlock(v.first);

        if (v.second == 1) {
          pthread_rwlock_destroy(v.first);
          delete v.first;
        }
        v.second--;
        return v.second == 0;
      });
    }

    size_t size() {
      return locks_->size();
    }

   private:
    libcuckoo::cuckoohash_map<K, std::pair<pthread_rwlock_t *, int>, HashFunction> *locks_;
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseReadLockGuard {
   public:
    SparseReadLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->rdlock(key_);
    }

    ~SparseReadLockGuard() {
      table_->unlock(key_);
    }

   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseWriteLockGuard {
   public:
    SparseWriteLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->wrlock(key_);
    }

    ~SparseWriteLockGuard() {
      table_->unlock(key_);
    }

   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  class LockTable {
   public:
    LockTable(size_t size) : size_(size) {
      locks_ = new pthread_rwlock_t[size];
      for (size_t i = 0; i < size; i++) {
        pthread_rwlock_init(&locks_[i], nullptr);
      }
    }
    ~LockTable() {
      for (size_t i = 0; i < size_; i++) {
        pthread_rwlock_destroy(&locks_[i]);
      }
      delete[] locks_;
    }

    inline pthread_rwlock_t *rdlock(uint32_t key) {
      auto lock = &locks_[Hash(key) % size_];
      pthread_rwlock_rdlock(lock);
      return lock;
    }

    inline uint64_t pos(uint64_t key) {
      return Hash(key) % size_;
    }

    inline pthread_rwlock_t *wrlock(uint32_t key) {
      auto lock = &locks_[Hash(key) % size_];
      pthread_rwlock_wrlock(lock);
      return lock;
    }

    inline bool tryrdlock(uint32_t key) {
      return (pthread_rwlock_tryrdlock(&locks_[Hash(key) % size_]) == 0);
    }

    inline bool trywrlock(uint32_t key) {
      return (pthread_rwlock_trywrlock(&locks_[Hash(key) % size_]) == 0);
    }

    inline void unlock(pthread_rwlock_t *lock) {
      pthread_rwlock_unlock(lock);
    }

    inline void unlock(uint32_t key) {
      pthread_rwlock_unlock(&locks_[Hash(key) % size_]);
    }

   private:
    size_t size_;
    pthread_rwlock_t *locks_;

    static const uint32_t c1 = 0xcc9e2d51;
    static const uint32_t c2 = 0x1b873593;

    static uint32_t fmix(uint32_t h) {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;
      return h;
    }

    static uint32_t Rotate32(uint32_t val, int shift) {
      // Avoid shifting by 32: doing so yields an undefined result.
      return shift == 0 ? val : ((val >> shift) | (val << (32 - shift)));
    }

    static uint32_t Mur(uint32_t a, uint32_t h) {
      // Helper from Murmur3 for combining two 32-bit values.
      a *= c1;
      a = Rotate32(a, 17);
      a *= c2;
      h ^= a;
      h = Rotate32(h, 19);
      return h * 5 + 0xe6546b64;
    }

    static uint32_t Hash32Len0to4(const char *s, size_t len) {
      uint32_t b = 0;
      uint32_t c = 9;
      for (size_t i = 0; i < len; i++) {
        signed char v = static_cast<signed char>(s[i]);
        b = b * c1 + static_cast<uint32_t>(v);
        c ^= b;
      }
      return fmix(Mur(b, Mur(static_cast<uint32_t>(len), c)));
    }

    static uint32_t Hash(uint32_t x) {
      return Hash32Len0to4((const char *) &x, sizeof(uint32_t));
    }
  };

  // RAII to avoid forgetting to unlock.
  class LockGuard {
   public:
    LockGuard(pthread_rwlock_t *lock) : lock_(lock) {
    }
    LockGuard &operator=(const LockGuard &) = delete;
    LockGuard &operator=(LockGuard &&rhs) {
      lock_ = rhs.lock_;
      rhs.lock_ = nullptr;
      return *this;
    }
    ~LockGuard() {
      if (lock_) {
        pthread_rwlock_unlock(lock_);
        lock_ = nullptr;
      }
    }

   private:
    pthread_rwlock_t *lock_;
  };

  // Used by fixed-size arrays (with resize).
  // Why use this? Resize is very rare, but a single shared_mutex incurs cache ping-pong.
  struct ReaderOptSharedMutex {
    static constexpr int N = 128;
    struct alignas(128) CachelineAlignedMutex {
      std::shared_mutex mutex;
      char padding[128 - sizeof(std::shared_mutex)];
    } locks_[N];
    void lock_shared(size_t idx = portable_thread_id()) {
      locks_[idx % N].mutex.lock_shared();
    }
    void lock() {
      for (size_t i = 0; i < N; ++i) {
        locks_[i].mutex.lock();
      }
    }
    void unlock_shared(size_t idx = portable_thread_id()) {
      locks_[idx % N].mutex.unlock_shared();
    }
    void unlock() {
      for (size_t i = 0; i < N; ++i) {
        locks_[i].mutex.unlock();
      }
    }
  };

}  // namespace pipeann

#endif  // LOCK_TABLE_H_
