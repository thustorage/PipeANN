#pragma once

#include <fcntl.h>
#include <unistd.h>
#include "utils/page_cache.h"
#include "ssd_index_defs.h"

#include <malloc.h>
#include <cstdio>

class AlignedFileReader {
 public:
  int fd = -1;

  // returns the thread-specific io ring.
  // If not constructed, it will register the thread (using the flag) and return the context.
  // For io_uring reader, the flag is used to set up the ring; all algorithms use 0.
  virtual void *get_ctx(int flag = 0) = 0;

  virtual ~AlignedFileReader() {};

  // register buffer (e.g., pin memory)
  virtual void register_buf(void *buf, uint64_t buf_size, int mrid) = 0;

  // Allocate/free DMA-capable I/O buffer. SPDK backend uses hugepage memory.
  virtual void *alloc_io_buf(uint64_t size, uint64_t align) {
    void *ptr = nullptr;
    pipeann::alloc_aligned(&ptr, size, align);
    return ptr;
  }
  
  // the second param (size) unused, but used by SPDK backend to manage free list.
  virtual void free_io_buf(void *ptr, uint64_t size = 0) { 
    std::ignore = size;
    pipeann::aligned_free(ptr); 
  }

  // Open & close ops
  // Blocking calls
  virtual void open(const std::string &fname, bool enable_writes, bool enable_create) = 0;
  virtual void close() = 0;

  // process batch of aligned requests in parallel
  virtual void read(std::vector<IORequest> &read_reqs, void *ctx) = 0;
  virtual void write(std::vector<IORequest> &write_reqs, void *ctx) = 0;
  virtual void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) = 0;
  virtual void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) = 0;

  /*
  virtual int submit_reqs(std::vector<IORequest>& read_reqs, void *ctx) = 0;
  virtual void get_events(void *ctx, int n_ops) = 0;
  */

  virtual void read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref = nullptr) = 0;

  inline void wbc_write(std::vector<IORequest> &write_reqs, void *ctx, std::vector<uint64_t> *page_ref = nullptr) {
    // auto locked_reqs = pipeann::lockReqs(pipeann::cache.lock_table, write_reqs);
    for (auto &req : write_reqs) {
      for (uint64_t i = 0; i < req.len; i += SECTOR_LEN) {
        uint32_t blk_no = (req.offset + i) / SECTOR_LEN;
        pipeann::cache.put(pipeann::PageCacheKey(this->fd, blk_no), (uint8_t *) req.buf + i, true);
      }
    }
    // pipeann::unlockReqs(pipeann::cache.lock_table, locked_reqs);
    if (page_ref != nullptr) {
      for (auto &req : write_reqs) {
        for (uint64_t i = 0; i < req.len; i += SECTOR_LEN) {
          uint32_t blk_no = (req.offset + i) / SECTOR_LEN;
          page_ref->push_back(pipeann::PageCacheKey(this->fd, blk_no).raw);
        }
      }
    }
  }
  inline void deref(std::vector<uint64_t> *page_ref) {
#ifndef READ_ONLY_TESTS
    if (page_ref == nullptr) {
      return;
    }
    for (auto &x : *page_ref) {
      // implicitly convert to CacheKey
      pipeann::cache.deref(pipeann::PageCacheKey::from_raw(x));
    }
#endif
  }

  virtual void send_io(IORequest &reqs, void *ctx, bool write) = 0;
  virtual void send_io(std::vector<IORequest> &reqs, void *ctx, bool write) = 0;
  virtual int send_read(IORequest &req, void *ctx) = 0;
  virtual int send_read(std::vector<IORequest> &reqs, void *ctx) = 0;
  // Send reqs[0..n) together, each served from the page cache when it holds the pages; an engine
  // that cannot batch sends them one by one.
  virtual void send_reads(IORequest **reqs, uint32_t n, void *ctx) {
    for (uint32_t i = 0; i < n; i++) {
      send_read(*reqs[i], ctx);
    }
  }
  virtual int poll(void *ctx) = 0;
  virtual void poll_alloc(void *ctx, std::vector<uint64_t> *page_ref) = 0;

  // register thread-id for a context
  virtual void register_thread(int flag = 0) = 0;
  // de-register thread-id for a context
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;
};
