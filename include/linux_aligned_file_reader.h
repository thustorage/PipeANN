#pragma once

#include "aligned_file_reader.h"
#include "utils/lock_table.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t file_sz;
  void *bad_ctx = nullptr;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  void *get_ctx(int flag = 0);

  void register_buf(void *buf, uint64_t buf_size, int mrid);

#if defined(USE_SPDK)
  void *alloc_io_buf(uint64_t size, uint64_t align); // override 
  void free_io_buf(void *ptr, uint64_t size = 0); // override
  void copy_file_to_spdk(int src_fd, uint64_t offset = 0);
#endif

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<IORequest> &read_reqs, void *ctx);
  void write(std::vector<IORequest> &write_reqs, void *ctx);
  void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx);
  void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx);

  // read and update cache.
  void read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref = nullptr);
  int send_read(IORequest &req, void *ctx);
  int send_read(std::vector<IORequest> &reqs, void *ctx);
#if !defined(USE_SPDK) && !defined(USE_URING)
  void send_reads(IORequest **reqs, uint32_t n, void *ctx);
#endif

  void send_io(IORequest &reqs, void *ctx, bool write);
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write);
  int poll(void *ctx);
  void poll_alloc(void *ctx, std::vector<uint64_t> *page_ref);

  // register thread-id for a context
  void register_thread(int flag = 0);

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();
};

namespace pipeann {
  inline std::vector<uint64_t> lockReqs(SparseLockTable<uint64_t> &lock_table, std::vector<IORequest> &reqs) {
    std::vector<uint64_t> ret;
    for (auto &req : reqs) {
      for (uint64_t i = 0; i < req.len; i += SECTOR_LEN) {
        ret.push_back((req.offset + i) / SECTOR_LEN);
      }
    }
    std::sort(ret.begin(), ret.end());
    ret = std::vector<uint64_t>(ret.begin(), std::unique(ret.begin(), ret.end()));
    for (auto &x : ret) {
      lock_table.wrlock(x);
    }
    return ret;
  }

  inline void unlockReqs(SparseLockTable<uint64_t> &lock_table, std::vector<uint64_t> &reqs) {
    for (auto &x : reqs) {
      lock_table.unlock(x);
    }
  }
};  // namespace pipeann
