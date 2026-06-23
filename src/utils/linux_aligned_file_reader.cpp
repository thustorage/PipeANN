#include <bits/types/struct_iovec.h>
#include "ssd_index_defs.h"
#include "utils/lock_table.h"
#if defined(USE_SPDK)
#include "linux_aligned_file_reader.h"
#include "utils.h"
#include "utils/log.h"
#include "utils/picojson.h"
#include "utils/concurrent_queue.h"
#include "spdk/env.h"
#include "spdk/nvme.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <chrono>
#include <pthread.h>
#include "liburing.h"

namespace {
  static constexpr uint32_t STRIPE_SIZE = SECTOR_LEN;
  static constexpr int MAX_WORKER_THREADS = 256;
  static constexpr int POLLER_QUEUE_CAP = 4096;
  static constexpr size_t MARKER_LEN = 1024;

  union IOCallbackArg {
    void *raw;
    struct {
      uint64_t req : 56;
      uint64_t tid : 8;  // maximum of 256 workers, equal to MAX_WORKER_THREADS.
    };
  };

  struct Poller;
  static struct SpdkState {
    struct NamespaceEntry {
      spdk_nvme_ctrlr *ctrlr = nullptr;
      spdk_nvme_ns *ns = nullptr;
    };
    std::vector<NamespaceEntry> nss;
    std::vector<Poller *> pollers;
    std::vector<pipeann::MPSCQueue<IORequest *> *> cqs;
    uint32_t lba_shift{0};  // log2(lba_size)
    std::vector<std::string> ssds{};
    std::string hugedir_str{};
    int n_attached{0};
    bool initialized = false;
    bool file_opened = false;
    spdk_env_opts env_opts{};
    spdk_nvme_transport_id trid{};

    // spdk_free and spdk_zmalloc are so slow.
    // When free_io_buf, we do not directly free them but add them to free list.
    std::unordered_map<size_t, std::vector<void *>> free_list;
    std::mutex alloc_mu;
  } g_spdk;

  // One poller per SSD.
  struct Poller {
    struct Cmd {
      void *buf;
      uint64_t lba;
      uint32_t lba_count;
      bool write;
      IORequest *parent;
    };

    int ns_id;
    spdk_nvme_ns *ns;
    spdk_nvme_qpair *qp;
    pipeann::MPSCQueue<Cmd> *cmd_queue;
    std::thread thread;
    std::atomic<bool> running{true};
    static thread_local Poller *tls_poller;  // for io_cb to access the right poller instance.

    Poller(int ns_id, spdk_nvme_ns *ns, spdk_nvme_qpair *qp) : ns_id(ns_id), ns(ns), qp(qp) {
      cmd_queue = new pipeann::MPSCQueue<Cmd>(MAX_WORKER_THREADS, POLLER_QUEUE_CAP);
      this->thread = std::thread(&Poller::poller_fn, this);
    }

    static void io_cb(void *arg, const spdk_nvme_cpl *cpl) {
      if (unlikely(spdk_nvme_cpl_is_error(cpl))) {
        LOG(ERROR) << "SPDK I/O error status=" << cpl->status.sc << " sct=" << cpl->status.sct;
      }

      IOCallbackArg ctx{.raw = arg};
      g_spdk.cqs[ctx.tid]->push(reinterpret_cast<IORequest *>(ctx.req), tls_poller->ns_id);
    }

    void poller_fn() {
      tls_poller = this;

      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(this->ns_id, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

      while (this->running.load(std::memory_order_relaxed)) {
        spdk_nvme_qpair_process_completions(this->qp, 0);
        this->cmd_queue->pop_all_fn([this](const Poller::Cmd &cmd, uint8_t producer_id) {
          int rc = 0;
          IOCallbackArg ctx{.req = (uint64_t) cmd.parent, .tid = producer_id};
          do {
            rc = cmd.write
                     ? spdk_nvme_ns_cmd_write(this->ns, this->qp, cmd.buf, cmd.lba, cmd.lba_count, io_cb, ctx.raw, 0)
                     : spdk_nvme_ns_cmd_read(this->ns, this->qp, cmd.buf, cmd.lba, cmd.lba_count, io_cb, ctx.raw, 0);

            if (unlikely(rc == -EAGAIN)) {
              spdk_nvme_qpair_process_completions(this->qp, 0);
            }
          } while (unlikely(rc == -EAGAIN));

          if (unlikely(rc != 0)) {
            LOG(ERROR) << "SPDK submit failed rc=" << rc << " ns=" << this->ns_id;
          }
        });
      }
    }
  };
  thread_local Poller *Poller::tls_poller = nullptr;

  static thread_local int tls_thread_id = -1;

  void load_json_config(SpdkState &st) {
    const std::string path = "./spdk_bdevs.json";
    std::ifstream in(path);
    if (!in.good()) {
      LOG(INFO) << "SPDK config " << path << " not found";
      return;
    }

    picojson::value root_v;
    std::string err = picojson::parse(root_v, in);
    if (!err.empty()) {
      LOG(ERROR) << "SPDK config parse error: " << err;
      crash();
    }
    const auto &root = root_v.get<picojson::object>();

    for (const auto &v : root.find("ssds")->second.get<picojson::array>()) {
      st.ssds.push_back(v.get<std::string>());
    }
    st.hugedir_str = root.find("hugedir")->second.get<std::string>();
    st.env_opts.hugedir = st.hugedir_str.c_str();
    st.env_opts.hugepage_single_segments = true;
  }

  std::string marker_for(const std::string &path) {
    std::string m = path + "|ssds=";
    for (auto &s : g_spdk.ssds) {
      m += s + ",";
    }
    m += "|stripe=" + std::to_string(STRIPE_SIZE);
    return m.size() <= MARKER_LEN ? m : m.substr(m.size() - MARKER_LEN);
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  g_spdk.env_opts.opts_size = sizeof(g_spdk.env_opts);
  spdk_env_opts_init(&g_spdk.env_opts);
  load_json_config(g_spdk);
  if (spdk_env_init(&g_spdk.env_opts) < 0) {
    LOG(ERROR) << "spdk_env_init failed";
    crash();
  }
  g_spdk.nss.resize(g_spdk.ssds.size());
  spdk_nvme_trid_populate_transport(&g_spdk.trid, SPDK_NVME_TRANSPORT_PCIE);
  std::snprintf(g_spdk.trid.subnqn, sizeof(g_spdk.trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);
  spdk_nvme_probe(
      &g_spdk.trid, &g_spdk,
      [](void *cb_ctx, const spdk_nvme_transport_id *trid, spdk_nvme_ctrlr_opts *opts) {
        auto *st = static_cast<SpdkState *>(cb_ctx);
        std::string addr = trid->traddr;
        if (std::find(st->ssds.begin(), st->ssds.end(), addr) == st->ssds.end()) {
          return false;
        }
        opts->num_io_queues = std::max<uint32_t>(opts->num_io_queues, 1);
        opts->io_queue_size = UINT16_MAX;
        opts->io_queue_requests = UINT16_MAX;
        LOG(INFO) << "SPDK attaching to " << addr;
        return true;
      },
      [](void *cb_ctx, const spdk_nvme_transport_id *trid, spdk_nvme_ctrlr *ctrlr, const spdk_nvme_ctrlr_opts *opts) {
        auto &st = *static_cast<SpdkState *>(cb_ctx);
        std::string target = trid->traddr;
        auto it = std::find(st.ssds.begin(), st.ssds.end(), target);
        if (it == st.ssds.end())
          return;
        st.n_attached++;
        size_t pos = it - st.ssds.begin();
        for (int nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
             nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
          auto *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
          if (!ns || !spdk_nvme_ns_is_active(ns)) {
            continue;
          }
          st.nss[pos].ctrlr = ctrlr;
          st.nss[pos].ns = ns;
          LOG(INFO) << "SPDK namespace " << spdk_nvme_ns_get_id(ns) << " size " << (spdk_nvme_ns_get_size(ns) / 1e9)
                    << " GB (SSD " << pos << ")";

          const struct spdk_nvme_ns_data *nsdata = spdk_nvme_ns_get_data(ns);
          uint32_t lba_shift = nsdata->lbaf[nsdata->flbas.format].lbads;
          if (st.lba_shift != 0 && st.lba_shift != lba_shift) {
            LOG(ERROR) << "SPDK attached SSDs have different LBA sizes";
            crash();
          }
          st.lba_shift = lba_shift;
          break;
        }
      },
      nullptr);

  g_spdk.pollers.resize(g_spdk.ssds.size());
  for (size_t i = 0; i < g_spdk.nss.size(); ++i) {
    auto &ns = g_spdk.nss[i];
    if (!ns.ctrlr || !ns.ns) {
      LOG(ERROR) << "Requested SPDK SSD was not attached";
      crash();
    }
    struct spdk_nvme_io_qpair_opts opts;
    spdk_nvme_ctrlr_get_default_io_qpair_opts(ns.ctrlr, &opts, sizeof(opts));
    opts.delay_cmd_submit = false;
    auto *qp = spdk_nvme_ctrlr_alloc_io_qpair(ns.ctrlr, &opts, sizeof(opts));
    if (!qp) {
      LOG(ERROR) << "spdk_nvme_ctrlr_alloc_io_qpair failed for SSD " << i;
      crash();
    }

    auto poller = new Poller(i, ns.ns, qp);
    g_spdk.pollers[i] = poller;
  }

  for (size_t i = 0; i < MAX_WORKER_THREADS; i++) {
    g_spdk.cqs.push_back(new pipeann::MPSCQueue<IORequest *>(g_spdk.ssds.size(), POLLER_QUEUE_CAP));
  }

  LOG(INFO) << "SPDK initialized: " << g_spdk.ssds.size() << " pollers, LBA size " << (1ul << g_spdk.lba_shift)
            << " bytes";
  g_spdk.initialized = true;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  if (!g_spdk.initialized) {
    return;
  }
  for (auto *poller : g_spdk.pollers) {
    poller->running = false;
    poller->thread.join();
    spdk_nvme_ctrlr_free_io_qpair(poller->qp);
    delete poller->cmd_queue;
    delete poller;
  }
  for (auto &ns : g_spdk.nss) {
    if (ns.ctrlr) {
      spdk_nvme_detach_ctx *ctx = nullptr;
      spdk_nvme_detach_async(ns.ctrlr, &ctx);
      if (ctx) {
        spdk_nvme_detach_poll(ctx);
      }
    }
  }
  for (auto *cq : g_spdk.cqs) {
    delete cq;
  }
  g_spdk.cqs.clear();

  for (auto &kv : g_spdk.free_list) {
    for (auto &ptr : kv.second) {
      spdk_free(ptr);
    }
  }
  g_spdk.initialized = false;
}

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(tls_thread_id < 0)) {
    register_thread(flag);
  }
  return (void *) (intptr_t) tls_thread_id;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  static std::atomic<int> nxt_thread_id{0};
  if (tls_thread_id < 0) {
    int id = nxt_thread_id++;
    if (id >= MAX_WORKER_THREADS) {
      LOG(ERROR) << "SPDK max_worker_threads=" << MAX_WORKER_THREADS << " exceeded";
      crash();
    }
    tls_thread_id = id;

    // bind to #poller + thread_id to maximize core locality.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int cpu = static_cast<int>(g_spdk.ssds.size()) + id;
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  tls_thread_id = -1;
}

void LinuxAlignedFileReader::deregister_all_threads() {
  return;
}

void LinuxAlignedFileReader::register_buf(void *buf, uint64_t size, int mrid) {
  return;
}

void *LinuxAlignedFileReader::alloc_io_buf(uint64_t size, uint64_t align) {
  std::lock_guard<std::mutex> lock(g_spdk.alloc_mu);
  if (g_spdk.free_list.find(size) != g_spdk.free_list.end() && !g_spdk.free_list[size].empty()) {
    void *ptr = g_spdk.free_list[size].back();
    g_spdk.free_list[size].pop_back();
    return ptr;
  }
  return spdk_dma_zmalloc(size, align, nullptr);
}

void LinuxAlignedFileReader::free_io_buf(void *ptr, uint64_t size) {
  std::lock_guard<std::mutex> lock(g_spdk.alloc_mu);
  g_spdk.free_list[size].push_back(ptr);
  // spdk_free(ptr);
}

// not close src_fd. Note that the tail is SECTOR_LEN padded.
void LinuxAlignedFileReader::copy_file_to_spdk(int src_fd, uint64_t offset) {
  static constexpr uint32_t COPY_IO_SIZE = 8 * 1024 * 1024;

  auto *ctx = this->get_ctx();
  char *buf = (char *) spdk_dma_zmalloc(COPY_IO_SIZE, SECTOR_LEN, nullptr);
  uint64_t total = lseek(src_fd, 0, SEEK_END);

  LOG(INFO) << "Copying file to SPDK RAID0 target: size=" << (total >> 20) << " MiB" << " offset = " << (offset >> 20) << " MiB";
  auto start = std::chrono::steady_clock::now();
  uint64_t copied_bytes = 0, last_log_bytes = 0;

  while (copied_bytes < total) {
    uint64_t len = std::min<uint64_t>(COPY_IO_SIZE, total - copied_bytes);
    ssize_t got = ::pread(src_fd, buf, len, copied_bytes);
    std::vector<IORequest> copy_req{IORequest(offset + copied_bytes, ROUND_UP(len, SECTOR_LEN), buf, 0, len)};
    write(copy_req, ctx);
    copied_bytes += len;

    if (copied_bytes - last_log_bytes >= (10ULL << 30) || copied_bytes == total) {
      auto now = std::chrono::steady_clock::now();
      double sec = std::chrono::duration<double>(now - start).count();
      LOG(INFO) << "SPDK copied " << (copied_bytes >> 20) << "/" << (total >> 20) << " MiB, "
                << (copied_bytes / 1e9 / std::max(sec, 1e-9)) << " GB/s";
      last_log_bytes = copied_bytes;
    }
  }
  spdk_free(buf);
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes, bool enable_create) {
  // SPDK backend supports open only once; close and re-open not supported.
  // However, it ensures correctness:
  // Only index reload re-opens the reader.
  // We take a tricky approach: during merge, we directly overwrite the original index (in wfd)
  // instead of using two files in io_uring and libaio.
  // Thus, index close-to-open does not need to re-copy the file; it does nothing.
  // The on-SSD index metadata may be incorrect, but it's OK as we do not reload it.
  if (g_spdk.file_opened) {
    LOG(INFO) << "Safely ignoring the open of " << fname;
    return;
  }
  auto *ctx = this->get_ctx();
  char *hdr = (char *) spdk_dma_zmalloc(SECTOR_LEN, SECTOR_LEN, nullptr);
  std::vector<IORequest> hdr_req{IORequest(0, SECTOR_LEN, hdr, 0, SECTOR_LEN)};
  read(hdr_req, ctx);
  std::string marker = marker_for(fname);
  char *stored = hdr + SECTOR_LEN - MARKER_LEN;
  // for indexes supporting writes, force copy to SSD,
  // as the on-SSD version may have been updated by previous benchmarks.
  if (!enable_writes && std::strncmp(stored, marker.c_str(), MARKER_LEN) == 0) {
    LOG(INFO) << "SPDK index already copied for " << marker;
    spdk_free(hdr);
    return;
  }
  int src = ::open(fname.c_str(), O_DIRECT | O_LARGEFILE | O_RDONLY, 0644);

  copy_file_to_spdk(src, 0);

  // Commit marker.
  ssize_t got = ::pread(src, hdr, SECTOR_LEN, 0);
  memset(hdr + SECTOR_LEN - MARKER_LEN, 0, MARKER_LEN);
  memcpy(hdr + SECTOR_LEN - MARKER_LEN, marker.data(), marker.size());
  std::vector<IORequest> marker_req{IORequest(0, SECTOR_LEN, hdr, 0, SECTOR_LEN)};
  write(marker_req, ctx);

  spdk_free(hdr);
  LOG(INFO) << "SPDK copy complete.";
  g_spdk.file_opened = true;
  this->fd = src; // not used, kept only for page cache.
}

void LinuxAlignedFileReader::close() {
  // Do nothing, do not set file_opened to false.
  return;
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool wr) {
  int tid = (intptr_t) ctx;
  req.n_pending = 0;
  req.finished = false;

  const uint32_t lba_shift = g_spdk.lba_shift;

  for (uint64_t done = 0; done < req.len;) {
    void *buf = (char *) req.buf + done;
    uint64_t off = req.offset + done;
    uint64_t stripe_off = off % STRIPE_SIZE;
    uint64_t chunk_size = std::min(req.len - done, STRIPE_SIZE - stripe_off);
    uint64_t stripe_no = off / STRIPE_SIZE;
    int ns_id = stripe_no % g_spdk.ssds.size();
    uint64_t ns_off = (stripe_no / g_spdk.ssds.size()) * STRIPE_SIZE + stripe_off;

    Poller::Cmd cmd{buf, ns_off >> lba_shift, (uint32_t) (chunk_size >> lba_shift), wr, &req};
    while (unlikely(!g_spdk.pollers[ns_id]->cmd_queue->push(cmd, tid))) {
      pipeann::cpu_pause();
    }

    req.n_pending++;
    done += chunk_size;
  }
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool wr) {
  for (auto &req : reqs) {
    send_io(req, ctx, wr);
  }
}

int LinuxAlignedFileReader::poll(void *ctx) {
  int tid = (intptr_t) ctx;
  IORequest *req;
  int n = 0;
  while (g_spdk.cqs[tid]->pop(req)) {
    if (--req->n_pending == 0) {
      req->finished = true;
      n++;
    }
  }
  return n;
}

void LinuxAlignedFileReader::poll_alloc(void *ctx, std::vector<uint64_t> *page_ref) {
  int tid = (intptr_t) ctx;
  IORequest *req;
  while (g_spdk.cqs[tid]->pop(req)) {
    if (--req->n_pending == 0) {
      req->finished = true;
      for (uint64_t i = 0; i < req->len / SECTOR_LEN; i++) {
        uint32_t blk_no = req->offset / SECTOR_LEN + i;
        pipeann::cache.put(pipeann::PageCacheKey(fd, blk_no), (uint8_t *) req->buf + i * SECTOR_LEN, true);
        page_ref->push_back(pipeann::PageCacheKey(fd, blk_no).raw);
      }
    }
  }
}

static void blocking(std::vector<IORequest> &reqs, void *ctx, bool wr, LinuxAlignedFileReader *r) {
  r->send_io(reqs, ctx, wr);
  for (auto &req : reqs) {
    while (!req.finished) {
      r->poll(ctx);
    }
  }
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &reqs, void *ctx) {
  blocking(reqs, ctx, false, this);
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &reqs, void *ctx) {
  blocking(reqs, ctx, true, this);
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &reqs, void *ctx) {
  // Used by attribute reading.
  // Use a simple approach: directly read the file using io_uring.
  // TODO: change to SPDK by allocating attribute indexes at the tail.
  // Need to manage buffers allocated in attribute.h before read_fd.
  static __thread io_uring *ring = nullptr;
  if (unlikely(ring == nullptr)) {
    ring = new io_uring();
    int ret = io_uring_queue_init(MAX_N_SECTOR_READS, ring, 0);
    if (ret != 0) {
      LOG(ERROR) << "io_uring_queue_init failed: " << strerror(-ret);
      crash();
    }
  }

  while (true) {
    for (uint64_t j = 0; j < reqs.size(); j++) {
      auto sqe = io_uring_get_sqe(ring);
      io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
    io_uring_submit(ring);

    io_uring_cqe *cqe = nullptr;
    bool fail = false;
    for (uint64_t j = 0; j < reqs.size(); j++) {
      int ret = 0;
      do {
        ret = io_uring_wait_cqe(ring, &cqe);
      } while (ret == -EINTR);

      if (ret < 0 || cqe->res < 0) {
        fail = true;
        LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << j << " " << reqs[j].buf << " "
                    << reqs[j].len << " " << reqs[j].offset;
        break;  // CQE broken.
      }
      io_uring_cqe_seen(ring, cqe);
    }
    if (!fail) {  // repeat until no fails.
      break;
    }
  }
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &reqs, void *ctx) {
  // Only used by delete_merge.
  // We update the index file in-place, instead of using the separate file.
  return write(reqs, ctx);
}

#elif defined(USE_URING)
#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "liburing.h"

#define MAX_EVENTS 256

namespace {
  constexpr uint64_t kNoUserData = 0;
  void execute_io(void *context, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    io_uring *ring = (io_uring *) context;
    while (true) {
      for (uint64_t j = 0; j < reqs.size(); j++) {
        auto sqe = io_uring_get_sqe(ring);
        sqe->user_data = kNoUserData;
        if (write) {
          io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        } else {
          io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        }
      }
      io_uring_submit(ring);

      io_uring_cqe *cqe = nullptr;
      bool fail = false;
      for (uint64_t j = 0; j < reqs.size(); j++) {
        int ret = 0;
        do {
          ret = io_uring_wait_cqe(ring, &cqe);
        } while (ret == -EINTR);

        if (ret < 0 || cqe->res < 0) {
          fail = true;
          LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << j << " " << reqs[j].buf << " "
                     << reqs[j].len << " " << reqs[j].offset;
          break;  // CQE broken.
        }
        io_uring_cqe_seen(ring, cqe);
      }
      if (!fail) {  // repeat until no fails.
        break;
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure fd is closed
  ret = ::fcntl(this->fd, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->fd);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_uring *ring = nullptr;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ring == nullptr)) {
    register_thread(flag);
  }
  return ioctx::ring;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ring == nullptr) {
    ioctx::ring = new io_uring();
    int ret = io_uring_queue_init(MAX_EVENTS, ioctx::ring, flag);
    if (ret != 0) {
      LOG(ERROR) << "io_uring_queue_init failed: " << strerror(-ret);
      crash();
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  io_uring_queue_exit(ioctx::ring);
  delete ioctx::ring;
  ioctx::ring = nullptr;
}

void LinuxAlignedFileReader::deregister_all_threads() {
  return;
}

void LinuxAlignedFileReader::register_buf(void *buf, uint64_t buf_size, int mrid) {
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->fd = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->fd != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure fd is closed
  ::fcntl(this->fd, F_GETFD);
  //  assert(ret != -1);

  ::close(this->fd);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, this->fd, read_reqs);
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, this->fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  auto sqe = io_uring_get_sqe(ring);
  req.finished = false;
  sqe->user_data = (uint64_t) &req;
  if (write) {
    io_uring_prep_write(sqe, fd, req.buf, req.len, req.offset);
  } else {
    io_uring_prep_read(sqe, fd, req.buf, req.len, req.offset);
  }
  io_uring_submit(ring);
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  for (uint64_t j = 0; j < reqs.size(); j++) {
    auto sqe = io_uring_get_sqe(ring);
    reqs[j].finished = false;
    sqe->user_data = (uint64_t) &reqs[j];
    if (write) {
      io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
  }
  io_uring_submit(ring);
}

int LinuxAlignedFileReader::poll(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  static __thread io_uring_cqe *cqes[MAX_EVENTS];
  int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
  if (ret < 0) {
    return 0;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    if (cqes[i]->res < 0) {
      LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
    }
    IORequest *req = (IORequest *) cqes[i]->user_data;
    if (req != nullptr) {
      req->finished = true;
    }
    io_uring_cqe_seen(ring, cqes[i]);
  }
  return ret;
}

void LinuxAlignedFileReader::poll_alloc(void *ctx, std::vector<uint64_t> *page_ref) {
  io_uring *ring = (io_uring *) ctx;
  static __thread io_uring_cqe *cqes[MAX_EVENTS];
  int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
  if (ret < 0) {
    return;
  }
  for (int i = 0; i < ret; i++) {
    if (cqes[i]->res < 0) {
      LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
    }
    IORequest *req = (IORequest *) cqes[i]->user_data;
    if (req != nullptr) {
      req->finished = true;
      for (uint64_t i = 0; i < req->len / SECTOR_LEN; i++) {
        uint32_t blk_no = req->offset / SECTOR_LEN + i;
        pipeann::cache.put(pipeann::PageCacheKey(fd, blk_no), (uint8_t *) req->buf + i * SECTOR_LEN, true);
        page_ref->push_back(pipeann::PageCacheKey(fd, blk_no).raw);
      }
    }
    io_uring_cqe_seen(ring, cqes[i]);
  }
}

#else
#include "linux_aligned_file_reader.h"

#include <libaio.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 256

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb iocb_t;

  void execute_io(void *ctx, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops = std::min((uint64_t) reqs.size() - (iter * MAX_EVENTS), (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *> cbs(n_ops, nullptr);
      std::vector<io_event_t> evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        if (write) {
          io_prep_pwrite(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                         reqs[j + iter * MAX_EVENTS].offset);
        } else {
          io_prep_pread(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                        reqs[j + iter * MAX_EVENTS].offset);
        }
      }

      // initialize `cbs` using `cb` array
      //

      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno << "="
                     << ::strerror((int) -ret) << ", try #" << n_tries + 1 << " ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents((io_context_t) ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(), nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            LOG(ERROR) << "io_getevents() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                       << "=" << ::strerror((int) -ret) << ", try #" << n_tries + 1;
            exit(-1);
          } else {
            break;
          }
        }
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->fd = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure fd is closed
  ret = ::fcntl(this->fd, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->fd);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_context_t ctx;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ctx == nullptr)) {
    register_thread(flag);
  }
  return (void *) ioctx::ctx;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ctx == nullptr) {
    int ret = io_setup(MAX_EVENTS, &ioctx::ctx);
    if (ret != 0) {
      LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  if (ioctx::ctx != nullptr) {
    io_destroy(ioctx::ctx);
    ioctx::ctx = nullptr;
  }
}

void LinuxAlignedFileReader::deregister_all_threads() {
}

void LinuxAlignedFileReader::register_buf(void *buf, uint64_t buf_size, int mrid) {
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->fd = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->fd != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure fd is closed
  ::fcntl(this->fd, F_GETFD);
  //  assert(ret != -1);

  ::close(this->fd);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, this->fd, read_reqs);
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, this->fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->fd != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  auto cb = new iocb_t;
  req.finished = false;
  if (write) {
    io_prep_pwrite(cb, fd, req.buf, req.len, req.offset);
  } else {
    io_prep_pread(cb, fd, req.buf, req.len, req.offset);
  }
  cb->data = (void *) &req;  // set user data to point to the request

  iocb_t *cbs[1] = {cb};  // create an array of iocb_t pointers
  int ret = io_submit((io_context_t) ctx, 1, cbs);
  if (ret != 1) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    delete cb;
  }
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  uint64_t n_ops = std::min(reqs.size(), (uint64_t) MAX_EVENTS);
  std::vector<iocb_t *> cbs(n_ops, nullptr);
  for (uint64_t j = 0; j < n_ops; j++) {
    reqs[j].finished = false;
    cbs[j] = new iocb_t;
    if (write) {
      io_prep_pwrite(cbs[j], fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_prep_pread(cbs[j], fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
    cbs[j]->data = (void *) &reqs[j];  // set user data to point to the request
  }

  // issue reads
  int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
  // if requests didn't get accepted
  if (ret != (int64_t) n_ops) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", " << strerror(errno);
    uint64_t submitted = ret > 0 ? (uint64_t) ret : 0;
    for (uint64_t i = submitted; i < n_ops; i++) {
      delete cbs[i];
    }
  }
}

int LinuxAlignedFileReader::poll(void *ctx) {
  // Poll a single completed IO request in the io_uring context.
  static __thread io_event_t evts[MAX_EVENTS]; 
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, MAX_EVENTS, evts, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return 0;
  }
  for (int i = 0; i < ret; i++) {
    IORequest *req = (IORequest *) evts[i].data;
    if (req != nullptr) {
      req->finished = true;
    }
    delete (iocb_t *) evts[i].obj;
  }
  return ret;
}

void LinuxAlignedFileReader::poll_alloc(void *ctx, std::vector<uint64_t> *page_ref) {
  static __thread io_event_t evts[MAX_EVENTS];
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, MAX_EVENTS, evts, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;
  }
  for (int i = 0; i < ret; i++) {
    IORequest *req = (IORequest *) evts[i].data;
    if (req != nullptr) {
      req->finished = true;
      for (uint64_t j = 0; j < req->len / SECTOR_LEN; j++) {
        uint32_t blk_no = req->offset / SECTOR_LEN + j;
        pipeann::cache.put(pipeann::PageCacheKey(fd, blk_no), (uint8_t *) req->buf + j * SECTOR_LEN, true);
        page_ref->push_back(pipeann::PageCacheKey(fd, blk_no).raw);
      }
    }
    delete (iocb_t *) evts[i].obj;
  }
}
#endif

static bool read_from_cache(int fd, IORequest &req, bool ref = false) {
  uint64_t n_sectors = req.len / SECTOR_LEN;
  uint64_t base_sector = req.offset / SECTOR_LEN;
  for (uint64_t i = 0; i < n_sectors; i++) {
    if (!pipeann::cache.get(pipeann::PageCacheKey(fd, base_sector + i), (uint8_t *) req.buf + i * SECTOR_LEN, ref)) {
      return false;
    }
  }
  req.finished = true;  // mark as finished for cache hit
  return true;
}

int LinuxAlignedFileReader::send_read(IORequest &req, void *ring) {
#ifndef READ_ONLY_TESTS
  if (!read_from_cache(this->fd, req)) {
    send_io(req, ring, false);
  }
#else
  send_io(req, ring, false);
#endif
  return 1;
}

int LinuxAlignedFileReader::send_read(std::vector<IORequest> &reqs, void *ring) {
#ifndef READ_ONLY_TESTS
  // fetch from cache.
  uint32_t cnt = 0;
  for (auto &req : reqs) {
    if (req.offset % SECTOR_LEN != 0 || req.len % SECTOR_LEN != 0) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
    }
    // We do not use batched send_io here, to avoid copy IORequest objects.
    // This avoids use-after-free bug (reqs->finished is set to true when polling).
    if (!read_from_cache(this->fd, req)) {
      send_io(req, ring, false);
      cnt++;
    }
  }
  return cnt;
#else
  send_io(reqs, ring, false);
  return reqs.size();
#endif
}

void LinuxAlignedFileReader::read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;
  bool ref = page_ref != nullptr;

  for (auto &req : read_reqs) {
    if (req.offset % SECTOR_LEN != 0 || req.len % SECTOR_LEN != 0) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
      crash();
    }
    if (!read_from_cache(fd, req, ref)) {
      disk_read_reqs.push_back(req);
    }
  }

  if (disk_read_reqs.size() > 0) {
    read(disk_read_reqs, ctx);
  }

  if (ref) {
    // alloc cache space.
    for (auto &req : disk_read_reqs) {
      for (uint64_t i = 0; i < req.len / SECTOR_LEN; i++) {
        uint32_t blk_no = req.offset / SECTOR_LEN + i;
        pipeann::cache.put(pipeann::PageCacheKey(fd, blk_no), (uint8_t *) req.buf + i * SECTOR_LEN, true);
      }
    }

    // record ref for both cache read and write.
    for (auto &req : read_reqs) {
      for (uint64_t i = 0; i < req.len / SECTOR_LEN; i++) {
        uint32_t blk_no = req.offset / SECTOR_LEN + i;
        page_ref->push_back(pipeann::PageCacheKey(fd, blk_no).raw);
      }
    }
  }
#else
  read(read_reqs, ctx);
#endif
}
