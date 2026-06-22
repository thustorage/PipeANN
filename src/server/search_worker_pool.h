// Fixed-size search thread pool. Each worker lazily initializes its thread_local
// io_uring ring on first search, so the number of kernel io_uring workers stays
// bounded regardless of request concurrency. Shared by the gRPC server and the
// Python binding via CollectionStore.
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace pipeann {
namespace server {

class SearchWorkerPool {
 public:
  explicit SearchWorkerPool(int n_workers) {
    for (int i = 0; i < n_workers; i++) {
      workers_.emplace_back([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            task = std::move(queue_.front());
            queue_.pop();
          }
          task();
        }
      });
    }
  }

  ~SearchWorkerPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : workers_) t.join();
  }

  // Submit a task and block until it completes.
  void submit_and_wait(std::function<void()> task) {
    std::promise<void> done;
    auto fut = done.get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push([&task, &done] {
        task();
        done.set_value();
      });
    }
    cv_.notify_one();
    fut.wait();
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> queue_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace server
}  // namespace pipeann
