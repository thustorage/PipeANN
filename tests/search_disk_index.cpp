#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <iomanip>
#include <limits>

#include "utils/log.h"
#include "nbr/nbr.h"
#include "utils/timer.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"

template<typename T>
int search_disk_index(int argc, char **argv) {
  // load query bin
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<uint64_t> Lvec;

  bool tags_flag = true;

  int index = 2;
  std::string index_prefix_path(argv[index++]);
  uint32_t num_threads = std::atoi(argv[index++]);
  uint32_t beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  uint64_t recall_at = std::atoi(argv[index++]);
  std::string dist_metric(argv[index++]);
  std::string nbr_type = argv[index++];
  int search_mode = std::atoi(argv[index++]);
  uint32_t mem_L = std::atoi(argv[index++]);

  pipeann::Metric m = pipeann::get_metric(dist_metric);

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    uint64_t l_search = std::atoi(argv[ctr]);
    Lvec.push_back(l_search);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at" << std::endl;
    return -1;
  }

  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    std::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    std::cout << " beamwidth: " << beamwidth << std::endl;

  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);

  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(m, nbr_type);
  pipeann::IndexBuildParameters idx_params;
  idx_params.max_nthreads = std::max<uint32_t>(16, 2 * num_threads);
  std::unique_ptr<pipeann::SSDIndex<T>> _pFlashIndex(
      new pipeann::SSDIndex<T>(m, reader, nbr_handler, tags_flag, &idx_params));

  int res = _pFlashIndex->load(index_prefix_path.c_str(), false);
  if (res != 0) {
    return res;
  }

  if (mem_L != 0) {
    auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index from " << mem_index_path;
    _pFlashIndex->load_mem_index(mem_index_path);
  }

  omp_set_num_threads(num_threads);

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
    uint64_t L = Lvec[test_id];

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    auto s = std::chrono::high_resolution_clock::now();

    if (search_mode == SearchMode::PIPE_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->pipe_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i);
      }
    } else if (search_mode == SearchMode::PAGE_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->page_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i);
      }
    } else if (search_mode == SearchMode::CORO_SEARCH) {
      constexpr uint64_t kBatchSize = 8;
      T *q[kBatchSize];
      uint32_t *res_tags[kBatchSize];
      float *res_dists[kBatchSize];
      int N;
#pragma omp parallel for schedule(dynamic, 1) private(q, res_tags, res_dists, N)
      for (int64_t i = 0; i < (int64_t) query_num; i += kBatchSize) {
        N = std::min(kBatchSize, query_num - i);
        for (int v = 0; v < N; ++v) {
          q[v] = query + ((i + v) * query_dim);
          res_tags[v] = query_result_tags_32.data() + ((i + v) * recall_at);
          res_dists[v] = query_result_dists[test_id].data() + ((i + v) * recall_at);
        }

        _pFlashIndex->coro_search(q, (uint64_t) recall_at, mem_L, (uint64_t) L, res_tags, res_dists,
                                  (uint64_t) beamwidth, N);
      }
    } else if (search_mode == SearchMode::BEAM_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->beam_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i);
      }
    } else {
      std::cout << "Unknown search mode: " << search_mode << std::endl;
      exit(-1);
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));

    pipeann::convert_types<uint32_t, uint32_t>(query_result_tags_32.data(), query_result_tags[test_id].data(),
                                               (size_t) query_num, (size_t) recall_at);

    float mean_latency = (float) pipeann::get_mean_stats(
        stats, query_num, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float latency_99 = (float) pipeann::get_percentile_stats(
        stats, query_num, 0.99f, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float mean_hops = (float) pipeann::get_mean_stats(stats, query_num,
                                                      [](const pipeann::QueryStats &stats) { return stats.n_hops; });

    float mean_ios =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

    delete[] stats;

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        /* Attention: in SPACEV, there may be multiple vectors with the same distance,
          which may cause lower than expected recall@1 (?) */
        recall = (float) pipeann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                                   query_result_tags[test_id].data(), (uint32_t) recall_at,
                                                   (uint32_t) recall_at);
      }

      std::cout << std::setw(10) << L
                << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12) << mean_latency << std::setw(12)
                << latency_99 << std::setw(12) << mean_hops << std::setw(12) << mean_ios;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall << std::endl;
      }
    }
  };

  // LOG(INFO) << "Use two ANNS for warming up...";
  // uint32_t prev_L = Lvec[0];
  // Lvec[0] = 200;
  // run_tests(0, false);
  // run_tests(0, false);
  // Lvec[0] = prev_L;
  // LOG(INFO) << "Warming up finished.";

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(10) << "L" << std::setw(12) << "I/O Width" << std::setw(12)
            << "QPS" << std::setw(12) << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops"
            << std::setw(12) << "Mean IOs" << std::setw(12);
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "=============================================="
               "==========================================="
            << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 12) {
    // tags == 1!
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)>  <index_prefix_path>"
                 " <num_threads>  <pipeline width> "
                 " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
                 " <K> <similarity (cosine/l2/mips)> <nbr_type (pq/rabitq)>"
                 " <search_mode(0 for beam search / 1 for page search / 2 for pipe search)> <mem_L (0 means not "
                 "using mem index)> <L_search1> [<L_search2> ...]"
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8" << std::endl;
}
