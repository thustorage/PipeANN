#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <iomanip>

#include "utils/log.h"
#include "filter/attribute.h"
#include "filter/dsl_compiler.h"
#include "nbr/nbr.h"
#include "utils/timer.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"

template<typename T>
int search_disk_index(int argc, char **argv) {
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<uint64_t> Lvec;

  int index = 2;
  std::string index_prefix_path(argv[index++]);
  uint32_t num_threads = std::atoi(argv[index++]);
  uint32_t beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  uint64_t recall_at = std::atoi(argv[index++]);
  std::string dist_metric(argv[index++]);
  std::string nbr_type = argv[index++];
  std::string label_config_file(argv[index++]);
  uint32_t mem_L = std::atoi(argv[index++]);

  pipeann::Metric m = pipeann::get_metric(dist_metric);

  for (int ctr = index; ctr < argc; ctr++) {
    uint64_t curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.empty()) {
    LOG(ERROR) << "No valid Lsearch found. Lsearch must be >= recall_at";
    return -1;
  }

  LOG(INFO) << "Search parameters: threads=" << num_threads << ", beamwidth=" << beamwidth;

  // Load query vectors
  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);

  // Load ground truth if available
  bool calc_recall_flag = false;
  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      LOG(ERROR) << "Mismatch in number of queries and ground truth data";
    }
    calc_recall_flag = true;
  }

  // Initialize index
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(m, nbr_type);
  pipeann::IndexBuildParameters idx_params;
  idx_params.max_nthreads = num_threads; // limit the number of allocated buffers.
  std::unique_ptr<pipeann::SSDIndex<T>> _pFlashIndex(
      new pipeann::SSDIndex<T>(m, reader, nbr_handler, true, &idx_params));

  if (_pFlashIndex->load(index_prefix_path.c_str()) != 0) {
    return -1;
  }

  if (mem_L != 0) {
    auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index from " << mem_index_path;
    _pFlashIndex->load_mem_index(mem_index_path);
  }

  auto ret = pipeann::dsl::load_filter_from_json(
      label_config_file, _pFlashIndex->meta_.npoints);
  auto &selector = std::get<0>(ret);
  auto &query_attrs = std::get<1>(ret);
  auto &base_stores = std::get<2>(ret);
    
  LOG(INFO) << "Loaded filter from " << label_config_file << "; "
            << base_stores.size() << " attr indexes, "
            << query_attrs.size() << " queries";

  omp_set_num_threads(num_threads);

  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
    uint64_t L = Lvec[test_id];

    query_result_tags[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      _pFlashIndex->spec_filter_search(query + (i * query_dim), recall_at, L, selector, query_attrs[i],
                                       query_result_tags[test_id].data() + (i * recall_at),
                                       query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                       stats + i);
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) query_num / diff.count();

    float mean_latency =
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.total_us; });
    float latency_99 =
        pipeann::get_percentile_stats(stats, query_num, 0.99f, [](const pipeann::QueryStats &s) { return s.total_us; });
    float mean_hops = pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_hops; });
    float mean_ios = pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_ios; });

    float filter_ratio[pipeann::N_FILTER_TYPES] = {0};
    for (int i = 0; i < pipeann::N_FILTER_TYPES; i++) {
      filter_ratio[i] =
          pipeann::get_mean_stats(stats, query_num, [i](const pipeann::QueryStats &s) { return s.n_filter[i]; });
    }

    struct FilterStats {
      double n_est_filter_reads[pipeann::N_FILTER_TYPES] = {0};
      double n_est_filter_cmps[pipeann::N_FILTER_TYPES] = {0};
      double n_filter_reads[pipeann::N_FILTER_TYPES] = {0};
      double n_filter_accessed_vectors[pipeann::N_FILTER_TYPES] = {0};
      double n_filter_false_positives[pipeann::N_FILTER_TYPES] = {0};
    };
    FilterStats filter_stats;

    for (int i = 0; i < pipeann::N_FILTER_TYPES; i++) {
      filter_stats.n_est_filter_reads[i] =
          pipeann::get_mean_stats(stats, query_num * filter_ratio[i], query_num,
                                  [i](const pipeann::QueryStats &s) { return s.n_est_filter_reads[i]; });
      filter_stats.n_filter_reads[i] =
          pipeann::get_mean_stats(stats, query_num * filter_ratio[i], query_num,
                                  [i](const pipeann::QueryStats &s) { return s.n_filter_reads[i]; });
      filter_stats.n_filter_accessed_vectors[i] =
          pipeann::get_mean_stats(stats, query_num * filter_ratio[i], query_num,
                                  [i](const pipeann::QueryStats &s) { return s.n_filter_accessed_vectors[i]; });
      filter_stats.n_filter_false_positives[i] =
          pipeann::get_mean_stats(stats, query_num * filter_ratio[i], query_num,
                                  [i](const pipeann::QueryStats &s) { return s.n_filter_false_positives[i]; });
      filter_stats.n_est_filter_cmps[i] =
          pipeann::get_mean_stats(stats, query_num * filter_ratio[i], query_num,
                                  [i](const pipeann::QueryStats &s) { return s.n_est_filter_cmps[i]; });
    }
    delete[] stats;

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        recall =
            pipeann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                      query_result_tags[test_id].data(), (uint32_t) recall_at, (uint32_t) recall_at);
      }

      std::cout << std::setw(4) << L << std::setw(4) << beamwidth << std::setw(10) << qps << std::setw(8)
                << mean_latency << std::setw(10) << query_num * filter_ratio[pipeann::PRE_FILTER] << std::setw(8)
                << filter_stats.n_est_filter_reads[pipeann::PRE_FILTER] << std::setw(8)
                << filter_stats.n_filter_reads[pipeann::PRE_FILTER] << std::setw(8)
                << filter_stats.n_filter_accessed_vectors[pipeann::PRE_FILTER] << std::setw(6)
                << filter_stats.n_filter_false_positives[pipeann::PRE_FILTER] << std::setw(8)
                << filter_stats.n_est_filter_cmps[pipeann::PRE_FILTER] << std::setw(10)
                << query_num * filter_ratio[pipeann::IN_FILTER] << std::setw(8)
                << filter_stats.n_est_filter_reads[pipeann::IN_FILTER] << std::setw(8)
                << filter_stats.n_filter_reads[pipeann::IN_FILTER] << std::setw(8)
                << filter_stats.n_filter_accessed_vectors[pipeann::IN_FILTER] << std::setw(6)
                << filter_stats.n_filter_false_positives[pipeann::IN_FILTER] << std::setw(8)
                << filter_stats.n_est_filter_cmps[pipeann::IN_FILTER] << std::setw(10)
                << query_num * filter_ratio[pipeann::POST_FILTER] << std::setw(8)
                << filter_stats.n_est_filter_reads[pipeann::POST_FILTER] << std::setw(8)
                << filter_stats.n_filter_reads[pipeann::POST_FILTER] << std::setw(8)
                << filter_stats.n_filter_accessed_vectors[pipeann::POST_FILTER] << std::setw(6)
                << filter_stats.n_filter_false_positives[pipeann::POST_FILTER] << std::setw(8)
                << filter_stats.n_est_filter_cmps[pipeann::POST_FILTER] << std::setw(8) << mean_ios;
      if (calc_recall_flag) {
        std::cout << std::setw(10) << recall;
      }
      std::cout << std::endl;
    }
  };

  // Print header
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(1);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "L" << std::setw(4) << "BW" << std::setw(10) << "QPS" << std::setw(8)
            << "Avg(us)" << std::setw(10) << "#Pre" << std::setw(8) << "EstIO" << std::setw(8) << "IO"
            << std::setw(8) << "#Vec" << std::setw(6) << "FP" << std::setw(8) << "EstCmp" << std::setw(10)
            << "#In" << std::setw(8) << "EstIO" << std::setw(8) << "IO" << std::setw(8) << "#Vec"
            << std::setw(6) << "#FP" << std::setw(8) << "EstCmp" << std::setw(10) << "#Post" << std::setw(8)
            << "EstIO" << std::setw(8) << "IO" << std::setw(8) << "#Vec" << std::setw(6) << "#FP" << std::setw(8)
            << "EstCmp" << std::setw(8) << "AvgIO";
  if (calc_recall_flag) {
    std::cout << std::setw(10) << recall_string;
  }
  std::cout << std::endl;
  std::cout << std::string(90, '=') << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }

  // Selector + base_stores are leaked; OS reclaims at process exit.
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 12) {
    std::cout << "Usage: " << argv[0] << " <index_type (float/int8/uint8)>"
              << " <index_prefix_path>"
              << " <num_threads>"
              << " <beamwidth>"
              << " <query_file.bin>"
              << " <truthset.bin (use \"null\" for none)>"
              << " <K>"
              << " <similarity (cosine/l2/mips)>"
              << " <nbr_type (pq/rabitq)>"
              << " <label_config_file.json>"
              << " <mem_L (0 means no mem index)>"
              << " <L1> [L2] ..." << std::endl;
    return -1;
  }

  std::string index_type = argv[1];
  if (index_type == "float")
    return search_disk_index<float>(argc, argv);
  else if (index_type == "int8")
    return search_disk_index<int8_t>(argc, argv);
  else if (index_type == "uint8")
    return search_disk_index<uint8_t>(argc, argv);
  else {
    std::cout << "Unsupported index type: " << index_type << ". Use float/int8/uint8" << std::endl;
    return -1;
  }
}
