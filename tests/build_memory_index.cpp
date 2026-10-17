#include <index.h>
#include <omp.h>
#include <string.h>
#include <numeric>
#include "ssd_index.h"
#include "utils.h"
#include "utils/index_build_utils.h"
#include <sys/mman.h>
#include <unistd.h>

template<typename T>
int build_in_memory_index(const std::string &data_path, const std::string &tags_file, const unsigned R,
                          const unsigned L, const float alpha, const std::string &save_path, const unsigned num_threads,
                          pipeann::Metric distMetric, const std::string &train_query_path, uint32_t R_ood,
                          uint32_t L_ood, pipeann::VectorType node_type) {
  pipeann::IndexBuildParameters paras;
  paras.set(R, L, 750, alpha, num_threads, false);

  uint64_t data_num, data_dim;
  pipeann::get_bin_metadata(data_path, data_num, data_dim);
  std::cout << "Building in-memory index with parameters: data_file: " << data_path << " tags file: " << tags_file
            << " R: " << R << " L: " << L << " alpha: " << alpha << " index_path: " << save_path
            << " #threads: " << num_threads << " train_query: " << train_query_path << " R_ood: " << R_ood
            << " L_ood: " << L_ood
            << ", using distance metric: " << (distMetric == pipeann::Metric::COSINE ? "cosine " : "l2 ");

  typedef uint32_t TagT;

  std::cout << "Opening bin file " << tags_file << "... " << std::endl;
  std::vector<TagT> tags;
  size_t tag_rows, tag_cols;
  pipeann::load_bin<TagT>(tags_file, tags, tag_rows, tag_cols);

  auto s = std::chrono::high_resolution_clock::now();
  const auto graph_file = pipeann::prepare_build_vectors<T>(data_path, distMetric, node_type);
  pipeann::Index<T, TagT> index(pipeann::load_neighbor<T>(graph_file, distMetric));
  index.build(paras, tags, train_query_path, R_ood, L_ood);
  std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";

  pipeann::SSDIndex<T, TagT>::save_from_mem(index, save_path);

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 10 || argc > 14) {
    std::cout << "Usage: " << argv[0]
              << " <data_type(int8/uint8/float)> <data_file.bin> <tags_file>"
                 " <output_index_file>"
              << " <R> <L> <alpha> <num_threads_to_use>"
              << " <distance_metric(l2/cosine/mips)>"
              << " [train_query_path] [R_ood] [L_ood] [node_type (raw/rabitq8)]."
              << " See README for more information on parameters." << std::endl;
    exit(-1);
  }

  int arg_no = 2;

  const std::string data_path(argv[arg_no++]);
  const std::string tags_file(argv[arg_no++]);
  const std::string save_path(argv[arg_no++]);
  const unsigned R = (unsigned) atoi(argv[arg_no++]);
  const unsigned L = (unsigned) atoi(argv[arg_no++]);
  const float alpha = (float) atof(argv[arg_no++]);
  const unsigned num_threads = (unsigned) atoi(argv[arg_no++]);
  const std::string dist_metric_str = argv[arg_no++];
  pipeann::Metric distMetric = pipeann::get_metric(dist_metric_str);

  std::string train_query_path = (argc > 10) ? std::string(argv[10]) : std::string();
  uint32_t R_ood = (argc > 11) ? (uint32_t) atoi(argv[11]) : 0;
  uint32_t L_ood = (argc > 12) ? (uint32_t) atoi(argv[12]) : 1500;

  const auto node_type = argc <= 13 || std::string(argv[13]) == "rabitq8"
                             ? pipeann::VectorType::RABITQ8 : pipeann::VectorType::RAW;

  if (std::string(argv[1]) == std::string("int8"))
    build_in_memory_index<int8_t>(data_path, tags_file, R, L, alpha, save_path, num_threads, distMetric,
                                  train_query_path, R_ood, L_ood, node_type);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_in_memory_index<uint8_t>(data_path, tags_file, R, L, alpha, save_path, num_threads, distMetric,
                                   train_query_path, R_ood, L_ood, node_type);
  else if (std::string(argv[1]) == std::string("float"))
    build_in_memory_index<float>(data_path, tags_file, R, L, alpha, save_path, num_threads, distMetric,
                                 train_query_path, R_ood, L_ood, node_type);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
