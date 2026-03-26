#include <cstdint>
#include "distance.h"
#include "nbr/nbr.h"
#include "filter/selector.h"
#include "filter/label.h"
#include "omp.h"

#include "pipnn.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc < 12) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_prefix_path> <R>  <L1Fanout>  <L2Fanout>  <PQ_bytes>  <M>  <T>"
                 " <similarity metric (cosine/l2/mips) case sensitive> <nbr_type (pq/rabitq)> <(optional) label_type "
                 "(spmat)> <(optional) label_file.spmat>."
                 " See README for more information on parameters."
              << std::endl;
  } else {
    std::string dist_metric(argv[10]);

    pipeann::Metric m = pipeann::get_metric(dist_metric);

    std::string nbr_type = argv[11];
    std::string label_type = argc > 13 ? argv[12] : "null";
    std::string label_file = argc > 13 ? argv[13] : "";

    pipeann::AbstractLabel *label = nullptr;
    if (label_type == "spmat") {
      if (label_file.empty()) {
        LOG(ERROR) << "Error. label_file is required for spmat label writer.";
        crash();
      }
      label = new pipeann::SpmatLabel(label_file);
    }

    uint32_t R = std::stoi(argv[4]);
    uint32_t l1_fanout = std::stoi(argv[5]);
    uint32_t l2_fanout = std::stoi(argv[6]);
    uint32_t pq_bytes = std::stoi(argv[7]);
    uint32_t M = std::stoi(argv[8]);
    uint32_t T = std::stoi(argv[9]);

    if (std::string(argv[1]) == std::string("float")) {
      pipeann::AbstractNeighbor<float> *nbr_handler = pipeann::get_nbr_handler<float>(m, nbr_type);
      pipeann::build_pipnn_index<float>(argv[2], argv[3], R, l1_fanout, l2_fanout, M, T, pq_bytes, m, nullptr,
                                        nbr_handler, label);
    } else if (std::string(argv[1]) == std::string("int8")) {
      pipeann::AbstractNeighbor<int8_t> *nbr_handler = pipeann::get_nbr_handler<int8_t>(m, nbr_type);
      pipeann::build_pipnn_index<int8_t>(argv[2], argv[3], R, l1_fanout, l2_fanout, M, T, pq_bytes, m, nullptr,
                                         nbr_handler, label);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      pipeann::AbstractNeighbor<uint8_t> *nbr_handler = pipeann::get_nbr_handler<uint8_t>(m, nbr_type);
      pipeann::build_pipnn_index<uint8_t>(argv[2], argv[3], R, l1_fanout, l2_fanout, M, T, pq_bytes, m, nullptr,
                                          nbr_handler, label);
    } else {
      LOG(ERROR) << "Error. wrong file type";
    }
  }
}
