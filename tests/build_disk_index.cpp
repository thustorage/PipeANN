#include <cstdint>
#include "distance.h"
#include "nbr/nbr.h"
#include "omp.h"

#include "utils/index_build_utils.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc < 11 || argc > 16) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_prefix_path> <R>  <L_or_L1>  <PQ_bytes>  <M>  <T>"
                 " <similarity metric (cosine/l2/mips) case sensitive> <nbr_type (pq/rabitq/rabitq2/rabitq4/rabitq8)> [L2 (0 for Vamana, >0 for PiPNN)]"
                 " [train_query_path (for OOD build)] [R_ood] [L_ood] [node_type (raw/rabitq8)]"
                 " See README for more information on parameters."
              << std::endl;
  } else {
    std::string dist_metric(argv[9]);

    pipeann::Metric m = pipeann::get_metric(dist_metric);

    std::string nbr_type = argv[10];
    uint32_t L2 = argc >= 12 ? std::stoi(argv[11]) : 0;
    std::string train_query_path = argc >= 13 ? std::string(argv[12]) : std::string();
    uint16_t R_ood = argc >= 14 ? (uint16_t) std::stoi(argv[13]) : 0;
    uint32_t L_ood = argc >= 15 ? (uint32_t) std::stoi(argv[14]) : 1500;
    const auto node_type = argc >= 16 && std::string(argv[15]) == "rabitq8"
                               ? pipeann::VectorType::RABITQ8 : pipeann::VectorType::RAW;

    if (std::string(argv[1]) == std::string("float")) {
      pipeann::AbstractNeighbor<float> *nbr_handler = pipeann::create_neighbor<float>(m, nbr_type);

      pipeann::build_disk_index<float>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                       std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler, nullptr, 0, L2,
                                       train_query_path, R_ood, L_ood, node_type);
    } else if (std::string(argv[1]) == std::string("int8")) {
      pipeann::AbstractNeighbor<int8_t> *nbr_handler = pipeann::create_neighbor<int8_t>(m, nbr_type);
      pipeann::build_disk_index<int8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                        std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler, nullptr, 0, L2,
                                        train_query_path, R_ood, L_ood, node_type);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      pipeann::AbstractNeighbor<uint8_t> *nbr_handler = pipeann::create_neighbor<uint8_t>(m, nbr_type);
      pipeann::build_disk_index<uint8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                         std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler, nullptr, 0,
                                         L2, train_query_path, R_ood, L_ood, node_type);
    } else {
      LOG(ERROR) << "Error. wrong file type";
    }
  }
}
