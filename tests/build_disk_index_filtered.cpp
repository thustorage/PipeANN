#include <cstdint>
#include "distance.h"
#include "nbr/nbr.h"
#include "filter/attribute.h"
#include "omp.h"

#include "utils/index_build_utils.h"
#include "utils.h"

// by default, we output the label index file to /index_prefix_path.label.key.
inline std::string label_filename(const std::string &index_prefix_path, uint32_t key) {
  return index_prefix_path + ".label." + std::to_string(key);
}

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin (null if only build labels)>"
                 " <index_prefix_path> <R> <R_dense> <L>  <PQ_bytes>  <M>  <T>"
                 " <similarity metric (cosine/l2/mips) case sensitive> <nbr_type (pq/rabitq)> "
                 "<label_type_1 (label_spmat/range)> <label_file_1> <label_type_2> <label_file_2> ..."
              << std::endl;
  } else {
    std::string index_prefix_path(argv[3]);
    std::string dist_metric(argv[10]);

    pipeann::Metric m = pipeann::get_metric(dist_metric);

    std::map<uint32_t, pipeann::AttrIndex *> labels;
    // Save AttrIndex to disk.
    int first_label_index = 12;
    for (int i = first_label_index; i < argc; i += 2) {
      uint32_t key = (i - first_label_index) / 2;
      std::string label_type = argv[i];
      std::string label_file = argv[i + 1];
      if (label_type == "label_spmat") {
        LOG(INFO) << "Loading label spmat " << key << " from " << label_file;
        auto val = new pipeann::InvertedLabelAttrIndex(label_filename(index_prefix_path, key), 0);
        val->load_from_spmat(label_file);
        labels[key] = val;
      } else if (label_type == "range") {
        auto val = new pipeann::SortedRangeAttrIndex(label_filename(index_prefix_path, key), 0);
        if (pipeann::is_suffix(label_file, ".bin")) {
          LOG(INFO) << "Loading range bin " << key << " from " << label_file;
          val->load_from_bin(label_file);
        } else if (pipeann::is_suffix(label_file, ".spmat")) {
          LOG(INFO) << "Loading range spmat " << key << " from " << label_file;
          val->load_from_spmat(label_file);
        }
        labels[key] = val;
      } else {
        LOG(ERROR) << "Unknown label type: " << label_type;
        exit(-1);
      }
      labels[key]->save();
      LOG(INFO) << "Saved label " << key << " to " << label_filename(index_prefix_path, key);
    }

    if (std::string(argv[2]) != "null") {
      auto attr_writer = new pipeann::AttrIndexWriter(labels);

      std::string nbr_type = argv[11];
      if (std::string(argv[1]) == std::string("float")) {
        pipeann::AbstractNeighbor<float> *nbr_handler = pipeann::create_neighbor<float>(m, nbr_type);
        pipeann::build_disk_index<float>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[6]), std::stoi(argv[8]),
                                         std::stoi(argv[9]), std::stoi(argv[7]), m, nullptr, nbr_handler, attr_writer,
                                         std::stoi(argv[5]));
      } else if (std::string(argv[1]) == std::string("int8")) {
        pipeann::AbstractNeighbor<int8_t> *nbr_handler = pipeann::create_neighbor<int8_t>(m, nbr_type);
        pipeann::build_disk_index<int8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[6]), std::stoi(argv[8]),
                                          std::stoi(argv[9]), std::stoi(argv[7]), m, nullptr, nbr_handler, attr_writer,
                                          std::stoi(argv[5]));
      } else if (std::string(argv[1]) == std::string("uint8")) {
        pipeann::AbstractNeighbor<uint8_t> *nbr_handler = pipeann::create_neighbor<uint8_t>(m, nbr_type);
        pipeann::build_disk_index<uint8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[6]), std::stoi(argv[8]),
                                           std::stoi(argv[9]), std::stoi(argv[7]), m, nullptr, nbr_handler,
                                           attr_writer, std::stoi(argv[5]));
      } else {
        LOG(ERROR) << "Error. wrong file type";
      }
    }
  }
}
