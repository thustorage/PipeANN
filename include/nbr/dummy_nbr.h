#pragma once

#include "nbr/abstract_nbr.h"
#include "utils.h"
#include <sstream>
#include <string_view>
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index_defs.h"

namespace pipeann {
  template<typename T>
  class DummyNeighbor : public AbstractNeighbor<T> {
   public:
    DummyNeighbor(pipeann::Metric metric) : AbstractNeighbor<T>(metric) {
    }

    virtual ~DummyNeighbor() = default;

    std::string get_name() {
      return "DummyNeighbor";
    }
  };
}  // namespace pipeann
