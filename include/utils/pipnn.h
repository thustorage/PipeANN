#pragma once
#include <cstdint>

namespace pipeann {
  // Candidate edges produced by PiPNN's overlapping partitions.
  struct WeightedEdge {
    uint32_t u, v;
    float dist;
  };
}
