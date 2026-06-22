#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cblas.h>

namespace kmeans {
  void compute_vecs_l2sq(float *vecs_l2sq, float *data, const size_t num_points, const size_t dim);

  // Given data in num_points * dim row major
  // Pivots stored in pivot_data as num_centers * dim row major
  // Calculate the k closest centers for each point and store in closest_centers_ivf
  // (which needs to be pre-allocated as num_points * k)
  void compute_closest_centers(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers,
                               size_t k, uint32_t *closest_centers_ivf);

  // Run Elkan until max_reps or stopping criterion
  // Elkan's algorithm uses triangle inequality to avoid unnecessary distance computations
  // If you pass NULL for closest_docs and closest_center, it will NOT return the results,
  // else it will assume appropriate allocation as closest_docs = new vector<size_t> [num_centers],
  // and closest_center = new size_t[num_points]
  // Final centers are output in centers as row major num_centers * dim
  float run_elkan(float *data, size_t num_points, size_t dim, float *centers, const size_t num_centers,
                  const size_t max_reps, std::vector<size_t> *closest_docs, uint32_t *closest_center);

  // assumes already memory allocated for pivot_data as new
  // float[num_centers*dim] and select randomly num_centers points as pivots
  void kmeanspp_selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers);
}  // namespace kmeans
