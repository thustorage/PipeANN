#pragma once

#include "dynamic_index.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PyIndexInterface {
 public:
  PyIndexInterface() = delete;
  PyIndexInterface(uint32_t data_dim, const std::string &data_type, pipeann::Metric metric,
                   pipeann::IndexBuildParameters *params = nullptr, uint32_t attr_size = 0);

  // lifecycle and I/O
  void load(const std::string &index_prefix);
  bool save(const std::string &index_prefix);

  // build
  void build(const std::string &data_path, const std::string &index_prefix, const char *tag_file = nullptr,
             bool build_mem_index = false, uint32_t max_nbrs = 0, uint32_t build_L = 0, uint32_t PQ_bytes = 0,
             uint32_t memory_use_GB = 0, const std::vector<pipeann::Attributes> *attrs_vec = nullptr,
             uint32_t range_dense = 0, const std::string &train_query_path = "", uint32_t R_ood = 0,
             uint32_t L_ood = 1500);
  std::shared_ptr<pipeann::AttrIndex> load_attr_index_from_file(uint32_t key, const std::string &filename,
                                                                const std::string &attr_type);
  std::shared_ptr<pipeann::AttrIndex> create_attr_index(uint32_t key, const std::string &filename,
                                                        const std::string &attr_type);
  // Compile a SQL-like filter string using the supplied schema.
  // Schema entries: {field_name: (key_uint32, type_str, attr_index_shared_ptr)}.
  // Returns CompiledFilter (selector + attrs_template + slot_map for `$$var`
  // placeholders + var_field_type). When no placeholders are used, slot_map is
  // empty and attrs_template can be passed directly to search.
  pipeann::dsl::CompiledFilter compile_filter(const std::string &json, const py::dict &schema);

  // End-to-end JSON-driven filter loader. Parses the unified config
  //   {"attr_indexes": [...], "filter": "<SQL expr>", "bindings": {var: spmat_path}}
  // and returns (Selector*, vector<Attributes>) — one Attributes per query row.
  // The AttrIndexes declared in `attr_indexes` are owned by the underlying
  // DynamicIndex (kept alive in native_attr_indexes_).
  std::tuple<pipeann::Selector *, std::vector<pipeann::Attributes>> load_filter_from_json(
      const std::string &config_path);

  // Pure scalar filter: pre_filter(query_attrs) → tags, truncated to limit (0 = unlimited).
  py::array filter_only(pipeann::Selector *selector, const pipeann::Attributes &query_attrs, size_t limit);

  // updates and queries
  std::tuple<py::array, py::array> search(py::array &queries, uint32_t topk, uint32_t L, pipeann::Selector *selector,
                                          const std::vector<pipeann::Attributes> &query_attrs, float range);
  void add(py::array &vectors, py::array &tags, const std::vector<pipeann::Attributes> *attrs_vec = nullptr);
  void remove(py::array &tags);
  void set_index_prefix(const std::string &index_prefix);
  void omp_set_num_threads(uint32_t num_threads);
  size_t npoints() const;

  std::string to_string() const;

 private:
  template<typename T>
  DynamicIndex<T> *get() const {
    return dynamic_cast<DynamicIndex<T> *>(impl_.get());
  }

  // Type-erased dispatch: calls fn(DynamicIndex<T>*) for the correct T.
  template<typename F>
  decltype(auto) dispatch(F &&fn) const {
    if (auto *p = get<float>())
      return fn(p);
    if (auto *p = get<uint8_t>())
      return fn(p);
    if (auto *p = get<int8_t>())
      return fn(p);
    throw std::runtime_error("Invalid underlying index");
  }

 private:
  uint32_t data_dim_;
  pipeann::Metric metric_;
  std::unique_ptr<BaseDynamicIndex> impl_;
};
