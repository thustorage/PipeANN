#include "pyindex.h"

#include <pybind11/stl.h>

PyIndexInterface::PyIndexInterface(uint32_t data_dim, const std::string &data_type, pipeann::Metric metric,
                                   pipeann::IndexBuildParameters *params, uint32_t attr_size)
    : data_dim_(data_dim), metric_(metric) {
  if (data_type == "float32")
    impl_.reset(new DynamicIndex<float>(data_dim_, metric_, params, attr_size));
  else if (data_type == "uint8")
    impl_.reset(new DynamicIndex<uint8_t>(data_dim_, metric_, params, attr_size));
  else if (data_type == "int8")
    impl_.reset(new DynamicIndex<int8_t>(data_dim_, metric_, params, attr_size));
  else
    throw std::runtime_error("Unsupported data type: " + data_type);
}

void PyIndexInterface::load(const std::string &index_prefix) {
  dispatch([&](auto *p) { p->load(index_prefix); });
}

bool PyIndexInterface::save(const std::string &index_prefix) {
  return dispatch([&](auto *p) { return p->save(index_prefix); });
}

void PyIndexInterface::build(const std::string &data_path, const std::string &index_prefix, const char *tag_file,
                             bool build_mem_index, uint32_t max_nbrs, uint32_t build_L, uint32_t PQ_bytes,
                             uint32_t memory_use_GB, const std::vector<pipeann::Attributes> *attrs_vec,
                             uint32_t range_dense, const std::string &train_query_path, uint32_t R_ood,
                             uint32_t L_ood) {
  auto *attr_writer =
      attrs_vec ? new pipeann::AttrVecWriter(*attrs_vec, dispatch([](auto *p) { return p->attr_size(); })) : nullptr;
  dispatch([&](auto *p) {
    p->build(data_path, index_prefix, tag_file, build_mem_index, max_nbrs, build_L, PQ_bytes, memory_use_GB,
             attr_writer, range_dense, train_query_path, R_ood, L_ood);
  });
  if (attr_writer) {
    delete attr_writer;
  }
}

std::shared_ptr<pipeann::AttrIndex> PyIndexInterface::load_attr_index_from_file(uint32_t key,
                                                                                const std::string &filename,
                                                                                const std::string &attr_type) {
  return dispatch([&](auto *p) { return p->load_attr_index_from_file(key, filename, attr_type); });
}

std::shared_ptr<pipeann::AttrIndex> PyIndexInterface::create_attr_index(uint32_t key, const std::string &filename,
                                                                        const std::string &attr_type) {
  return dispatch([&](auto *p) { return p->create_attr_index(key, filename, attr_type); });
}

std::tuple<py::array, py::array> PyIndexInterface::search(py::array &queries, uint32_t topk, uint32_t L,
                                                          pipeann::Selector *selector,
                                                          const std::vector<pipeann::Attributes> &query_attrs,
                                                          float range) {
  using TagT = uint32_t;

  auto n_queries = queries.shape(0);

  auto ret_ids = py::array_t<TagT>({n_queries, (py::ssize_t) topk});
  auto ret_dists = py::array_t<float>({n_queries, (py::ssize_t) topk});

  auto *ret_ids_p = static_cast<TagT *>(ret_ids.request().ptr);
  auto *ret_dists_p = static_cast<float *>(ret_dists.request().ptr);

  dispatch([&](auto *p) {
    using T = typename std::remove_pointer_t<decltype(p)>::data_type;
    auto q = py::array_t<T>(queries);
    auto *queries_p = static_cast<T *>(q.request().ptr);
    py::gil_scoped_release release;
    p->search(queries_p, n_queries, topk, L, ret_ids_p, ret_dists_p, selector, query_attrs, range);
  });

  return std::make_tuple(ret_ids, ret_dists);
}

void PyIndexInterface::add(py::array &vectors, py::array &tags, const std::vector<pipeann::Attributes> *attrs_vec) {
  using TagT = uint32_t;
  auto t = py::array_t<TagT>(tags);
  auto *tags_p = static_cast<TagT *>(t.request().ptr);
  auto n_vectors = vectors.shape(0);

  dispatch([&](auto *p) {
    using T = typename std::remove_pointer_t<decltype(p)>::data_type;
    auto v = py::array_t<T>(vectors);
    p->add(static_cast<T *>(v.request().ptr), tags_p, n_vectors, attrs_vec);
  });
}

void PyIndexInterface::remove(py::array &tags) {
  using TagT = uint32_t;
  auto t = py::array_t<TagT>(tags);
  auto *tags_p = static_cast<TagT *>(t.request().ptr);
  auto n_tags = t.shape(0);
  dispatch([&](auto *p) { p->remove(tags_p, n_tags); });
}

void PyIndexInterface::set_index_prefix(const std::string &index_prefix) {
  dispatch([&](auto *p) { p->set_index_prefix(index_prefix); });
}

void PyIndexInterface::omp_set_num_threads(uint32_t num_threads) {
  dispatch([&](auto *p) { p->omp_set_num_threads(num_threads); });
}

size_t PyIndexInterface::npoints() const {
  return dispatch([](auto *p) { return p->npoints(); });
}

pipeann::dsl::CompiledFilter PyIndexInterface::compile_filter(const std::string &json, const py::dict &schema) {
  pipeann::dsl::Schema cpp_schema;
  for (auto item : schema) {
    std::string name = py::cast<std::string>(item.first);
    auto tup = py::cast<py::tuple>(item.second);
    if (tup.size() != 3) {
      throw std::runtime_error("compile_filter schema entry must be (key:int, type:str, attr_index)");
    }
    pipeann::dsl::FieldInfo info;
    info.key = py::cast<uint32_t>(tup[0]);
    info.type = py::cast<std::string>(tup[1]);
    auto ai = py::cast<std::shared_ptr<pipeann::AttrIndex>>(tup[2]);
    info.attr_index = ai.get();
    info.n_vectors = static_cast<uint32_t>(ai->n_vectors.load());
    cpp_schema[name] = info;
  }
  return dispatch([&](auto *p) { return p->compile_filter(json, cpp_schema); });
}

std::tuple<pipeann::Selector *, std::vector<pipeann::Attributes>> PyIndexInterface::load_filter_from_json(
    const std::string &config_path) {
  return dispatch([&](auto *p) { return p->load_filter_from_json(config_path); });
}

py::array PyIndexInterface::filter_only(pipeann::Selector *selector, const pipeann::Attributes &query_attrs,
                                        size_t limit) {
  auto tags = dispatch([&](auto *p) { return p->filter_only(selector, query_attrs, limit); });
  py::array_t<uint32_t> arr(tags.size());
  std::memcpy(arr.mutable_data(), tags.data(), tags.size() * sizeof(uint32_t));
  return arr;
}

std::string PyIndexInterface::to_string() const {
  return dispatch([](auto *p) { return p->to_string(); });
}
