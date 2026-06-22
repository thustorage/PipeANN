#include "pyindex.h"
#include "dynamic_index.h"
#include "doc_store.h"  // from src/server/
#include "filter/selector.h"
#include <limits>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<pipeann::Attributes>);

namespace {
  pipeann::Attributes attrs_from_dict(const py::dict &data) {
    pipeann::Attributes attrs;
    for (auto item : data) {
      uint32_t key = item.first.cast<uint32_t>();
      auto vals = item.second.cast<std::vector<uint32_t>>();
      attrs.set(key, vals);
    }
    return attrs;
  }

  template<typename SelectorT>
  SelectorT *make_native_children_selector(const py::args &children) {
    std::vector<pipeann::Selector *> copied_children;
    copied_children.reserve(children.size());
    for (auto child : children) {
      copied_children.push_back(child.cast<pipeann::Selector *>()->copy());
    }
    return new SelectorT(std::move(copied_children));
  }

  struct PySelector : pipeann::Selector {
    using pipeann::Selector::Selector;
    // Bound lazily for the trampoline object created by pybind11, and already
    // set for the deep-copied selector returned by copy().
    mutable py::object self_ = py::none();

    PySelector() = default;

    explicit PySelector(py::object self) : self_(std::move(self)) {
    }

    ~PySelector() override {
      if (!self_.is_none()) {
        py::gil_scoped_acquire gil;
        self_ = py::none();
      }
    }

    py::object python_self() const {
      if (self_.is_none()) {
        self_ = py::cast(const_cast<PySelector *>(this), py::return_value_policy::reference);
      }
      return self_;
    }

    pipeann::Selector *copy() const override {
      py::gil_scoped_acquire gil;
      py::object copied = python_self().attr("__copy__")();
      return new PySelector(copied);
    }

    double estimate_selectivity(const pipeann::Attributes &query_attrs) override {
      py::gil_scoped_acquire gil;
      return python_self().attr("estimate_selectivity")(query_attrs).cast<double>();
    }

    double estimate_precision(const pipeann::Attributes &query_attrs) override {
      py::gil_scoped_acquire gil;
      return python_self().attr("estimate_precision")(query_attrs).cast<double>();
    }

    uint32_t estimate_prefilter_reads(const pipeann::Attributes &query_attrs) override {
      py::gil_scoped_acquire gil;
      return python_self().attr("estimate_prefilter_reads")(query_attrs).cast<uint32_t>();
    }

    pipeann::VectorIDList pre_filter(const pipeann::Attributes &query_attrs, AlignedFileReader *, bool /*strict*/ = false) override {
      py::gil_scoped_acquire gil;
      return python_self().attr("pre_filter")(query_attrs).cast<pipeann::VectorIDList>();
    }

    bool is_member(uint32_t target_id, const pipeann::Attributes &query_attrs,
                   const pipeann::Attributes &target_attrs) override {
      py::gil_scoped_acquire gil;
      return python_self().attr("is_member")(target_id, query_attrs, target_attrs).cast<bool>();
    }

    uint32_t estimate_infilter_reads(const pipeann::Attributes &query_attrs) override {
      py::gil_scoped_acquire gil;
      return python_self().attr("estimate_infilter_reads")(query_attrs).cast<uint32_t>();
    }

    void prepare_in_filter(const pipeann::Attributes &query_attrs, AlignedFileReader *) override {
      py::gil_scoped_acquire gil;
      python_self().attr("prepare_in_filter")(query_attrs);
    }

    bool is_member_approx(uint32_t target_id, const pipeann::Attributes &query_attrs) override {
      py::gil_scoped_acquire gil;
      return python_self().attr("is_member_approx")(target_id, query_attrs).cast<bool>();
    }
  };
}  // namespace

// Defined in pycollection.cpp: binds the neutral CollectionStore engine.
void bind_collection_store(py::module_ &m);

PYBIND11_MODULE(C, m) {
  m.doc() = "PipeANN";
  m.attr("__version__") = "dev";

  bind_collection_store(m);

  m.def("_save_attr_index_from_rows", &pipeann::save_attr_index_from_rows, py::arg("rows"), py::arg("file_out"),
        py::arg("attr_type"));

  // RocksDB-backed id<->tag<->document store (shared with the C++ gRPC server).
  py::class_<pipeann::DocStore>(m, "DocStore")
      .def(py::init<>())
      .def("open", &pipeann::DocStore::open, py::arg("dir"), py::call_guard<py::gil_scoped_release>())
      .def("close", &pipeann::DocStore::close, py::call_guard<py::gil_scoped_release>())
      .def("is_open", &pipeann::DocStore::is_open)
      .def("put_batch", &pipeann::DocStore::put_batch, py::arg("rows"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "get_by_tags",
          [](const pipeann::DocStore &self, const std::vector<uint32_t> &tags) {
            std::vector<std::string> ids, docs;
            {
              py::gil_scoped_release release;
              self.get_by_tags(tags, ids, docs);
            }
            return py::make_tuple(std::move(ids), std::move(docs));
          },
          py::arg("tags"))
      .def(
          "get_tags_by_ids",
          [](const pipeann::DocStore &self, const std::vector<std::string> &ids) {
            std::vector<int64_t> tags;
            {
              py::gil_scoped_release release;
              self.get_tags_by_ids(ids, tags);
            }
            return tags;
          },
          py::arg("ids"))
      .def(
          "get_docs_by_ids",
          [](const pipeann::DocStore &self, const std::vector<std::string> &ids) {
            std::vector<std::string> out_ids, docs;
            {
              py::gil_scoped_release release;
              self.get_docs_by_ids(ids, out_ids, docs);
            }
            return py::make_tuple(std::move(out_ids), std::move(docs));
          },
          py::arg("ids"))
      .def("delete_by_ids", &pipeann::DocStore::delete_by_ids, py::arg("ids"),
           py::call_guard<py::gil_scoped_release>())
      .def("max_tag", &pipeann::DocStore::max_tag, py::call_guard<py::gil_scoped_release>())
      .def(
          "scan",
          [](const pipeann::DocStore &self) {
            std::vector<uint32_t> tags;
            std::vector<std::string> ids, docs;
            {
              py::gil_scoped_release release;
              self.scan(tags, ids, docs);
            }
            return py::make_tuple(std::move(tags), std::move(ids), std::move(docs));
          })
      .def("clear", &pipeann::DocStore::clear, py::call_guard<py::gil_scoped_release>())
      .def("flush", &pipeann::DocStore::flush, py::call_guard<py::gil_scoped_release>());

  py::enum_<pipeann::Metric>(m, "Metric")
      .value("L2", pipeann::Metric::L2)
      .value("INNER_PRODUCT", pipeann::Metric::INNER_PRODUCT)
      .value("COSINE", pipeann::Metric::COSINE)
      .export_values();

  py::class_<pipeann::Attributes>(m, "Attributes")
      .def(py::init<>())
      .def(py::init([](py::dict d) { return attrs_from_dict(d); }), py::arg("data"))
      .def("find", &pipeann::Attributes::find, py::arg("key"))
      .def("get", &pipeann::Attributes::get, py::arg("key"))
      .def("set", &pipeann::Attributes::set, py::arg("key"), py::arg("values"))
      .def("remove", &pipeann::Attributes::remove, py::arg("key"))
      .def("clear", &pipeann::Attributes::clear)
      .def("to_dict",
           [](const pipeann::Attributes &self) {
             py::dict d;
             for (const auto &[key, vals] : self.attrs_) {
               d[py::int_(key)] = py::cast(vals);
             }
             return d;
           })
      .def("__repr__", [](const pipeann::Attributes &self) {
        std::string s = "Attributes({";
        bool first = true;
        for (const auto &[key, vals] : self.attrs_) {
          if (!first)
            s += ", ";
          s += std::to_string(key) + ": [";
          for (size_t i = 0; i < vals.size(); i++) {
            if (i)
              s += ", ";
            s += std::to_string(vals[i]);
          }
          s += "]";
          first = false;
        }
        return s + "})";
      });

  py::bind_vector<std::vector<pipeann::Attributes>>(m, "NativeAttrsVec")
      .def(py::init<>())
      .def(
          "load_from_file",
          [](std::vector<pipeann::Attributes> &self, uint32_t key, const std::string &attr_type,
             const std::string &filename) {
            std::vector<pipeann::Attributes> loaded;
            pipeann::load_query_attrs_from_file(filename, attr_type, key, loaded);
            if (self.empty()) {
              self = std::move(loaded);
              return;
            }
            for (size_t i = 0; i < self.size(); i++) {
              self[i].set(key, loaded[i].get(key));
            }
          },
          py::arg("key"), py::arg("attr_type"), py::arg("filename"));

  py::class_<pipeann::Selector, PySelector>(m, "Selector")
      .def(py::init<>())
      .def(
          "copy", [](pipeann::Selector &self) { return self.copy(); }, py::return_value_policy::take_ownership)
      .def("__copy__",
           [](py::object self) {
             py::object cls = py::reinterpret_borrow<py::object>(py::type::of(self));
             py::object copied = cls.attr("__new__")(cls);
             if (py::hasattr(self, "__dict__")) {
               copied.attr("__dict__").attr("update")(self.attr("__dict__"));
             }
             return copied;
           })
      .def("estimate_selectivity", &pipeann::Selector::estimate_selectivity, py::arg("query_attrs"))
      .def("estimate_precision", &pipeann::Selector::estimate_precision, py::arg("query_attrs"))
      .def("estimate_prefilter_reads", &pipeann::Selector::estimate_prefilter_reads, py::arg("query_attrs"))
      .def("estimate_infilter_reads", &pipeann::Selector::estimate_infilter_reads, py::arg("query_attrs"))
      .def(
          "pre_filter",
          [](pipeann::Selector &self, const pipeann::Attributes &query_attrs) {
            return self.pre_filter(query_attrs, nullptr);
          },
          py::arg("query_attrs"))
      .def(
          "prepare_in_filter",
          [](pipeann::Selector &self, const pipeann::Attributes &query_attrs) {
            self.prepare_in_filter(query_attrs, nullptr);
          },
          py::arg("query_attrs"))
      .def("is_member", &pipeann::Selector::is_member, py::arg("target_id"), py::arg("query_attrs"),
           py::arg("target_attrs"))
      .def("is_member_approx", &pipeann::Selector::is_member_approx, py::arg("target_id"), py::arg("query_attrs"));

  py::class_<pipeann::AttrIndex, std::shared_ptr<pipeann::AttrIndex>>(m, "NativeAttrIndex")
      .def_property_readonly("attr_type",
                             [](const pipeann::AttrIndex &self) { return pipeann::native_attr_type(&self); })
      .def_property_readonly("n_vectors", [](const pipeann::AttrIndex &self) { return self.n_vectors.load(); });

  py::class_<pipeann::LabelOrSelector, pipeann::Selector>(m, "LabelOrSelector")
      .def(py::init([](uint32_t key, uint32_t base_key, const std::shared_ptr<pipeann::AttrIndex> &attr_index) {
             return new pipeann::LabelOrSelector(key, base_key, attr_index.get());
           }),
           py::arg("key"), py::arg("base_key"), py::arg("attr_index"));

  py::class_<pipeann::LabelAndSelector, pipeann::Selector>(m, "LabelAndSelector")
      .def(py::init([](uint32_t key, uint32_t base_key, const std::shared_ptr<pipeann::AttrIndex> &attr_index) {
             return new pipeann::LabelAndSelector(key, base_key, attr_index.get());
           }),
           py::arg("key"), py::arg("base_key"), py::arg("attr_index"));

  py::class_<pipeann::RangeSelector, pipeann::Selector>(m, "RangeSelector")
      .def(py::init([](uint32_t key, uint32_t base_key, const std::shared_ptr<pipeann::AttrIndex> &attr_index) {
             return new pipeann::RangeSelector(key, base_key, attr_index.get());
           }),
           py::arg("key"), py::arg("base_key"), py::arg("attr_index"));

  py::class_<pipeann::StringEqSelector, pipeann::Selector>(m, "StringEqSelector")
      .def(py::init([](uint32_t key, uint32_t base_key, const std::shared_ptr<pipeann::AttrIndex> &attr_index) {
             return new pipeann::StringEqSelector(key, base_key, attr_index.get());
           }),
           py::arg("key"), py::arg("base_key"), py::arg("attr_index"));

  py::class_<pipeann::StringPrefixSelector, pipeann::Selector>(m, "StringPrefixSelector")
      .def(py::init([](uint32_t key, uint32_t base_key, const std::shared_ptr<pipeann::AttrIndex> &attr_index) {
             return new pipeann::StringPrefixSelector(key, base_key, attr_index.get());
           }),
           py::arg("key"), py::arg("base_key"), py::arg("attr_index"));

  py::class_<pipeann::StringSuffixSelector, pipeann::Selector>(m, "StringSuffixSelector")
      .def(py::init([](uint32_t key, uint32_t base_key, const std::shared_ptr<pipeann::AttrIndex> &attr_index) {
             return new pipeann::StringSuffixSelector(key, base_key, attr_index.get());
           }),
           py::arg("key"), py::arg("base_key"), py::arg("attr_index"));

  py::class_<pipeann::StringLikeSelector, pipeann::Selector>(m, "StringLikeSelector")
      .def(py::init([](uint32_t key, uint32_t base_key, const std::shared_ptr<pipeann::AttrIndex> &attr_index) {
             return new pipeann::StringLikeSelector(key, base_key, attr_index.get());
           }),
           py::arg("key"), py::arg("base_key"), py::arg("attr_index"));

  py::class_<pipeann::AndSelector, pipeann::Selector>(m, "AndSelector").def(py::init([](py::args children) {
    return make_native_children_selector<pipeann::AndSelector>(children);
  }));

  py::class_<pipeann::OrSelector, pipeann::Selector>(m, "OrSelector").def(py::init([](py::args children) {
    return make_native_children_selector<pipeann::OrSelector>(children);
  }));

  py::class_<pipeann::NotSelector, pipeann::Selector>(m, "NotSelector")
      .def(py::init([](pipeann::Selector *child, uint32_t n_vectors) {
             return new pipeann::NotSelector(child->copy(), n_vectors);
           }),
           py::arg("child"), py::arg("n_vectors"));

  py::class_<PyIndexInterface>(m, "PyIndex")
      .def(py::init([](uint32_t data_dim, const std::string &data_type, pipeann::Metric metric, uint32_t attr_size) {
             return new PyIndexInterface(data_dim, data_type, metric, nullptr, attr_size);
           }),
           py::arg("data_dim"), py::arg("data_type"), py::arg("metric"), py::arg("attr_size") = 0)
      .def("load", &PyIndexInterface::load, py::arg("index_prefix"))
      .def("save", &PyIndexInterface::save, py::arg("index_prefix"))
      .def("build", &PyIndexInterface::build, py::arg("data_path"), py::arg("index_prefix"),
           py::arg("tag_file") = nullptr, py::arg("build_mem_index") = false, py::arg("max_nbrs") = 0,
           py::arg("build_L") = 0, py::arg("PQ_bytes") = 0, py::arg("memory_use_GB") = 0,
           py::arg("attrs_vec") = (std::vector<pipeann::Attributes> *) nullptr, py::arg("range_dense") = 0,
           py::arg("train_query_path") = std::string(), py::arg("R_ood") = 0, py::arg("L_ood") = 1500)
      .def("load_attr_index_from_file", &PyIndexInterface::load_attr_index_from_file, py::arg("key"),
           py::arg("filename"), py::arg("attr_type"))
      .def("create_attr_index", &PyIndexInterface::create_attr_index, py::arg("key"), py::arg("filename"),
           py::arg("attr_type"))
      .def(
          "compile_filter",
          [](PyIndexInterface &self, const std::string &json, const py::dict &schema) {
            auto cf = self.compile_filter(json, schema);
            py::dict slot_map;
            for (const auto &[name, slot] : cf.slot_map) slot_map[py::str(name)] = slot;
            py::dict var_field_type;
            for (const auto &[name, t] : cf.var_field_type) var_field_type[py::str(name)] = t;
            return py::make_tuple(py::cast(cf.selector, py::return_value_policy::take_ownership),
                                  std::move(cf.attrs_template), slot_map, var_field_type);
          },
          py::arg("json"), py::arg("schema"))
      .def(
          "load_filter_from_json",
          [](PyIndexInterface &self, const std::string &config_path) {
            auto [selector, attrs_vec] = self.load_filter_from_json(config_path);
            return py::make_tuple(py::cast(selector, py::return_value_policy::take_ownership),
                                  std::move(attrs_vec));
          },
          py::arg("config_path"))
      .def("filter_only", &PyIndexInterface::filter_only, py::arg("selector"), py::arg("query_attrs"),
           py::arg("limit") = 0)
      .def("search", &PyIndexInterface::search, py::arg("queries"), py::arg("topk"), py::arg("L"),
           py::arg("selector") = (pipeann::Selector *) nullptr,
           py::arg("query_attrs") = std::vector<pipeann::Attributes>(),
           py::arg("range") = std::numeric_limits<float>::infinity())
      .def("add", &PyIndexInterface::add, py::arg("vectors"), py::arg("tags"),
           py::arg("attrs_vec") = (std::vector<pipeann::Attributes> *) nullptr)
      .def("remove", &PyIndexInterface::remove, py::arg("tags"))
      .def("set_index_prefix", &PyIndexInterface::set_index_prefix, py::arg("index_prefix"))
      .def("omp_set_num_threads", &PyIndexInterface::omp_set_num_threads, py::arg("num_threads"))
      .def("npoints", &PyIndexInterface::npoints)
      .def("__repr__", &PyIndexInterface::to_string);
}
