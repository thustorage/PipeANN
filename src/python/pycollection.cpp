// pybind layer for the neutral CollectionStore engine. Exposes the same data
// model the gRPC server uses, so the Python MilvusClient is a thin wrapper:
// row<->column shaping in Python, all schema/filter/projection logic in C++.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "collection_store.h"

namespace py = pybind11;
using namespace pipeann::server;

namespace {

FieldKind kind_from_str(const std::string &s) {
  if (s == "int64" || s == "int") return FieldKind::Int64;
  if (s == "float" || s == "double") return FieldKind::Float;
  if (s == "bool") return FieldKind::Bool;
  if (s == "varchar" || s == "string") return FieldKind::VarChar;
  if (s == "array" || s == "array_int") return FieldKind::ArrayInt;
  if (s == "float_vector" || s == "vector") return FieldKind::FloatVector;
  return FieldKind::Json;
}

const char *kind_to_str(FieldKind k) {
  switch (k) {
    case FieldKind::Int64: return "int64";
    case FieldKind::Float: return "float";
    case FieldKind::Bool: return "bool";
    case FieldKind::VarChar: return "varchar";
    case FieldKind::ArrayInt: return "array_int";
    case FieldKind::FloatVector: return "float_vector";
    case FieldKind::Json: return "json";
  }
  return "int64";
}

// Build a ScalarColumn from a Python list, dispatching on the declared kind.
ScalarColumn scalar_column_from_py(FieldKind kind, const py::list &values) {
  ScalarColumn col;
  col.kind = kind;
  switch (kind) {
    case FieldKind::Float:
      for (auto v : values) col.doubles.push_back(v.cast<double>());
      break;
    case FieldKind::Bool:
      for (auto v : values) col.bools.push_back(v.cast<bool>());
      break;
    case FieldKind::VarChar:
    case FieldKind::Json:
      for (auto v : values) col.strings.push_back(v.cast<std::string>());
      break;
    case FieldKind::ArrayInt:
      for (auto v : values) col.int_arrays.push_back(v.cast<std::vector<int64_t>>());
      break;
    case FieldKind::Int64:
    default:
      for (auto v : values) col.ints.push_back(v.cast<int64_t>());
      break;
  }
  return col;
}

// Convert one OutputColumn into a Python list of per-row values.
py::list scalar_column_to_py(const OutputColumn &oc) {
  py::list out;
  switch (oc.kind) {
    case FieldKind::FloatVector: {
      size_t dim = oc.dim ? oc.dim : 1;
      size_t n = oc.vectors.size() / dim;
      for (size_t i = 0; i < n; i++) {
        py::list row;
        for (size_t d = 0; d < dim; d++) row.append(oc.vectors[i * dim + d]);
        out.append(std::move(row));
      }
      break;
    }
    case FieldKind::Float:
      for (double v : oc.scalar.doubles) out.append(v);
      break;
    case FieldKind::Bool:
      for (bool v : oc.scalar.bools) out.append(v);
      break;
    case FieldKind::VarChar:
    case FieldKind::Json:
      for (const auto &s : oc.scalar.strings) out.append(s);
      break;
    case FieldKind::ArrayInt:
      for (const auto &a : oc.scalar.int_arrays) out.append(py::cast(a));
      break;
    case FieldKind::Int64:
    default:
      for (int64_t v : oc.scalar.ints) out.append(v);
      break;
  }
  return out;
}

// QueryResult -> a Python dict: {ids, scores, topks, columns:{name:[...]},
// output_fields:[...], primary_field}.
py::dict query_result_to_py(const QueryResult &qr) {
  py::dict d;
  d["ids"] = py::cast(qr.ids);
  d["scores"] = py::cast(qr.scores);
  d["topks"] = py::cast(qr.topks);
  py::dict cols;
  for (const auto &oc : qr.columns) cols[py::str(oc.name)] = scalar_column_to_py(oc);
  d["columns"] = cols;
  d["output_fields"] = py::cast(qr.output_field_names);
  d["primary_field"] = py::str(qr.primary_field_name);
  return d;
}

}  // namespace

void bind_collection_store(py::module_ &m) {
  py::class_<CollectionStore>(m, "CollectionStore")
      .def(py::init<const std::string &, int>(), py::arg("data_dir"), py::arg("omp_threads") = 0)

      .def(
          "create_collection",
          // fields: list of (name, kind_str, is_primary, dim). metric: str.
          [](CollectionStore &self, const std::string &name, const py::list &fields, const std::string &metric) {
            CollectionSpec spec;
            spec.name = name;
            spec.metric = metric;
            for (auto item : fields) {
              auto t = item.cast<py::tuple>();
              FieldSpec fs;
              fs.name = t[0].cast<std::string>();
              fs.kind = kind_from_str(t[1].cast<std::string>());
              fs.is_primary = t[2].cast<bool>();
              fs.dim = t.size() > 3 ? t[3].cast<uint32_t>() : 0;
              spec.fields.push_back(fs);
            }
            py::gil_scoped_release rel;
            self.create_collection(spec);
          },
          py::arg("name"), py::arg("fields"), py::arg("metric") = "l2")

      .def("drop_collection", &CollectionStore::drop_collection, py::arg("name"),
           py::call_guard<py::gil_scoped_release>())
      .def("has_collection", &CollectionStore::has_collection, py::arg("name"))
      .def("list_collections", &CollectionStore::list_collections)
      .def("row_count", &CollectionStore::row_count, py::arg("name"))
      .def("build_index", &CollectionStore::build_index, py::arg("name"), py::arg("build_R") = 0,
           py::arg("build_L") = 0, py::call_guard<py::gil_scoped_release>())
      .def("index_built", &CollectionStore::index_built, py::arg("name"))
      .def("count", &CollectionStore::count, py::arg("name"), py::arg("filter") = std::string(),
           py::call_guard<py::gil_scoped_release>())
      .def("flush", &CollectionStore::flush, py::arg("name") = std::string(),
           py::call_guard<py::gil_scoped_release>())

      .def(
          "describe",
          [](CollectionStore &self, const std::string &name) -> py::object {
            auto meta = self.describe(name);
            if (!meta) return py::none();
            py::dict d;
            d["name"] = meta->name;
            d["dim"] = meta->dim;
            d["metric"] = meta->metric;
            d["primary_field"] = meta->primary_field;
            d["vector_field"] = meta->vector_field;
            py::list fields;
            for (const auto &f : meta->fields) {
              fields.append(py::make_tuple(f.name, std::string(kind_to_str(f.kind)), f.is_primary, f.dim));
            }
            d["fields"] = fields;
            return d;
          },
          py::arg("name"))

      .def(
          "insert",
          [](CollectionStore &self, const std::string &name, uint32_t num_rows, const std::vector<std::string> &pks,
             py::array_t<float, py::array::c_style | py::array::forcecast> vectors, const py::dict &scalars,
             const py::dict &kinds, bool is_upsert) {
            InsertColumns cols;
            cols.num_rows = num_rows;
            cols.pks = pks;
            auto buf = vectors.request();
            cols.vectors.assign(static_cast<float *>(buf.ptr),
                                static_cast<float *>(buf.ptr) + buf.size);
            for (auto item : scalars) {
              std::string fname = item.first.cast<std::string>();
              FieldKind kind = kind_from_str(kinds[item.first].cast<std::string>());
              cols.scalars[fname] = scalar_column_from_py(kind, item.second.cast<py::list>());
            }
            InsertResult res;
            {
              py::gil_scoped_release rel;
              res = is_upsert ? self.upsert(name, cols) : self.insert(name, cols);
            }
            py::dict d;
            d["count"] = res.count;
            d["ids"] = py::cast(res.ids);
            return d;
          },
          py::arg("name"), py::arg("num_rows"), py::arg("pks"), py::arg("vectors"), py::arg("scalars"),
          py::arg("kinds"), py::arg("is_upsert") = false)

      .def(
          "search",
          [](CollectionStore &self, const std::string &name, int n_queries, int topk, int L,
             py::array_t<float, py::array::c_style | py::array::forcecast> queries, const std::string &filter,
             const std::vector<std::string> &output_fields) {
            SearchParams params;
            params.n_queries = n_queries;
            params.topk = topk;
            params.L = L;
            params.filter = filter;
            params.output_fields = output_fields;
            auto buf = queries.request();
            params.queries.assign(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.size);
            QueryResult qr;
            {
              py::gil_scoped_release rel;
              qr = self.search(name, params);
            }
            return query_result_to_py(qr);
          },
          py::arg("name"), py::arg("n_queries"), py::arg("topk"), py::arg("L"), py::arg("queries"),
          py::arg("filter") = std::string(), py::arg("output_fields") = std::vector<std::string>())

      .def(
          "query",
          [](CollectionStore &self, const std::string &name, const std::string &filter, size_t limit,
             const std::vector<std::string> &output_fields) {
            QueryParams params;
            params.filter = filter;
            params.limit = limit;
            params.output_fields = output_fields;
            QueryResult qr;
            {
              py::gil_scoped_release rel;
              qr = self.query(name, params);
            }
            return query_result_to_py(qr);
          },
          py::arg("name"), py::arg("filter") = std::string(), py::arg("limit") = 0,
          py::arg("output_fields") = std::vector<std::string>())

      .def(
          "remove",
          [](CollectionStore &self, const std::string &name, const std::vector<std::string> &pks,
             const std::string &filter) {
            py::gil_scoped_release rel;
            return self.remove(name, pks, filter);
          },
          py::arg("name"), py::arg("pks") = std::vector<std::string>(), py::arg("filter") = std::string());
}
