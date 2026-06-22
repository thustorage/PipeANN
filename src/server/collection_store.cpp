#include "collection_store.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>

#include "utils/picojson.h"

namespace pipeann {
namespace server {

// ---------------------------------------------------------------------------
// Field-kind helpers
// ---------------------------------------------------------------------------
static const char *attr_type_for(FieldKind k) {
  switch (k) {
    case FieldKind::ArrayInt: return "label";
    case FieldKind::VarChar: return "string";
    case FieldKind::Int64:
    case FieldKind::Float:
    case FieldKind::Bool: return "range";
    default: return "";  // not filterable
  }
}

static const char *logical_type_for(FieldKind k) {
  switch (k) {
    case FieldKind::ArrayInt: return "label";
    case FieldKind::VarChar: return "string";
    case FieldKind::Float: return "float";
    case FieldKind::Bool: return "bool";
    case FieldKind::Int64: return "int";
    default: return "";
  }
}

static FieldKind kind_from_logical(const std::string &logical) {
  if (logical == "label") return FieldKind::ArrayInt;
  if (logical == "string") return FieldKind::VarChar;
  if (logical == "float") return FieldKind::Float;
  if (logical == "bool") return FieldKind::Bool;
  return FieldKind::Int64;
}

static pipeann::Metric metric_from_string(const std::string &m) {
  if (m == "inner_product" || m == "ip") return pipeann::Metric::INNER_PRODUCT;
  if (m == "cosine") return pipeann::Metric::COSINE;
  return pipeann::Metric::L2;
}

static std::string metric_to_string(pipeann::Metric m) {
  switch (m) {
    case pipeann::Metric::INNER_PRODUCT: return "inner_product";
    case pipeann::Metric::COSINE: return "cosine";
    default: return "l2";
  }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
CollectionStore::CollectionStore(const std::string &data_dir, int omp_threads)
    : data_dir_(data_dir), omp_threads_(omp_threads) {
  int pool_size = omp_threads > 0 ? omp_threads : kDefaultWorkerThreads;
  search_pool_ = std::make_unique<SearchWorkerPool>(pool_size);
  load_existing_collections();
}

CollectionStore::~CollectionStore() = default;

CollectionStore::Collection *CollectionStore::get(const std::string &name) {
  std::lock_guard<std::mutex> lk(collections_mu_);
  auto it = collections_.find(name);
  return it != collections_.end() ? it->second.get() : nullptr;
}

// ---------------------------------------------------------------------------
// Collection lifecycle
// ---------------------------------------------------------------------------
void CollectionStore::create_collection(const CollectionSpec &spec) {
  CollectionMeta meta;
  meta.name = spec.name;
  meta.metric = spec.metric;
  uint32_t next_attr_key = 0;
  for (const auto &f : spec.fields) {
    if (f.is_primary) meta.primary_field = f.name;
    if (f.kind == FieldKind::FloatVector) {
      meta.vector_field = f.name;
      meta.dim = f.dim;
    } else if (!f.is_primary) {
      const char *at = attr_type_for(f.kind);
      if (at[0] != '\0') {
        ScalarFieldMeta sfm;
        sfm.name = f.name;
        sfm.key = next_attr_key++;
        sfm.attr_type = at;
        sfm.logical_type = logical_type_for(f.kind);
        meta.scalar_fields.push_back(sfm);
      } else {
        // JSON / unsupported-for-filtering: display-only.
        meta.display_fields.push_back(f.name);
      }
    }
    meta.fields.push_back(f);
  }

  if (meta.dim == 0) throw std::runtime_error("no FloatVector field with dim found");

  auto col = std::make_unique<Collection>();
  col->meta = std::move(meta);
  pipeann::IndexBuildParameters build_params;
  build_params.num_threads = omp_threads_ > 0 ? omp_threads_ : 0;
  build_params.max_nthreads = omp_threads_ > 0 ? omp_threads_ : kDefaultWorkerThreads;
  col->index = std::make_unique<DynamicIndex<float>>(col->meta.dim, metric_from_string(col->meta.metric), &build_params,
                                                     /*attr_size=*/0, /*enable_tag2id=*/true);
  if (omp_threads_ > 0) col->index->omp_set_num_threads(omp_threads_);

  std::string index_prefix = data_dir_ + "/" + spec.name;
  col->index->set_index_prefix(index_prefix);
  col->index_prefix = index_prefix;
  if (col->doc_store.open(index_prefix + ".docs.rocksdb")) {
    col->doc_store.clear();
  }

  std::lock_guard<std::mutex> lk(collections_mu_);
  collections_[spec.name] = std::move(col);
}

void CollectionStore::drop_collection(const std::string &name) {
  std::unique_ptr<Collection> removed;
  {
    std::lock_guard<std::mutex> lk(collections_mu_);
    auto it = collections_.find(name);
    if (it != collections_.end()) {
      removed = std::move(it->second);
      collections_.erase(it);
    }
  }
  if (removed) {
    removed->doc_store.close();
    std::error_code ec;
    std::filesystem::remove_all(removed->index_prefix + ".docs.rocksdb", ec);
  }
}

bool CollectionStore::has_collection(const std::string &name) { return get(name) != nullptr; }

std::vector<std::string> CollectionStore::list_collections() {
  std::lock_guard<std::mutex> lk(collections_mu_);
  std::vector<std::string> out;
  out.reserve(collections_.size());
  for (auto &[n, _] : collections_) out.push_back(n);
  return out;
}

std::optional<CollectionMeta> CollectionStore::describe(const std::string &name) {
  auto *col = get(name);
  if (!col) return std::nullopt;
  return col->meta;
}

int64_t CollectionStore::row_count(const std::string &name) {
  auto *col = get(name);
  if (!col) return 0;
  std::lock_guard<std::mutex> lk(col->mu);
  return col->index_built ? (int64_t) col->index->npoints() : (int64_t) col->buf_tags.size();
}

bool CollectionStore::index_built(const std::string &name) {
  auto *col = get(name);
  if (!col) return false;
  std::lock_guard<std::mutex> lk(col->mu);
  return col->index_built;
}

void CollectionStore::build_index(const std::string &name, uint32_t build_R, uint32_t build_L) {
  auto *col = get(name);
  if (!col) throw std::runtime_error("collection not found");
  build_index_locked(col, build_R, build_L);
}

void CollectionStore::build_index_locked(Collection *col, uint32_t build_R, uint32_t build_L) {
  std::lock_guard<std::mutex> lk(col->mu);
  if (col->index_built) return;
  if (col->buf_tags.empty()) {
    col->index_built = true;
    return;
  }
  uint64_t n = col->buf_tags.size();
  std::map<uint32_t, std::string> field_types;
  for (auto &sf : col->meta.scalar_fields) field_types[sf.key] = sf.attr_type;
  bool has_scalar = !field_types.empty();

  col->index->build_from_buffer(col->buf_vectors.data(), col->buf_tags.data(), n,
                                has_scalar ? &col->buf_attrs : nullptr, field_types, build_R, build_L);
  col->index_built = true;
  std::vector<float>().swap(col->buf_vectors);
  std::vector<uint32_t>().swap(col->buf_tags);
  std::vector<pipeann::Attributes>().swap(col->buf_attrs);

  write_schema_json(col);
  col->doc_store.flush();
}

// ---------------------------------------------------------------------------
// Insert / encode
// ---------------------------------------------------------------------------
void CollectionStore::encode_attrs_and_docs(Collection *col, const InsertColumns &cols,
                                            std::vector<pipeann::Attributes> *attrs_vec,
                                            std::vector<std::string> *docs) {
  uint32_t n = cols.num_rows;
  // Filterable scalar attributes.
  if (!col->meta.scalar_fields.empty()) {
    attrs_vec->resize(n);
    for (const auto &sf : col->meta.scalar_fields) {
      auto it = cols.scalars.find(sf.name);
      if (it == cols.scalars.end()) continue;
      const ScalarColumn &sc = it->second;
      LogicalType ltype = logical_type_from_string(sf.logical_type);
      switch (ltype) {
        case LogicalType::Label:
          for (uint32_t i = 0; i < n && i < sc.int_arrays.size(); i++) {
            pipeann::Attribute labels;
            for (int64_t v : sc.int_arrays[i]) labels.push_back(static_cast<uint32_t>(v));
            (*attrs_vec)[i].set(sf.key, std::move(labels));
          }
          break;
        case LogicalType::Float:
          for (uint32_t i = 0; i < n && i < sc.doubles.size(); i++)
            (*attrs_vec)[i].set(sf.key, encode_float_range(sc.doubles[i]));
          break;
        case LogicalType::Bool:
          for (uint32_t i = 0; i < n && i < sc.bools.size(); i++)
            (*attrs_vec)[i].set(sf.key, encode_bool_range(sc.bools[i]));
          break;
        case LogicalType::String:
          for (uint32_t i = 0; i < n && i < sc.strings.size(); i++)
            (*attrs_vec)[i].set(sf.key, pipeann::pack_string_attr(sc.strings[i]));
          break;
        case LogicalType::Int:
        default:
          for (uint32_t i = 0; i < n && i < sc.ints.size(); i++)
            (*attrs_vec)[i].set(sf.key, encode_int_range(sc.ints[i]));
          break;
      }
    }
  }

  // Display-only fields -> per-row JSON document.
  if (!col->meta.display_fields.empty()) {
    std::vector<picojson::object> row_objs(n);
    for (const auto &fname : col->meta.display_fields) {
      auto it = cols.scalars.find(fname);
      if (it == cols.scalars.end()) continue;
      const ScalarColumn &sc = it->second;
      switch (sc.kind) {
        case FieldKind::Int64:
          for (uint32_t i = 0; i < n && i < sc.ints.size(); i++)
            row_objs[i][fname] = picojson::value((double) sc.ints[i]);
          break;
        case FieldKind::Float:
          for (uint32_t i = 0; i < n && i < sc.doubles.size(); i++)
            row_objs[i][fname] = picojson::value(sc.doubles[i]);
          break;
        case FieldKind::Bool:
          for (uint32_t i = 0; i < n && i < sc.bools.size(); i++)
            row_objs[i][fname] = picojson::value((bool) sc.bools[i]);
          break;
        default:  // VarChar / Json
          for (uint32_t i = 0; i < n && i < sc.strings.size(); i++)
            row_objs[i][fname] = picojson::value(sc.strings[i]);
          break;
      }
    }
    docs->resize(n);
    for (uint32_t i = 0; i < n; i++) {
      if (!row_objs[i].empty()) (*docs)[i] = picojson::value(row_objs[i]).serialize();
    }
  }
}

InsertResult CollectionStore::insert(const std::string &name, const InsertColumns &cols) {
  auto *col = get(name);
  if (!col) throw std::runtime_error("collection not found");

  uint32_t n = cols.num_rows;
  uint32_t dim = col->meta.dim;
  if (cols.vectors.size() != (size_t) n * dim) throw std::runtime_error("vector count mismatch");

  std::vector<pipeann::Attributes> attrs_vec;
  std::vector<std::string> docs;
  encode_attrs_and_docs(col, cols, &attrs_vec, &docs);
  bool has_scalar = !attrs_vec.empty();
  docs.resize(n);

  std::vector<uint32_t> tags(n);
  bool built;
  {
    std::lock_guard<std::mutex> lk(col->mu);
    built = col->index_built;
    for (uint32_t i = 0; i < n; i++) tags[i] = col->next_tag++;
    if (!built) {
      col->buf_tags.insert(col->buf_tags.end(), tags.begin(), tags.end());
      col->buf_vectors.insert(col->buf_vectors.end(), cols.vectors.begin(), cols.vectors.end());
      if (has_scalar)
        for (auto &a : attrs_vec) col->buf_attrs.push_back(std::move(a));
    }
  }

  if (col->doc_store.is_open()) {
    std::vector<std::tuple<uint32_t, std::string, std::string>> rows;
    rows.reserve(n);
    for (uint32_t i = 0; i < n; i++) rows.emplace_back(tags[i], cols.pks[i], std::move(docs[i]));
    col->doc_store.put_batch(rows);
  }

  if (built) {
    col->index->add(cols.vectors.data(), tags.data(), n, has_scalar ? &attrs_vec : nullptr);
  }

  InsertResult res;
  res.ids = cols.pks;
  res.count = n;
  return res;
}

InsertResult CollectionStore::upsert(const std::string &name, const InsertColumns &cols) {
  auto *col = get(name);
  if (!col) throw std::runtime_error("collection not found");
  delete_by_pks(col, cols.pks);
  InsertResult res = insert(name, cols);
  return res;
}

// ---------------------------------------------------------------------------
// Filter compilation
// ---------------------------------------------------------------------------
pipeann::dsl::Schema CollectionStore::build_dsl_schema(Collection *col) {
  pipeann::dsl::Schema schema;
  for (auto &sf : col->meta.scalar_fields) {
    pipeann::dsl::FieldInfo fi;
    fi.key = sf.key;
    fi.type = sf.attr_type;
    fi.n_vectors = col->index->npoints();
    fi.attr_index = col->index->get_attr_index(sf.key);
    schema[sf.name] = fi;
  }
  return schema;
}

std::string CollectionStore::rewrite_float_literals(Collection *col, const std::string &expr) {
  std::string out = expr;
  for (const auto &sf : col->meta.scalar_fields) {
    if (sf.logical_type != "float") continue;
    std::regex cmp("(\\b" + sf.name +
                   "\\b\\s*(?:==|!=|>=|<=|>|<)\\s*)([-+]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::smatch m;
    std::string acc;
    std::string rest = out;
    while (std::regex_search(rest, m, cmp)) {
      acc += m.prefix().str();
      double val = std::stod(m[2].str());
      acc += m[1].str() + std::to_string(encode_float_ordered(static_cast<float>(val)));
      rest = m.suffix().str();
    }
    acc += rest;
    out = acc;
  }
  return out;
}

std::shared_ptr<pipeann::dsl::CompiledFilter> CollectionStore::compile_filter(Collection *col, const std::string &sql) {
  std::lock_guard<std::mutex> lk(col->compile_mu);
  auto it = col->compile_cache.find(sql);
  if (it != col->compile_cache.end()) return it->second;
  auto schema = build_dsl_schema(col);
  auto cf = std::make_shared<pipeann::dsl::CompiledFilter>(
      pipeann::dsl::compile(rewrite_float_literals(col, sql), schema));
  col->compile_cache[sql] = cf;
  return cf;
}

// ---------------------------------------------------------------------------
// PK expression fast path + delete
// ---------------------------------------------------------------------------
bool CollectionStore::parse_pk_expr(const std::string &pk_field, const std::string &expr,
                                    std::vector<std::string> *pks) {
  std::smatch m;
  std::regex eq_re("^\\s*" + pk_field + "\\s*(?:==|=)\\s*[\"']?([^\"']+?)[\"']?\\s*$", std::regex::icase);
  std::regex in_re("^\\s*" + pk_field + "\\s+in\\s*\\[(.*)\\]\\s*$", std::regex::icase);
  if (std::regex_match(expr, m, eq_re)) {
    pks->push_back(m[1].str());
    return true;
  }
  if (std::regex_match(expr, m, in_re)) {
    std::string list = m[1].str();
    std::regex item("[\"']?([^,\"'\\s][^,\"']*?)[\"']?\\s*(?:,|$)");
    for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) {
      std::string v = (*it)[1].str();
      if (!v.empty()) pks->push_back(v);
    }
    return true;
  }
  return false;
}

int64_t CollectionStore::delete_by_pks(Collection *col, const std::vector<std::string> &pks) {
  if (pks.empty() || !col->doc_store.is_open()) return 0;
  std::vector<int64_t> tags;
  col->doc_store.get_tags_by_ids(pks, tags);
  std::vector<uint32_t> present_tags;
  std::vector<std::string> present_ids;
  for (size_t i = 0; i < pks.size(); i++) {
    if (tags[i] >= 0) {
      present_tags.push_back(static_cast<uint32_t>(tags[i]));
      present_ids.push_back(pks[i]);
    }
  }
  if (present_tags.empty()) return 0;
  col->index->remove(present_tags.data(), static_cast<uint32_t>(present_tags.size()));
  col->doc_store.delete_by_ids(present_ids);
  return static_cast<int64_t>(present_tags.size());
}

int64_t CollectionStore::remove(const std::string &name, const std::vector<std::string> &explicit_pks,
                                const std::string &filter) {
  auto *col = get(name);
  if (!col) throw std::runtime_error("collection not found");
  std::vector<std::string> pks = explicit_pks;

  if (!filter.empty()) {
    std::vector<std::string> pk_from_expr;
    if (parse_pk_expr(col->meta.primary_field, filter, &pk_from_expr)) {
      pks.insert(pks.end(), pk_from_expr.begin(), pk_from_expr.end());
    } else if (!col->meta.scalar_fields.empty()) {
      auto cf = compile_filter(col, filter);
      auto tags = col->index->filter_only(cf->selector, cf->attrs_template, /*limit=*/0);
      std::vector<std::string> ids, docs;
      col->doc_store.get_by_tags(tags, ids, docs);
      for (auto &id : ids)
        if (!id.empty()) pks.push_back(id);
    }
  }
  return delete_by_pks(col, pks);
}

// ---------------------------------------------------------------------------
// Output-field projection
// ---------------------------------------------------------------------------
void CollectionStore::build_output(Collection *col, const std::vector<std::string> &output_fields,
                                   const std::vector<OutputRow> &rows, QueryResult *out) {
  if (output_fields.empty()) return;

  std::map<std::string, const ScalarFieldMeta *> scalar_by_name;
  for (const auto &sf : col->meta.scalar_fields) scalar_by_name[sf.name] = &sf;
  std::map<std::string, bool> display_set;
  for (const auto &d : col->meta.display_fields) display_set[d] = true;

  // Pre-parse each row's display JSON once.
  std::vector<picojson::object> docs(rows.size());
  for (size_t i = 0; i < rows.size(); i++) {
    if (rows[i].doc.empty()) continue;
    picojson::value v;
    if (picojson::parse(v, rows[i].doc).empty() && v.is<picojson::object>()) docs[i] = v.get<picojson::object>();
  }

  for (const auto &fname : output_fields) {
    if (fname == "count(*)") continue;
    OutputColumn oc;
    oc.name = fname;

    if (fname == col->meta.primary_field || fname == "id") {
      oc.kind = FieldKind::VarChar;
      oc.scalar.kind = FieldKind::VarChar;
      for (const auto &r : rows) oc.scalar.strings.push_back(r.pk);
      out->columns.push_back(std::move(oc));
      continue;
    }

    if (fname == col->meta.vector_field || fname == "vector") {
      oc.kind = FieldKind::FloatVector;
      oc.dim = col->meta.dim;
      for (const auto &r : rows) {
        if (r.payload != nullptr && r.payload->coords.size() == col->meta.dim) {
          for (float c : r.payload->coords) oc.vectors.push_back(c);
        } else {
          for (uint32_t d = 0; d < col->meta.dim; d++) oc.vectors.push_back(0.0f);
        }
      }
      out->columns.push_back(std::move(oc));
      continue;
    }

    auto sit = scalar_by_name.find(fname);
    if (sit != scalar_by_name.end()) {
      const ScalarFieldMeta *sf = sit->second;
      LogicalType ltype = logical_type_from_string(sf->logical_type);
      uint32_t key = sf->key;
      auto get_attr = [&](const OutputRow &r, pipeann::Attribute &a) -> bool {
        if (r.payload == nullptr || !r.payload->attrs.find(key)) return false;
        a = r.payload->attrs.get(key);
        return true;
      };
      pipeann::Attribute a;
      oc.kind = kind_from_logical(sf->logical_type);
      oc.scalar.kind = oc.kind;
      switch (ltype) {
        case LogicalType::Float:
          for (const auto &r : rows) oc.scalar.doubles.push_back(get_attr(r, a) ? decode_field(a, ltype).get<double>() : 0.0);
          break;
        case LogicalType::Bool:
          for (const auto &r : rows) oc.scalar.bools.push_back(get_attr(r, a) ? decode_field(a, ltype).get<bool>() : false);
          break;
        case LogicalType::String:
          for (const auto &r : rows) oc.scalar.strings.push_back(get_attr(r, a) ? decode_field(a, ltype).get<std::string>() : std::string());
          break;
        case LogicalType::Label:
          for (const auto &r : rows) {
            std::vector<int64_t> labels;
            if (get_attr(r, a))
              for (uint32_t v : a) labels.push_back(static_cast<int64_t>(v));
            oc.scalar.int_arrays.push_back(std::move(labels));
          }
          break;
        case LogicalType::Int:
        default:
          for (const auto &r : rows)
            oc.scalar.ints.push_back(get_attr(r, a) ? static_cast<int64_t>(decode_field(a, ltype).get<double>()) : 0);
          break;
      }
      out->columns.push_back(std::move(oc));
      continue;
    }

    // Display-only field: infer type from the first present value.
    const picojson::value *sample = nullptr;
    for (const auto &o : docs) {
      auto dit = o.find(fname);
      if (dit != o.end()) { sample = &dit->second; break; }
    }
    auto value_for = [&](size_t i) -> const picojson::value * {
      auto dit = docs[i].find(fname);
      return dit != docs[i].end() ? &dit->second : nullptr;
    };
    if (sample != nullptr && sample->is<bool>()) {
      oc.kind = FieldKind::Bool;
      oc.scalar.kind = FieldKind::Bool;
      for (size_t i = 0; i < docs.size(); i++) {
        const picojson::value *v = value_for(i);
        oc.scalar.bools.push_back(v && v->is<bool>() ? v->get<bool>() : false);
      }
    } else if (sample != nullptr && sample->is<double>()) {
      oc.kind = FieldKind::Float;
      oc.scalar.kind = FieldKind::Float;
      for (size_t i = 0; i < docs.size(); i++) {
        const picojson::value *v = value_for(i);
        oc.scalar.doubles.push_back(v && v->is<double>() ? v->get<double>() : 0.0);
      }
    } else {
      oc.kind = FieldKind::VarChar;
      oc.scalar.kind = FieldKind::VarChar;
      for (size_t i = 0; i < docs.size(); i++) {
        const picojson::value *v = value_for(i);
        if (!v) oc.scalar.strings.push_back(std::string());
        else if (v->is<std::string>()) oc.scalar.strings.push_back(v->get<std::string>());
        else oc.scalar.strings.push_back(v->serialize());
      }
    }
    out->columns.push_back(std::move(oc));
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
QueryResult CollectionStore::search(const std::string &name, const SearchParams &params) {
  auto *col = get(name);
  if (!col) throw std::runtime_error("collection not found");
  if (!col->index_built) throw std::runtime_error("index not built; call create_index/load_collection first");

  int n_queries = params.n_queries;
  int topk = params.topk;
  int L = std::max(params.L, topk);
  uint32_t dim = col->meta.dim;

  pipeann::Selector *selector = nullptr;
  std::vector<pipeann::Attributes> query_attrs_vec;
  std::shared_ptr<pipeann::dsl::CompiledFilter> cf;
  if (!params.filter.empty() && !col->meta.scalar_fields.empty()) {
    cf = compile_filter(col, params.filter);
    selector = cf->selector;
    query_attrs_vec.resize(n_queries, cf->attrs_template);
  }

  bool want_fields = false;
  for (const auto &f : params.output_fields) {
    if (f != "id") { want_fields = true; break; }
  }
  std::vector<DynamicIndex<float>::NodeOut> node_outs;
  if (want_fields) node_outs.resize(n_queries);

  std::vector<uint32_t> out_ids((size_t) n_queries * topk);
  std::vector<float> out_dists((size_t) n_queries * topk);

  search_pool_->submit_and_wait([&] {
    for (int i = 0; i < n_queries; i++) {
      col->index->search(params.queries.data() + (size_t) i * dim, topk, L, out_ids.data() + (size_t) i * topk,
                         out_dists.data() + (size_t) i * topk, nullptr, selector,
                         selector != nullptr ? &query_attrs_vec[i] : nullptr,
                         std::numeric_limits<float>::infinity(), want_fields ? &node_outs[i] : nullptr);
    }
  });

  // Resolve all result tags in one batched doc-store lookup.
  std::vector<uint32_t> result_tags;
  result_tags.reserve((size_t) n_queries * topk);
  for (int q = 0; q < n_queries; q++) {
    for (int k = 0; k < topk; k++) {
      uint32_t tag = out_ids[(size_t) q * topk + k];
      if (tag == UINT32_MAX) break;
      result_tags.push_back(tag);
    }
  }
  std::vector<std::string> resolved_ids, resolved_docs;
  col->doc_store.get_by_tags(result_tags, resolved_ids, resolved_docs);

  QueryResult res;
  res.primary_field_name = col->meta.primary_field;
  size_t pos = 0;
  std::vector<OutputRow> output_rows;
  if (want_fields) output_rows.reserve(result_tags.size());
  for (int q = 0; q < n_queries; q++) {
    int count = 0;
    for (int k = 0; k < topk; k++) {
      uint32_t tag = out_ids[(size_t) q * topk + k];
      float dist = out_dists[(size_t) q * topk + k];
      if (tag == UINT32_MAX) break;
      const std::string &doc_id = resolved_ids[pos];
      std::string pk = doc_id.empty() ? std::to_string(tag) : doc_id;
      res.ids.push_back(pk);
      res.scores.push_back(dist);
      if (want_fields) {
        OutputRow row;
        row.pk = pk;
        row.doc = resolved_docs[pos];
        auto it = node_outs[q].find(tag);
        if (it != node_outs[q].end()) row.payload = &it->second;
        output_rows.push_back(std::move(row));
      }
      ++pos;
      ++count;
    }
    res.topks.push_back(count);
  }

  if (want_fields) {
    build_output(col, params.output_fields, output_rows, &res);
    res.output_field_names = params.output_fields;
  }
  return res;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------
int64_t CollectionStore::count(const std::string &name, const std::string &filter) {
  auto *col = get(name);
  if (!col) throw std::runtime_error("collection not found");
  if (!filter.empty()) {
    QueryParams qp;
    qp.filter = filter;
    qp.output_fields = {"id"};
    return (int64_t) query(name, qp).ids.size();
  }
  std::lock_guard<std::mutex> lk(col->mu);
  if (col->doc_store.is_open()) return col->doc_store.count();
  return col->index_built ? (int64_t) col->index->npoints() : (int64_t) col->buf_tags.size();
}

QueryResult CollectionStore::query(const std::string &name, const QueryParams &params) {
  auto *col = get(name);
  if (!col) throw std::runtime_error("collection not found");
  if (!col->index_built) throw std::runtime_error("index not built; call create_index/load_collection first");

  const std::string &expr = params.filter;
  size_t limit = params.limit;

  std::vector<std::string> ofs = params.output_fields;
  if (ofs.empty()) ofs.push_back("id");

  QueryResult res;
  res.primary_field_name = col->meta.primary_field;

  if (expr.empty()) {
    // Full scan from the doc-store.
    std::vector<uint32_t> all_tags;
    std::vector<std::string> all_ids, all_docs;
    col->doc_store.scan(all_tags, all_ids, all_docs);
    size_t n = (limit > 0 && limit < all_tags.size()) ? limit : all_tags.size();
    std::vector<OutputRow> rows(n);
    for (size_t i = 0; i < n; i++) {
      rows[i].pk = all_ids[i].empty() ? std::to_string(all_tags[i]) : all_ids[i];
      rows[i].doc = all_docs[i];
    }
    for (const auto &r : rows) res.ids.push_back(r.pk);
    build_output(col, ofs, rows, &res);
    res.output_field_names = ofs;
    return res;
  }

  std::vector<uint32_t> tags;
  DynamicIndex<float>::NodeOut node_out;

  std::vector<std::string> pks;
  if (parse_pk_expr(col->meta.primary_field, expr, &pks)) {
    std::vector<int64_t> resolved_tags;
    col->doc_store.get_tags_by_ids(pks, resolved_tags);
    for (size_t i = 0; i < pks.size(); i++) {
      if (resolved_tags[i] >= 0) tags.push_back((uint32_t) resolved_tags[i]);
    }
    if (limit > 0 && tags.size() > limit) tags.resize(limit);
    if (!tags.empty()) {
      search_pool_->submit_and_wait([&] { col->index->get_nodes_by_tags(tags, &node_out); });
    }
  } else {
    if (col->meta.scalar_fields.empty()) throw std::runtime_error("query filter requires indexed scalar fields");
    auto cf = compile_filter(col, expr);
    search_pool_->submit_and_wait([&] {
      tags = col->index->filter_only(cf->selector, cf->attrs_template, limit, &node_out);
    });
  }

  std::vector<std::string> resolved_ids, resolved_docs;
  std::vector<char> found;
  col->doc_store.get_by_tags(tags, resolved_ids, resolved_docs, &found);

  // Drop tags whose doc-store row is gone (lazy-delete tombstones the attr index
  // may still match).
  std::vector<OutputRow> rows;
  rows.reserve(tags.size());
  for (size_t i = 0; i < tags.size(); i++) {
    if (!found[i]) continue;
    OutputRow row;
    row.pk = resolved_ids[i].empty() ? std::to_string(tags[i]) : resolved_ids[i];
    row.doc = resolved_docs[i];
    auto it = node_out.find(tags[i]);
    if (it != node_out.end()) row.payload = &it->second;
    rows.push_back(std::move(row));
  }
  for (const auto &r : rows) res.ids.push_back(r.pk);
  build_output(col, ofs, rows, &res);
  res.output_field_names = ofs;
  return res;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
void CollectionStore::flush(const std::string &name) {
  if (name.empty()) { flush_all(); return; }
  auto *col = get(name);
  if (col && col->doc_store.is_open()) col->doc_store.flush();
}

void CollectionStore::flush_all() {
  std::lock_guard<std::mutex> lk(collections_mu_);
  for (auto &[n, col] : collections_) {
    if (col->doc_store.is_open()) col->doc_store.flush();
  }
}

// ---------------------------------------------------------------------------
// Flat-layout schema.json sidecar
// ---------------------------------------------------------------------------
std::string CollectionStore::schema_path(const std::string &index_prefix) const {
  return index_prefix + ".schema.json";
}

static const char *field_kind_to_str(FieldKind k) {
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

static FieldKind field_kind_from_str(const std::string &s) {
  if (s == "float") return FieldKind::Float;
  if (s == "bool") return FieldKind::Bool;
  if (s == "varchar") return FieldKind::VarChar;
  if (s == "array_int") return FieldKind::ArrayInt;
  if (s == "float_vector") return FieldKind::FloatVector;
  if (s == "json") return FieldKind::Json;
  return FieldKind::Int64;
}

void CollectionStore::write_schema_json(Collection *col) {
  const std::string &prefix = col->index_prefix;
  if (prefix.empty()) return;
  std::string base = std::filesystem::path(prefix).filename().string();

  picojson::object root;
  root["type"] = picojson::value(std::string("collection"));
  root["format"] = picojson::value(std::string("pipeann-flat-v2"));
  root["data_dim"] = picojson::value(static_cast<double>(col->meta.dim));
  root["data_type"] = picojson::value(col->meta.data_type);
  root["metric"] = picojson::value(col->meta.metric);
  root["primary_field"] = picojson::value(col->meta.primary_field);
  root["vector_field"] = picojson::value(col->meta.vector_field);
  root["index_prefix"] = picojson::value(base);
  root["npoints"] = picojson::value(static_cast<double>(col->index->npoints()));

  picojson::array scalar_fields;
  for (const auto &sf : col->meta.scalar_fields) {
    picojson::object o;
    o["name"] = picojson::value(sf.name);
    o["key"] = picojson::value(static_cast<double>(sf.key));
    o["type"] = picojson::value(sf.attr_type);
    o["logical_type"] = picojson::value(sf.logical_type);
    scalar_fields.push_back(picojson::value(o));
  }
  root["scalar_fields"] = picojson::value(scalar_fields);

  picojson::array display_fields;
  for (const auto &name : col->meta.display_fields) display_fields.push_back(picojson::value(name));
  root["display_fields"] = picojson::value(display_fields);

  // Neutral field schema (kind-tagged) so describe() can be served after reload.
  picojson::array fields;
  for (const auto &f : col->meta.fields) {
    picojson::object fo;
    fo["name"] = picojson::value(f.name);
    fo["kind"] = picojson::value(std::string(field_kind_to_str(f.kind)));
    fo["is_primary"] = picojson::value(f.is_primary);
    fo["dim"] = picojson::value(static_cast<double>(f.dim));
    fields.push_back(picojson::value(fo));
  }
  root["fields"] = picojson::value(fields);

  std::ofstream out(schema_path(prefix));
  if (!out) return;
  out << picojson::value(root).serialize(/*prettify=*/true);
}

bool CollectionStore::load_collection_from_schema(const std::string &schema_file) {
  std::ifstream in(schema_file);
  if (!in) return false;
  picojson::value v;
  std::string perr = picojson::parse(v, in);
  if (!perr.empty() || !v.is<picojson::object>()) return false;
  const auto &o = v.get<picojson::object>();
  auto get_str = [&](const char *k, const std::string &dflt) -> std::string {
    auto it = o.find(k);
    return (it != o.end() && it->second.is<std::string>()) ? it->second.get<std::string>() : dflt;
  };
  if (get_str("type", "") != "collection") return false;

  const std::string suffix = ".schema.json";
  std::string prefix = schema_file.substr(0, schema_file.size() - suffix.size());
  std::string name = std::filesystem::path(prefix).filename().string();

  CollectionMeta meta;
  meta.name = name;
  auto dim_it = o.find("data_dim");
  meta.dim = (dim_it != o.end() && dim_it->second.is<double>()) ? (uint32_t) dim_it->second.get<double>() : 0;
  meta.data_type = get_str("data_type", "float32");
  meta.metric = get_str("metric", "l2");
  meta.primary_field = get_str("primary_field", "id");
  meta.vector_field = get_str("vector_field", "vector");

  auto sf_it = o.find("scalar_fields");
  if (sf_it != o.end() && sf_it->second.is<picojson::array>()) {
    for (const auto &sfv : sf_it->second.get<picojson::array>()) {
      if (!sfv.is<picojson::object>()) continue;
      const auto &so = sfv.get<picojson::object>();
      ScalarFieldMeta sfm;
      sfm.name = so.count("name") ? so.at("name").get<std::string>() : "";
      sfm.key = so.count("key") ? (uint32_t) so.at("key").get<double>() : 0;
      sfm.attr_type = so.count("type") ? so.at("type").get<std::string>() : "label";
      if (so.count("logical_type")) sfm.logical_type = so.at("logical_type").get<std::string>();
      else if (sfm.attr_type == "range") sfm.logical_type = "int";
      else sfm.logical_type = sfm.attr_type;
      meta.scalar_fields.push_back(sfm);
    }
  }
  auto df_it = o.find("display_fields");
  if (df_it != o.end() && df_it->second.is<picojson::array>()) {
    for (const auto &dfv : df_it->second.get<picojson::array>())
      if (dfv.is<std::string>()) meta.display_fields.push_back(dfv.get<std::string>());
  }
  auto field_it = o.find("fields");
  if (field_it != o.end() && field_it->second.is<picojson::array>()) {
    for (const auto &fv : field_it->second.get<picojson::array>()) {
      if (!fv.is<picojson::object>()) continue;
      const auto &fo = fv.get<picojson::object>();
      FieldSpec f;
      if (fo.count("name")) f.name = fo.at("name").get<std::string>();
      if (fo.count("kind")) f.kind = field_kind_from_str(fo.at("kind").get<std::string>());
      if (fo.count("is_primary")) f.is_primary = fo.at("is_primary").get<bool>();
      if (fo.count("dim")) f.dim = (uint32_t) fo.at("dim").get<double>();
      meta.fields.push_back(f);
    }
  }

  if (meta.dim == 0) return false;

  auto col = std::make_unique<Collection>();
  col->meta = std::move(meta);
  col->index_prefix = prefix;

  pipeann::IndexBuildParameters build_params;
  build_params.num_threads = omp_threads_ > 0 ? omp_threads_ : 0;
  // Match the SSD I/O buffer pool to the search worker count (see create_collection).
  build_params.max_nthreads = omp_threads_ > 0 ? omp_threads_ : kDefaultWorkerThreads;
  col->index = std::make_unique<DynamicIndex<float>>(col->meta.dim, metric_from_string(col->meta.metric), &build_params,
                                                     /*attr_size=*/0, /*enable_tag2id=*/true);
  if (omp_threads_ > 0) col->index->omp_set_num_threads(omp_threads_);
  col->index->set_index_prefix(prefix);
  col->index->load(prefix, /*copy_to_shadow=*/false);

  for (const auto &sf : col->meta.scalar_fields) {
    col->index->load_attr_index_from_file(sf.key, col->index->attr_index_file(sf.key), sf.attr_type);
  }
  col->index_built = true;

  uint32_t max_tag = 0;
  if (col->doc_store.open(prefix + ".docs.rocksdb")) {
    int64_t mt = col->doc_store.max_tag();
    if (mt >= 0) max_tag = static_cast<uint32_t>(mt) + 1;
  }
  uint32_t npts = (uint32_t) col->index->npoints();
  col->next_tag = std::max(max_tag, npts);

  std::string nm = col->meta.name;
  std::lock_guard<std::mutex> lk(collections_mu_);
  collections_[nm] = std::move(col);
  return true;
}

void CollectionStore::load_existing_collections() {
  std::error_code ec;
  if (!std::filesystem::is_directory(data_dir_, ec)) return;
  const std::string suffix = ".schema.json";
  for (const auto &entry : std::filesystem::directory_iterator(data_dir_, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    std::string p = entry.path().string();
    if (p.size() < suffix.size() || p.compare(p.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
    try {
      load_collection_from_schema(p);
    } catch (const std::exception &e) {
      LOG(ERROR) << "Failed to load collection from " << p << ": " << e.what();
    }
  }
}

}  // namespace server
}  // namespace pipeann
