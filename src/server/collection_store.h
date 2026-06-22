// Neutral, front-end-agnostic collection engine for the Milvus-compatible layer.
//
// CollectionStore is the SINGLE source of truth for the Milvus data model:
// schema, typed scalar attr-index wiring, bulk-build buffering, the doc-store,
// filtered/unfiltered search + query, delete, output-field projection, and the
// flat-layout schema.json sidecar. It operates only on neutral C++ types (this
// header) — no protobuf, no pybind — so both the gRPC server and the Python
// binding drive the same engine and stay bit-for-bit consistent.
//
// Front-ends (src/server/milvus_server.cpp for gRPC, src/python for pybind) are
// thin marshallers: protobuf/Python <-> the neutral types below.
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "doc_store.h"
#include "dynamic_index.h"
#include "field_codec.h"
#include "search_worker_pool.h"

namespace pipeann {
namespace server {

// Default number of search worker threads (and the SSD I/O buffer pool size
// derived from it) when the caller does not pin a thread count. Used both for
// the SearchWorkerPool and for IndexBuildParameters::max_nthreads, which sizes
// the per-thread query buffers (init_buffers allocates max_nthreads*2 of them).
constexpr int kDefaultWorkerThreads = 32;

// ---------------------------------------------------------------------------
// Neutral field type tags. A coarse, front-end-agnostic view of a Milvus field.
// ---------------------------------------------------------------------------
enum class FieldKind {
  Int64,        // integer scalar (filterable range, logical "int")
  Float,        // float/double scalar (filterable range, logical "float")
  Bool,         // boolean scalar (filterable range, logical "bool")
  VarChar,      // string scalar (filterable, logical "string")
  ArrayInt,     // array of int label ids (filterable, logical "label")
  FloatVector,  // the embedding column
  Json,         // display-only structured value (not filterable)
};

// One field in a collection schema.
struct FieldSpec {
  std::string name;
  FieldKind kind = FieldKind::Int64;
  bool is_primary = false;
  uint32_t dim = 0;  // FloatVector only
};

struct CollectionSpec {
  std::string name;
  std::vector<FieldSpec> fields;
  std::string metric = "l2";  // l2 | inner_product | cosine
};

// ---------------------------------------------------------------------------
// Columnar scalar payload. Exactly one typed buffer is populated per column,
// selected by `kind`. Mirrors the columnar shape both protobuf FieldData and
// the Python row->column transform already use, so marshalling is a copy.
// ---------------------------------------------------------------------------
struct ScalarColumn {
  FieldKind kind = FieldKind::Int64;
  std::vector<int64_t> ints;                     // Int64 / Bool(0,1)
  std::vector<double> doubles;                   // Float
  std::vector<bool> bools;                       // Bool
  std::vector<std::string> strings;              // VarChar / Json (raw)
  std::vector<std::vector<int64_t>> int_arrays;  // ArrayInt label lists
};

// Columnar batch handed to insert/upsert. `pks` holds the primary key per row
// (stringified, matching the doc-store id encoding). `vectors` is row-major
// num_rows * dim. `scalars` is keyed by field name; any subset of the schema's
// scalar/display fields may be present.
struct InsertColumns {
  uint32_t num_rows = 0;
  std::vector<std::string> pks;
  std::vector<float> vectors;
  std::map<std::string, ScalarColumn> scalars;
};

struct InsertResult {
  std::vector<std::string> ids;
  int64_t count = 0;
};

// ---------------------------------------------------------------------------
// Output projection. A QueryResult carries the row primary keys plus one
// OutputColumn per requested output_field, flattened row-major. For Search,
// `topks` slices the flat rows back into per-query result lists; for Query it
// is empty (single flat list).
// ---------------------------------------------------------------------------
struct OutputColumn {
  std::string name;
  FieldKind kind = FieldKind::Int64;
  uint32_t dim = 0;  // FloatVector only
  ScalarColumn scalar;
  std::vector<float> vectors;  // FloatVector: flat n_rows * dim
};

struct QueryResult {
  std::vector<std::string> ids;       // primary key per row (row-major)
  std::vector<float> scores;          // search only; distance per row
  std::vector<int64_t> topks;         // search only; hits per query
  std::vector<OutputColumn> columns;  // one per requested output field
  std::vector<std::string> output_field_names;
  std::string primary_field_name;
};

struct SearchParams {
  std::vector<float> queries;  // flat n_queries * dim
  int n_queries = 0;
  int topk = 10;
  int L = 50;
  std::string filter;  // SQL filter expression ("" = none)
  std::vector<std::string> output_fields;
};

struct QueryParams {
  std::string filter;  // SQL filter expression ("" = full scan)
  size_t limit = 0;    // 0 = no cap
  std::vector<std::string> output_fields;
};

// ---------------------------------------------------------------------------
// Per-field metadata derived from the schema and persisted in schema.json.
// ---------------------------------------------------------------------------
struct ScalarFieldMeta {
  std::string name;
  std::string attr_type;     // "label" | "range" | "string"
  std::string logical_type;  // "int" | "float" | "bool" | "string" | "label"
  uint32_t key = 0;          // slot in DynamicIndex attr_index_map
};

// Full resolved metadata for one collection.
struct CollectionMeta {
  std::string name;
  uint32_t dim = 0;
  std::string data_type = "float32";
  std::string metric = "l2";
  std::string primary_field = "id";
  std::string vector_field = "vector";
  std::vector<ScalarFieldMeta> scalar_fields;   // filterable, in an attr index
  std::vector<std::string> display_fields;      // stored verbatim as JSON
  std::vector<FieldSpec> fields;                 // full schema, for describe
};

// ---------------------------------------------------------------------------
// CollectionStore: the engine. One process-wide instance backed by a data_dir;
// owns all collections, their indexes, doc-stores, and the shared search pool.
// Thread-safe. All methods take/return neutral types only.
// ---------------------------------------------------------------------------
class CollectionStore {
 public:
  explicit CollectionStore(const std::string &data_dir, int omp_threads = 0);
  ~CollectionStore();

  // --- Collection lifecycle ---
  // Create a collection from a neutral spec. Throws std::runtime_error on error
  // (e.g. no vector field). No-op-safe: caller checks has() first if desired.
  void create_collection(const CollectionSpec &spec);
  void drop_collection(const std::string &name);
  bool has_collection(const std::string &name);
  std::vector<std::string> list_collections();
  // Returns a copy of the resolved metadata, or nullopt if absent.
  std::optional<CollectionMeta> describe(const std::string &name);
  int64_t row_count(const std::string &name);

  // Build the disk index from buffered inserts. Idempotent. Throws on build
  // failure. Both create_index and load_collection map here. build_R / build_L
  // override graph degree and build search-list size (0 = engine auto-config);
  // they only take effect on the build that actually materializes the index.
  void build_index(const std::string &name, uint32_t build_R = 0, uint32_t build_L = 0);
  bool index_built(const std::string &name);

  // --- Data ops ---
  InsertResult insert(const std::string &name, const InsertColumns &cols);
  InsertResult upsert(const std::string &name, const InsertColumns &cols);
  // Delete by explicit pks and/or a SQL filter expression. Returns count removed.
  int64_t remove(const std::string &name, const std::vector<std::string> &pks, const std::string &filter);

  QueryResult search(const std::string &name, const SearchParams &params);
  QueryResult query(const std::string &name, const QueryParams &params);
  int64_t count(const std::string &name, const std::string &filter);

  // --- Persistence ---
  void flush(const std::string &name);  // "" => all
  void flush_all();

 private:
  struct Collection {
    CollectionMeta meta;
    std::unique_ptr<DynamicIndex<float>> index;
    pipeann::DocStore doc_store;
    std::string index_prefix;
    std::atomic<uint32_t> next_tag{0};
    std::mutex mu;

    bool index_built = false;
    std::vector<float> buf_vectors;
    std::vector<uint32_t> buf_tags;
    std::vector<pipeann::Attributes> buf_attrs;

    // Compiled-filter cache: SQL expr -> compiled filter. Invalidated on schema
    // change (collections here have fixed schemas, so keyed by expr alone).
    std::mutex compile_mu;
    std::map<std::string, std::shared_ptr<pipeann::dsl::CompiledFilter>> compile_cache;
  };

  Collection *get(const std::string &name);
  void build_index_locked(Collection *col, uint32_t build_R = 0, uint32_t build_L = 0);

  // Compile (and cache) a SQL filter against a collection's scalar schema.
  std::shared_ptr<pipeann::dsl::CompiledFilter> compile_filter(Collection *col, const std::string &sql);
  pipeann::dsl::Schema build_dsl_schema(Collection *col);
  std::string rewrite_float_literals(Collection *col, const std::string &expr);

  // One result row to project. pk + optional node payload + display-field JSON.
  struct OutputRow {
    std::string pk;
    const DynamicIndex<float>::NodeOut::mapped_type *payload = nullptr;
    std::string doc;
  };
  void build_output(Collection *col, const std::vector<std::string> &output_fields,
                    const std::vector<OutputRow> &rows, QueryResult *out);

  // Encode one InsertColumns batch into per-row Attributes (filterable) + the
  // per-row display JSON document.
  void encode_attrs_and_docs(Collection *col, const InsertColumns &cols,
                             std::vector<pipeann::Attributes> *attrs_vec, std::vector<std::string> *docs);

  // Parse "id == x" / "id in [...]" PK fast-path expressions. Returns true and
  // fills `pks` if the expr is a pure PK predicate; false otherwise.
  bool parse_pk_expr(const std::string &pk_field, const std::string &expr, std::vector<std::string> *pks);

  int64_t delete_by_pks(Collection *col, const std::vector<std::string> &pks);

  // Flat-layout persistence.
  std::string schema_path(const std::string &index_prefix) const;
  void write_schema_json(Collection *col);
  void load_existing_collections();
  bool load_collection_from_schema(const std::string &schema_file);

  std::string data_dir_;
  int omp_threads_;
  std::unique_ptr<SearchWorkerPool> search_pool_;
  std::mutex collections_mu_;
  std::map<std::string, std::unique_ptr<Collection>> collections_;
};

}  // namespace server
}  // namespace pipeann

