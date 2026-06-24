#include "milvus_server.h"

#include <grpcpp/grpcpp.h>

#include <cstring>
#include <string>
#include <vector>

#include "utils/picojson.h"

namespace pipeann {
namespace server {

using grpc::Status;
using grpc::StatusCode;

static pb_common::Status ok_status() {
  pb_common::Status s;
  s.set_error_code(pb_common::ErrorCode::Success);
  return s;
}

static pb_common::Status err_status(const std::string &reason) {
  pb_common::Status s;
  s.set_error_code(pb_common::ErrorCode::UnexpectedError);
  s.set_reason(reason);
  return s;
}

// Map a Milvus proto DataType to a neutral FieldKind.
static FieldKind kind_from_proto(pb_schema::DataType dt) {
  switch (dt) {
    case pb_schema::DataType::FloatVector: return FieldKind::FloatVector;
    case pb_schema::DataType::Array: return FieldKind::ArrayInt;
    case pb_schema::DataType::Bool: return FieldKind::Bool;
    case pb_schema::DataType::Int64:
    case pb_schema::DataType::Int32:
    case pb_schema::DataType::Int16:
    case pb_schema::DataType::Int8: return FieldKind::Int64;
    case pb_schema::DataType::Float:
    case pb_schema::DataType::Double: return FieldKind::Float;
    case pb_schema::DataType::VarChar:
    case pb_schema::DataType::String: return FieldKind::VarChar;
    default: return FieldKind::Json;  // display-only
  }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
MilvusServiceImpl::MilvusServiceImpl(const std::string &data_dir, int omp_threads) {
  store_ = std::make_unique<CollectionStore>(data_dir, omp_threads);
}

MilvusServiceImpl::~MilvusServiceImpl() = default;

// ---------------------------------------------------------------------------
// Connection & Version
// ---------------------------------------------------------------------------
Status MilvusServiceImpl::Connect(grpc::ServerContext *, const pb_milvus::ConnectRequest *,
                                  pb_milvus::ConnectResponse *resp) {
  *resp->mutable_status() = ok_status();
  resp->set_identifier(1);
  return Status::OK;
}

Status MilvusServiceImpl::GetVersion(grpc::ServerContext *, const pb_milvus::GetVersionRequest *,
                                     pb_milvus::GetVersionResponse *resp) {
  *resp->mutable_status() = ok_status();
  resp->set_version("pipeann-milvus-cpp");
  return Status::OK;
}

// ---------------------------------------------------------------------------
// Collection CRUD
// ---------------------------------------------------------------------------
Status MilvusServiceImpl::CreateCollection(grpc::ServerContext *, const pb_milvus::CreateCollectionRequest *req,
                                           pb_common::Status *resp) {
  pb_schema::CollectionSchema cs;
  cs.ParseFromString(req->schema());

  CollectionSpec spec;
  spec.name = req->collection_name();
  for (auto &f : cs.fields()) {
    FieldSpec fs;
    fs.name = f.name();
    fs.kind = kind_from_proto(f.data_type());
    fs.is_primary = f.is_primary_key();
    if (fs.kind == FieldKind::FloatVector) {
      for (auto &p : f.type_params())
        if (p.key() == "dim") fs.dim = std::stoi(p.value());
    }
    spec.fields.push_back(fs);
  }

  try {
    store_->create_collection(spec);
  } catch (const std::exception &e) {
    *resp = err_status(e.what());
    return Status::OK;
  }
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::DropCollection(grpc::ServerContext *, const pb_milvus::DropCollectionRequest *req,
                                         pb_common::Status *resp) {
  store_->drop_collection(req->collection_name());
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::HasCollection(grpc::ServerContext *, const pb_milvus::HasCollectionRequest *req,
                                        pb_milvus::BoolResponse *resp) {
  *resp->mutable_status() = ok_status();
  resp->set_value(store_->has_collection(req->collection_name()));
  return Status::OK;
}

// Synthesize a proto FieldSchema from a neutral FieldSpec for DescribeCollection.
static void fill_proto_field(pb_schema::FieldSchema *f, const FieldSpec &fs) {
  f->set_name(fs.name);
  f->set_is_primary_key(fs.is_primary);
  switch (fs.kind) {
    case FieldKind::FloatVector: {
      f->set_data_type(pb_schema::DataType::FloatVector);
      auto *p = f->add_type_params();
      p->set_key("dim");
      p->set_value(std::to_string(fs.dim));
      break;
    }
    case FieldKind::ArrayInt:
      f->set_data_type(pb_schema::DataType::Array);
      f->set_element_type(pb_schema::DataType::Int64);
      break;
    case FieldKind::VarChar:
      f->set_data_type(pb_schema::DataType::VarChar);
      { auto *p = f->add_type_params(); p->set_key("max_length"); p->set_value("65535"); }
      break;
    case FieldKind::Bool: f->set_data_type(pb_schema::DataType::Bool); break;
    case FieldKind::Float: f->set_data_type(pb_schema::DataType::Float); break;
    case FieldKind::Json: f->set_data_type(pb_schema::DataType::JSON); break;
    case FieldKind::Int64:
    default: f->set_data_type(pb_schema::DataType::Int64); break;
  }
}

Status MilvusServiceImpl::DescribeCollection(grpc::ServerContext *, const pb_milvus::DescribeCollectionRequest *req,
                                             pb_milvus::DescribeCollectionResponse *resp) {
  auto meta = store_->describe(req->collection_name());
  if (!meta) {
    // pymilvus has_collection() treats this wording as "not found".
    *resp->mutable_status() = err_status("can't find collection: " + req->collection_name());
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  resp->set_collection_name(req->collection_name());
  auto *cs = resp->mutable_schema();
  cs->set_name(req->collection_name());
  for (auto &f : meta->fields) fill_proto_field(cs->add_fields(), f);
  return Status::OK;
}

Status MilvusServiceImpl::ShowCollections(grpc::ServerContext *, const pb_milvus::ShowCollectionsRequest *,
                                          pb_milvus::ShowCollectionsResponse *resp) {
  *resp->mutable_status() = ok_status();
  for (auto &name : store_->list_collections()) resp->add_collection_names(name);
  return Status::OK;
}

Status MilvusServiceImpl::GetCollectionStatistics(grpc::ServerContext *,
                                                  const pb_milvus::GetCollectionStatisticsRequest *req,
                                                  pb_milvus::GetCollectionStatisticsResponse *resp) {
  *resp->mutable_status() = ok_status();
  auto *kv = resp->add_stats();
  kv->set_key("row_count");
  kv->set_value(std::to_string(store_->row_count(req->collection_name())));
  return Status::OK;
}

// ---------------------------------------------------------------------------
// Index / Load / Flush
// ---------------------------------------------------------------------------
Status MilvusServiceImpl::CreateIndex(grpc::ServerContext *, const pb_milvus::CreateIndexRequest *req,
                                      pb_common::Status *resp) {
  // PipeANN has a single native index type, so index_type/metric_type in
  // extra_params are accepted and ignored for engine selection. We do honor the
  // build knobs, with two parameter conventions depending on index_type:
  //   DISKANN  -> PipeANN-native: R -> R (graph degree), L -> L (build list).
  //   otherwise (HNSW/AUTOINDEX) -> M -> R (kept at ~2*M to match HNSW's
  //               base-layer connectivity), efConstruction -> L.
  // Knobs may arrive as flat keys or nested inside a "params" JSON blob.
  int m = -1, ef = -1, r = -1, l = -1;
  std::string index_type;

  // Pull an integer out of a parsed "params" object (picojson numbers are
  // doubles here; these knobs are small so the cast is exact).
  auto obj_int = [](const picojson::object &o, const std::string &key) -> int {
    auto it = o.find(key);
    if (it == o.end() || !it->second.is<double>()) return -1;
    return static_cast<int>(it->second.get<double>());
  };

  auto parse_kv = [&](const std::string &k, const std::string &v) {
    if (k == "index_type") { index_type = v; }
    else if (k == "M") { try { m = std::stoi(v); } catch (...) {} }
    else if (k == "efConstruction") { try { ef = std::stoi(v); } catch (...) {} }
    else if (k == "R") { try { r = std::stoi(v); } catch (...) {} }
    else if (k == "L") { try { l = std::stoi(v); } catch (...) {} }
    else if (k == "params") {
      picojson::value pv;
      std::string perr = picojson::parse(pv, v);
      if (!perr.empty() || !pv.is<picojson::object>()) return;
      const auto &po = pv.get<picojson::object>();
      int pm = obj_int(po, "M"), pe = obj_int(po, "efConstruction");
      int pr = obj_int(po, "R"), pl = obj_int(po, "L");
      if (pm > 0) m = pm;
      if (pe > 0) ef = pe;
      if (pr > 0) r = pr;
      if (pl > 0) l = pl;
      auto it = po.find("index_type");
      if (index_type.empty() && it != po.end() && it->second.is<std::string>())
        index_type = it->second.get<std::string>();
    }
  };
  for (auto &p : req->extra_params()) parse_kv(p.key(), p.value());

  auto to_upper = [](std::string s) {
    for (auto &c : s) c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
    return s;
  };

  uint32_t build_R = 0, build_L = 0;
  if (to_upper(index_type) == "DISKANN") {
    // PipeANN-native knobs pass through unchanged.
    build_R = (r > 0) ? static_cast<uint32_t>(r) : 0;
    build_L = (l > 0) ? static_cast<uint32_t>(l) : 0;
  } else {
    // HNSW-style knobs map onto PipeANN's single-layer graph.
    build_R = (m > 0) ? static_cast<uint32_t>(2 * m) : 0;
    build_L = (ef > 0) ? static_cast<uint32_t>(ef) : 0;
  }
  try {
    store_->build_index(req->collection_name(), build_R, build_L);
  } catch (const std::exception &e) {
    *resp = err_status(std::string("index build failed: ") + e.what());
    return Status::OK;
  }
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::DropIndex(grpc::ServerContext *, const pb_milvus::DropIndexRequest *req,
                                    pb_common::Status *resp) {
  // No-op: PipeANN keeps the index for the lifetime of the collection. We only
  // validate the collection exists so clients get a sensible error otherwise.
  if (!store_->has_collection(req->collection_name())) {
    *resp = err_status("collection not found");
    return Status::OK;
  }
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::DescribeIndex(grpc::ServerContext *, const pb_milvus::DescribeIndexRequest *req,
                                        pb_milvus::DescribeIndexResponse *resp) {
  auto meta = store_->describe(req->collection_name());
  if (!meta) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  auto *desc = resp->add_index_descriptions();
  desc->set_index_name("pipeann");
  desc->set_field_name(meta->vector_field);
  desc->set_state(pb_common::IndexState::Finished);
  int64_t n = store_->row_count(req->collection_name());
  desc->set_indexed_rows(n);
  desc->set_total_rows(n);
  return Status::OK;
}

Status MilvusServiceImpl::GetIndexState(grpc::ServerContext *, const pb_milvus::GetIndexStateRequest *req,
                                        pb_milvus::GetIndexStateResponse *resp) {
  if (!store_->has_collection(req->collection_name())) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  resp->set_state(pb_common::IndexState::Finished);
  return Status::OK;
}

Status MilvusServiceImpl::GetIndexBuildProgress(grpc::ServerContext *,
                                                const pb_milvus::GetIndexBuildProgressRequest *req,
                                                pb_milvus::GetIndexBuildProgressResponse *resp) {
  if (!store_->has_collection(req->collection_name())) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  int64_t n = store_->row_count(req->collection_name());
  resp->set_indexed_rows(n);
  resp->set_total_rows(n);
  return Status::OK;
}

Status MilvusServiceImpl::LoadCollection(grpc::ServerContext *, const pb_milvus::LoadCollectionRequest *req,
                                         pb_common::Status *resp) {
  try {
    store_->build_index(req->collection_name());
  } catch (const std::exception &e) {
    *resp = err_status(std::string("index build failed: ") + e.what());
    return Status::OK;
  }
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::GetLoadingProgress(grpc::ServerContext *, const pb_milvus::GetLoadingProgressRequest *req,
                                             pb_milvus::GetLoadingProgressResponse *resp) {
  if (!store_->has_collection(req->collection_name())) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  resp->set_progress(100);
  return Status::OK;
}

Status MilvusServiceImpl::ReleaseCollection(grpc::ServerContext *, const pb_milvus::ReleaseCollectionRequest *,
                                            pb_common::Status *resp) {
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::GetLoadState(grpc::ServerContext *, const pb_milvus::GetLoadStateRequest *,
                                       pb_milvus::GetLoadStateResponse *resp) {
  *resp->mutable_status() = ok_status();
  resp->set_state(pb_common::LoadStateLoaded);
  return Status::OK;
}

Status MilvusServiceImpl::Flush(grpc::ServerContext *, const pb_milvus::FlushRequest *req,
                                pb_milvus::FlushResponse *resp) {
  for (const auto &name : req->collection_names()) {
    store_->flush(name);
    // Flush is synchronous; these fields only exist so pymilvus can poll
    // GetFlushState without failing.
    (*resp->mutable_coll_segids())[name];
    (*resp->mutable_flush_coll_segids())[name];
    (*resp->mutable_coll_flush_ts())[name] = 1;
  }
  *resp->mutable_status() = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::GetFlushState(grpc::ServerContext *, const pb_milvus::GetFlushStateRequest *req,
                                        pb_milvus::GetFlushStateResponse *resp) {
  if (!req->collection_name().empty() && !store_->has_collection(req->collection_name())) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  // Flush completes before the RPC returns, so any follow-up state check is
  // immediately satisfied.
  resp->set_flushed(true);
  return Status::OK;
}

// ---------------------------------------------------------------------------
// Partitions (compatibility stubs)
// ---------------------------------------------------------------------------
// PipeANN stores all rows of a collection together; partition-scoped queries are
// expressed through the attribute filter path instead. These RPCs let pymilvus
// partition calls succeed by treating the whole collection as a single implicit
// "_default" partition. We validate the collection exists so clients still get a
// sensible error for an unknown collection.
static constexpr const char *kDefaultPartition = "_default";

Status MilvusServiceImpl::CreatePartition(grpc::ServerContext *, const pb_milvus::CreatePartitionRequest *req,
                                          pb_common::Status *resp) {
  *resp = store_->has_collection(req->collection_name()) ? ok_status() : err_status("collection not found");
  return Status::OK;
}

Status MilvusServiceImpl::DropPartition(grpc::ServerContext *, const pb_milvus::DropPartitionRequest *req,
                                        pb_common::Status *resp) {
  *resp = store_->has_collection(req->collection_name()) ? ok_status() : err_status("collection not found");
  return Status::OK;
}

Status MilvusServiceImpl::HasPartition(grpc::ServerContext *, const pb_milvus::HasPartitionRequest *req,
                                       pb_milvus::BoolResponse *resp) {
  *resp->mutable_status() = ok_status();
  // Any partition of an existing collection is reported as present.
  resp->set_value(store_->has_collection(req->collection_name()));
  return Status::OK;
}

Status MilvusServiceImpl::LoadPartitions(grpc::ServerContext *, const pb_milvus::LoadPartitionsRequest *req,
                                         pb_common::Status *resp) {
  // Loading a partition is equivalent to loading the collection.
  try {
    store_->build_index(req->collection_name());
  } catch (const std::exception &e) {
    *resp = err_status(std::string("index build failed: ") + e.what());
    return Status::OK;
  }
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::ReleasePartitions(grpc::ServerContext *, const pb_milvus::ReleasePartitionsRequest *,
                                            pb_common::Status *resp) {
  *resp = ok_status();
  return Status::OK;
}

Status MilvusServiceImpl::GetPartitionStatistics(grpc::ServerContext *,
                                                 const pb_milvus::GetPartitionStatisticsRequest *req,
                                                 pb_milvus::GetPartitionStatisticsResponse *resp) {
  if (!store_->has_collection(req->collection_name())) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  auto *kv = resp->add_stats();
  kv->set_key("row_count");
  kv->set_value(std::to_string(store_->row_count(req->collection_name())));
  return Status::OK;
}

Status MilvusServiceImpl::ShowPartitions(grpc::ServerContext *, const pb_milvus::ShowPartitionsRequest *req,
                                         pb_milvus::ShowPartitionsResponse *resp) {
  if (!store_->has_collection(req->collection_name())) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  *resp->mutable_status() = ok_status();
  resp->add_partition_names(kDefaultPartition);
  resp->add_partitionids(0);
  resp->add_created_timestamps(0);
  resp->add_created_utc_timestamps(0);
  return Status::OK;
}

// ---------------------------------------------------------------------------
// Insert / Upsert / Delete
// ---------------------------------------------------------------------------
// Build a name->FieldKind map and primary/vector field names from meta.
bool MilvusServiceImpl::decode_insert_columns(
    const CollectionMeta &meta, const google::protobuf::RepeatedPtrField<pb_schema::FieldData> &fields_data,
    uint32_t num_rows, InsertColumns *out) {
  out->num_rows = num_rows;

  // Map each field name to its neutral kind (from the schema).
  std::map<std::string, FieldKind> kind_by_name;
  for (const auto &f : meta.fields) kind_by_name[f.name] = f.kind;

  for (const auto &fd : fields_data) {
    const std::string &fname = fd.field_name();
    if (fd.type() == pb_schema::DataType::FloatVector ||
        (kind_by_name.count(fname) && kind_by_name[fname] == FieldKind::FloatVector)) {
      const auto &fv = fd.vectors().float_vector();
      out->vectors.assign(fv.data().begin(), fv.data().end());
      continue;
    }

    if (fname == meta.primary_field) {
      if (fd.scalars().has_long_data()) {
        for (auto v : fd.scalars().long_data().data()) out->pks.push_back(std::to_string(v));
      } else if (fd.scalars().has_string_data()) {
        for (const auto &s : fd.scalars().string_data().data()) out->pks.push_back(s);
      }
      continue;
    }

    // Scalar / display field -> a neutral ScalarColumn.
    auto kit = kind_by_name.find(fname);
    FieldKind kind = kit != kind_by_name.end() ? kit->second : FieldKind::Json;
    ScalarColumn col;
    col.kind = kind;
    const auto &sc = fd.scalars();
    switch (kind) {
      case FieldKind::ArrayInt:
        if (sc.has_array_data()) {
          for (const auto &row : sc.array_data().data()) {
            std::vector<int64_t> labels;
            if (row.has_long_data())
              for (auto v : row.long_data().data()) labels.push_back(v);
            else if (row.has_int_data())
              for (auto v : row.int_data().data()) labels.push_back(v);
            col.int_arrays.push_back(std::move(labels));
          }
        }
        break;
      case FieldKind::Float:
        if (sc.has_float_data()) for (auto v : sc.float_data().data()) col.doubles.push_back(v);
        else if (sc.has_double_data()) for (auto v : sc.double_data().data()) col.doubles.push_back(v);
        else if (sc.has_long_data()) for (auto v : sc.long_data().data()) col.doubles.push_back((double) v);
        break;
      case FieldKind::Bool:
        if (sc.has_bool_data()) for (auto v : sc.bool_data().data()) col.bools.push_back(v);
        break;
      case FieldKind::VarChar:
        if (sc.has_string_data()) for (const auto &s : sc.string_data().data()) col.strings.push_back(s);
        break;
      case FieldKind::Json:  // display-only: store raw, type-tagged for the doc JSON
        if (sc.has_long_data()) { col.kind = FieldKind::Int64; for (auto v : sc.long_data().data()) col.ints.push_back(v); }
        else if (sc.has_int_data()) { col.kind = FieldKind::Int64; for (auto v : sc.int_data().data()) col.ints.push_back(v); }
        else if (sc.has_double_data()) { col.kind = FieldKind::Float; for (auto v : sc.double_data().data()) col.doubles.push_back(v); }
        else if (sc.has_float_data()) { col.kind = FieldKind::Float; for (auto v : sc.float_data().data()) col.doubles.push_back(v); }
        else if (sc.has_bool_data()) { col.kind = FieldKind::Bool; for (auto v : sc.bool_data().data()) col.bools.push_back(v); }
        else if (sc.has_string_data()) { col.kind = FieldKind::VarChar; for (const auto &s : sc.string_data().data()) col.strings.push_back(s); }
        else if (sc.has_json_data()) { col.kind = FieldKind::VarChar; for (const auto &s : sc.json_data().data()) col.strings.push_back(s); }
        break;
      default:  // Int64
        col.kind = FieldKind::Int64;
        if (sc.has_long_data()) for (auto v : sc.long_data().data()) col.ints.push_back(v);
        else if (sc.has_int_data()) for (auto v : sc.int_data().data()) col.ints.push_back(v);
        break;
    }
    out->scalars[fname] = std::move(col);
  }

  // pks default to stringified row index if no primary column was sent.
  if (out->pks.empty()) {
    for (uint32_t i = 0; i < num_rows; i++) out->pks.push_back(std::to_string(i));
  }
  return true;
}

Status MilvusServiceImpl::Insert(grpc::ServerContext *, const pb_milvus::InsertRequest *req,
                                 pb_milvus::MutationResult *resp) {
  auto meta = store_->describe(req->collection_name());
  if (!meta) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  InsertColumns cols;
  decode_insert_columns(*meta, req->fields_data(), req->num_rows(), &cols);
  try {
    auto res = store_->insert(req->collection_name(), cols);
    *resp->mutable_status() = ok_status();
    resp->set_insert_cnt(res.count);
    auto *out_ids = resp->mutable_ids()->mutable_str_id();
    for (auto &id : res.ids) out_ids->add_data(id);
  } catch (const std::exception &e) {
    *resp->mutable_status() = err_status(e.what());
  }
  return Status::OK;
}

Status MilvusServiceImpl::Upsert(grpc::ServerContext *, const pb_milvus::UpsertRequest *req,
                                 pb_milvus::MutationResult *resp) {
  auto meta = store_->describe(req->collection_name());
  if (!meta) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }
  InsertColumns cols;
  decode_insert_columns(*meta, req->fields_data(), req->num_rows(), &cols);
  try {
    auto res = store_->upsert(req->collection_name(), cols);
    *resp->mutable_status() = ok_status();
    resp->set_upsert_cnt(res.count);
    auto *out_ids = resp->mutable_ids()->mutable_str_id();
    for (auto &id : res.ids) out_ids->add_data(id);
  } catch (const std::exception &e) {
    *resp->mutable_status() = err_status(e.what());
  }
  return Status::OK;
}

Status MilvusServiceImpl::Delete(grpc::ServerContext *, const pb_milvus::DeleteRequest *req,
                                 pb_milvus::MutationResult *resp) {
  try {
    int64_t deleted = store_->remove(req->collection_name(), /*explicit_pks=*/{}, req->expr());
    *resp->mutable_status() = ok_status();
    resp->set_delete_cnt(deleted);
  } catch (const std::exception &e) {
    *resp->mutable_status() = err_status(e.what());
  }
  return Status::OK;
}

// ---------------------------------------------------------------------------
// Output marshalling: neutral OutputColumn -> protobuf FieldData
// ---------------------------------------------------------------------------
void MilvusServiceImpl::encode_output_columns(const QueryResult &qr,
                                              google::protobuf::RepeatedPtrField<pb_schema::FieldData> *out) {
  for (const auto &oc : qr.columns) {
    auto *fd = out->Add();
    fd->set_field_name(oc.name);
    switch (oc.kind) {
      case FieldKind::FloatVector: {
        fd->set_type(pb_schema::DataType::FloatVector);
        auto *vf = fd->mutable_vectors();
        vf->set_dim(oc.dim);
        auto *farr = vf->mutable_float_vector();
        for (float c : oc.vectors) farr->add_data(c);
        break;
      }
      case FieldKind::Float: {
        fd->set_type(pb_schema::DataType::Double);
        auto *dd = fd->mutable_scalars()->mutable_double_data();
        for (double v : oc.scalar.doubles) dd->add_data(v);
        break;
      }
      case FieldKind::Bool: {
        fd->set_type(pb_schema::DataType::Bool);
        auto *bd = fd->mutable_scalars()->mutable_bool_data();
        for (bool v : oc.scalar.bools) bd->add_data(v);
        break;
      }
      case FieldKind::VarChar: {
        fd->set_type(pb_schema::DataType::VarChar);
        auto *sd = fd->mutable_scalars()->mutable_string_data();
        for (const auto &s : oc.scalar.strings) sd->add_data(s);
        break;
      }
      case FieldKind::ArrayInt: {
        fd->set_type(pb_schema::DataType::Array);
        auto *ad = fd->mutable_scalars()->mutable_array_data();
        ad->set_element_type(pb_schema::DataType::Int64);
        for (const auto &labels : oc.scalar.int_arrays) {
          auto *row = ad->add_data();
          auto *ld = row->mutable_long_data();
          for (int64_t v : labels) ld->add_data(v);
        }
        break;
      }
      case FieldKind::Int64:
      default: {
        fd->set_type(pb_schema::DataType::Int64);
        auto *ld = fd->mutable_scalars()->mutable_long_data();
        for (int64_t v : oc.scalar.ints) ld->add_data(v);
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
Status MilvusServiceImpl::Search(grpc::ServerContext *, const pb_milvus::SearchRequest *req,
                                 pb_milvus::SearchResults *resp) {
  auto meta = store_->describe(req->collection_name());
  if (!meta) {
    *resp->mutable_status() = err_status("collection not found");
    return Status::OK;
  }

  SearchParams params;
  params.topk = 10;
  params.L = 50;
  auto extract_int = [](const std::string &json, const std::string &key) -> int {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return -1;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '"')) pos++;
    try { return std::stoi(json.substr(pos)); } catch (...) { return -1; }
  };
  for (auto &p : req->search_params()) {
    if (p.key() == "topk") {
      params.topk = std::stoi(p.value());
    } else if (p.key() == "L" || p.key() == "ef") {
      params.L = std::stoi(p.value());
    } else if (p.key() == "params") {
      int v = extract_int(p.value(), "L");
      if (v < 0) v = extract_int(p.value(), "ef");
      if (v > 0) params.L = v;
    }
  }

  pb_common::PlaceholderGroup pg;
  pg.ParseFromString(req->placeholder_group());
  if (pg.placeholders_size() == 0) {
    *resp->mutable_status() = err_status("no query vectors");
    return Status::OK;
  }
  auto &ph = pg.placeholders(0);
  params.n_queries = ph.values_size();
  uint32_t dim = meta->dim;
  params.queries.resize((size_t) params.n_queries * dim);
  for (int i = 0; i < params.n_queries; i++) {
    auto &v = ph.values(i);
    std::memcpy(params.queries.data() + (size_t) i * dim, v.data(), dim * sizeof(float));
  }

  params.filter = req->dsl();
  params.output_fields.assign(req->output_fields().begin(), req->output_fields().end());

  QueryResult qr;
  try {
    qr = store_->search(req->collection_name(), params);
  } catch (const std::exception &e) {
    *resp->mutable_status() = err_status(e.what());
    return Status::OK;
  }

  *resp->mutable_status() = ok_status();
  auto *data = resp->mutable_results();
  data->set_num_queries(params.n_queries);
  data->set_top_k(params.topk);
  auto *ids = data->mutable_ids()->mutable_str_id();
  for (const auto &id : qr.ids) ids->add_data(id);
  for (float s : qr.scores) data->add_scores(s);
  for (int64_t t : qr.topks) data->add_topks(t);
  if (!qr.columns.empty()) {
    encode_output_columns(qr, data->mutable_fields_data());
    for (const auto &f : qr.output_field_names) data->add_output_fields(f);
  }
  return Status::OK;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------
Status MilvusServiceImpl::Query(grpc::ServerContext *, const pb_milvus::QueryRequest *req,
                                pb_milvus::QueryResults *resp) {
  // count(*) shortcut.
  for (auto &f : req->output_fields()) {
    if (f == "count(*)") {
      int64_t cnt;
      try {
        cnt = store_->count(req->collection_name(), req->expr());
      } catch (const std::exception &e) {
        *resp->mutable_status() = err_status(e.what());
        return Status::OK;
      }
      *resp->mutable_status() = ok_status();
      auto *fd = resp->add_fields_data();
      fd->set_field_name("count(*)");
      fd->set_type(pb_schema::DataType::Int64);
      fd->mutable_scalars()->mutable_long_data()->add_data(cnt);
      return Status::OK;
    }
  }

  QueryParams params;
  params.filter = req->expr();
  params.output_fields.assign(req->output_fields().begin(), req->output_fields().end());
  for (const auto &p : req->query_params()) {
    if (p.key() == "limit") {
      try { params.limit = static_cast<size_t>(std::stoll(p.value())); } catch (...) {}
    }
  }

  QueryResult qr;
  try {
    qr = store_->query(req->collection_name(), params);
  } catch (const std::exception &e) {
    *resp->mutable_status() = err_status(e.what());
    return Status::OK;
  }

  *resp->mutable_status() = ok_status();
  encode_output_columns(qr, resp->mutable_fields_data());
  for (const auto &f : qr.output_field_names) resp->add_output_fields(f);
  resp->set_primary_field_name(qr.primary_field_name);
  return Status::OK;
}

}  // namespace server
}  // namespace pipeann
