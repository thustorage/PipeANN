// PipeANN Milvus-compatible gRPC server.
// Thin protobuf <-> neutral marshalling adapter over CollectionStore (the
// engine). All data-model logic lives in collection_store.{h,cpp}; this class
// only translates Milvus protobuf messages to/from the neutral types.
#pragma once

#include <memory>
#include <string>

#include "collection_store.h"
#include "milvus.grpc.pb.h"

namespace pipeann {
namespace server {

namespace pb_common = milvus::proto::common;
namespace pb_schema = milvus::proto::schema;
namespace pb_milvus = milvus::proto::milvus;

class MilvusServiceImpl final : public pb_milvus::MilvusService::Service {
 public:
  explicit MilvusServiceImpl(const std::string &data_dir, int omp_threads = 0);
  ~MilvusServiceImpl() override;

  // Connection
  grpc::Status Connect(grpc::ServerContext *ctx, const pb_milvus::ConnectRequest *req,
                       pb_milvus::ConnectResponse *resp) override;

  // Collection CRUD
  grpc::Status CreateCollection(grpc::ServerContext *ctx, const pb_milvus::CreateCollectionRequest *req,
                                pb_common::Status *resp) override;
  grpc::Status DropCollection(grpc::ServerContext *ctx, const pb_milvus::DropCollectionRequest *req,
                              pb_common::Status *resp) override;
  grpc::Status HasCollection(grpc::ServerContext *ctx, const pb_milvus::HasCollectionRequest *req,
                             pb_milvus::BoolResponse *resp) override;
  grpc::Status DescribeCollection(grpc::ServerContext *ctx, const pb_milvus::DescribeCollectionRequest *req,
                                  pb_milvus::DescribeCollectionResponse *resp) override;
  grpc::Status ShowCollections(grpc::ServerContext *ctx, const pb_milvus::ShowCollectionsRequest *req,
                               pb_milvus::ShowCollectionsResponse *resp) override;
  grpc::Status GetCollectionStatistics(grpc::ServerContext *ctx, const pb_milvus::GetCollectionStatisticsRequest *req,
                                       pb_milvus::GetCollectionStatisticsResponse *resp) override;

  // Index / Load. PipeANN builds its index from buffered inserts; these trigger
  // the build and report "ready" so pymilvus polling helpers succeed.
  grpc::Status CreateIndex(grpc::ServerContext *ctx, const pb_milvus::CreateIndexRequest *req,
                           pb_common::Status *resp) override;
  grpc::Status DropIndex(grpc::ServerContext *ctx, const pb_milvus::DropIndexRequest *req,
                         pb_common::Status *resp) override;
  grpc::Status DescribeIndex(grpc::ServerContext *ctx, const pb_milvus::DescribeIndexRequest *req,
                             pb_milvus::DescribeIndexResponse *resp) override;
  grpc::Status GetIndexState(grpc::ServerContext *ctx, const pb_milvus::GetIndexStateRequest *req,
                             pb_milvus::GetIndexStateResponse *resp) override;
  grpc::Status GetIndexBuildProgress(grpc::ServerContext *ctx, const pb_milvus::GetIndexBuildProgressRequest *req,
                                     pb_milvus::GetIndexBuildProgressResponse *resp) override;
  grpc::Status LoadCollection(grpc::ServerContext *ctx, const pb_milvus::LoadCollectionRequest *req,
                              pb_common::Status *resp) override;
  grpc::Status ReleaseCollection(grpc::ServerContext *ctx, const pb_milvus::ReleaseCollectionRequest *req,
                                 pb_common::Status *resp) override;
  grpc::Status GetLoadState(grpc::ServerContext *ctx, const pb_milvus::GetLoadStateRequest *req,
                            pb_milvus::GetLoadStateResponse *resp) override;
  grpc::Status GetLoadingProgress(grpc::ServerContext *ctx, const pb_milvus::GetLoadingProgressRequest *req,
                                  pb_milvus::GetLoadingProgressResponse *resp) override;
  grpc::Status Flush(grpc::ServerContext *ctx, const pb_milvus::FlushRequest *req,
                     pb_milvus::FlushResponse *resp) override;
  grpc::Status GetFlushState(grpc::ServerContext *ctx, const pb_milvus::GetFlushStateRequest *req,
                             pb_milvus::GetFlushStateResponse *resp) override;

  // Partitions. PipeANN has no partition concept (partition-scoped filtering is
  // handled through the attribute filter path), so these are compatibility
  // stubs: a collection behaves as a single implicit "_default" partition.
  grpc::Status CreatePartition(grpc::ServerContext *ctx, const pb_milvus::CreatePartitionRequest *req,
                               pb_common::Status *resp) override;
  grpc::Status DropPartition(grpc::ServerContext *ctx, const pb_milvus::DropPartitionRequest *req,
                             pb_common::Status *resp) override;
  grpc::Status HasPartition(grpc::ServerContext *ctx, const pb_milvus::HasPartitionRequest *req,
                            pb_milvus::BoolResponse *resp) override;
  grpc::Status LoadPartitions(grpc::ServerContext *ctx, const pb_milvus::LoadPartitionsRequest *req,
                              pb_common::Status *resp) override;
  grpc::Status ReleasePartitions(grpc::ServerContext *ctx, const pb_milvus::ReleasePartitionsRequest *req,
                                 pb_common::Status *resp) override;
  grpc::Status GetPartitionStatistics(grpc::ServerContext *ctx, const pb_milvus::GetPartitionStatisticsRequest *req,
                                      pb_milvus::GetPartitionStatisticsResponse *resp) override;
  grpc::Status ShowPartitions(grpc::ServerContext *ctx, const pb_milvus::ShowPartitionsRequest *req,
                              pb_milvus::ShowPartitionsResponse *resp) override;

  // Data ops
  grpc::Status Insert(grpc::ServerContext *ctx, const pb_milvus::InsertRequest *req,
                      pb_milvus::MutationResult *resp) override;
  grpc::Status Upsert(grpc::ServerContext *ctx, const pb_milvus::UpsertRequest *req,
                      pb_milvus::MutationResult *resp) override;
  grpc::Status Delete(grpc::ServerContext *ctx, const pb_milvus::DeleteRequest *req,
                      pb_milvus::MutationResult *resp) override;
  grpc::Status Search(grpc::ServerContext *ctx, const pb_milvus::SearchRequest *req,
                      pb_milvus::SearchResults *resp) override;
  grpc::Status Query(grpc::ServerContext *ctx, const pb_milvus::QueryRequest *req,
                     pb_milvus::QueryResults *resp) override;

  // Misc
  grpc::Status GetVersion(grpc::ServerContext *ctx, const pb_milvus::GetVersionRequest *req,
                          pb_milvus::GetVersionResponse *resp) override;

 private:
  // Decode a protobuf InsertRequest/UpsertRequest fields_data into neutral
  // columns against the collection's schema. Returns false on a structural error.
  bool decode_insert_columns(const CollectionMeta &meta,
                             const google::protobuf::RepeatedPtrField<pb_schema::FieldData> &fields_data,
                             uint32_t num_rows, InsertColumns *out);
  // Marshal a neutral QueryResult's columns into protobuf FieldData.
  void encode_output_columns(const QueryResult &qr,
                             google::protobuf::RepeatedPtrField<pb_schema::FieldData> *out);

  std::unique_ptr<CollectionStore> store_;
};

}  // namespace server
}  // namespace pipeann
