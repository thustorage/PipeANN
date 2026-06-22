// pipeann_milvus_server: standalone gRPC server binary.
#include "milvus_server.h"

#include <grpcpp/grpcpp.h>
#include <csignal>
#include <iostream>
#include <string>

static std::unique_ptr<grpc::Server> g_server;

static void signal_handler(int) {
  if (g_server) g_server->Shutdown();
}

int main(int argc, char **argv) {
  std::string data_dir = "./data";
  std::string host = "0.0.0.0";
  int port = 19530;
  int threads = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--data_dir" && i + 1 < argc) data_dir = argv[++i];
    else if (arg == "--host" && i + 1 < argc) host = argv[++i];
    else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
    else if (arg == "--help") {
      std::cout << "Usage: pipeann_milvus_server [--data_dir DIR] [--host HOST] [--port PORT] [--threads N]\n";
      return 0;
    }
  }

  std::string addr = host + ":" + std::to_string(port);

  pipeann::server::MilvusServiceImpl service(data_dir, threads);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize(256 * 1024 * 1024);  // 256MB for large inserts

  g_server = builder.BuildAndStart();
  if (!g_server) {
    std::cerr << "Failed to start server on " << addr << std::endl;
    return 1;
  }

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  std::cout << "PipeANN Milvus gRPC server on " << addr << std::endl;
  g_server->Wait();
  return 0;
}
