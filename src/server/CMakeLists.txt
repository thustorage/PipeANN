# Optional Milvus gRPC server target.
# Enable with: cmake -DBUILD_MILVUS_SERVER=ON ..

find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
pkg_check_modules(PROTOBUF REQUIRED protobuf)

# Proto source directory
set(PROTO_DIR "${PROJECT_SOURCE_DIR}/proto/milvus")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# Find protoc and grpc_cpp_plugin
find_program(PROTOC protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# Proto files to compile
set(PROTO_FILES
    common.proto
    schema.proto
    milvus.proto
    msg.proto
    feder.proto
    rg.proto
)

# Generate C++ from proto files
set(PROTO_SRCS "")
foreach(PROTO ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO} NAME_WE)
    set(PB_CC "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.cc")
    set(PB_H "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.h")
    set(GRPC_CC "${PROTO_GEN_DIR}/${PROTO_NAME}.grpc.pb.cc")
    set(GRPC_H "${PROTO_GEN_DIR}/${PROTO_NAME}.grpc.pb.h")
    add_custom_command(
        OUTPUT ${PB_CC} ${PB_H} ${GRPC_CC} ${GRPC_H}
        COMMAND ${PROTOC}
            --experimental_allow_proto3_optional
            --cpp_out=${PROTO_GEN_DIR}
            --grpc_out=${PROTO_GEN_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            -I${PROTO_DIR} -I/usr/include
            ${PROTO_DIR}/${PROTO}
        DEPENDS ${PROTO_DIR}/${PROTO}
        COMMENT "Generating C++ from ${PROTO}"
    )
    list(APPEND PROTO_SRCS ${PB_CC} ${GRPC_CC})
endforeach()

# Server sources
add_executable(pipeann_milvus_server
    main.cpp
    milvus_server.cpp
    collection_store.cpp
    ${PROTO_SRCS}
)

target_include_directories(pipeann_milvus_server PRIVATE
    ${PROTO_GEN_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/tsl/include
    ${GRPC_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIRS}
)

# Suppress warnings in generated code
set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS "-w")

find_library(ROCKSDB_LIB rocksdb REQUIRED)

target_link_libraries(pipeann_milvus_server PRIVATE
    pipeann
    pipeann_tcmalloc
    ${ROCKSDB_LIB}
    ${GRPC_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    pthread
)

target_compile_options(pipeann_milvus_server PRIVATE -w)
