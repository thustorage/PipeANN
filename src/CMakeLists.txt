file(GLOB PROJECT_CPP *.cpp search/*.cpp update/*.cpp utils/*.cpp)
add_library(${PROJECT_NAME} ${PROJECT_CPP})

if(IO_ENGINE STREQUAL "spdk")
    find_package(PkgConfig REQUIRED)
    set(SPDK_PATH "${PROJECT_SOURCE_DIR}/third_party/spdk")
    if(NOT EXISTS "${SPDK_PATH}/build/lib/pkgconfig")
        message(FATAL_ERROR "SPDK not found at ${SPDK_PATH}. Clone it there or create a symlink, e.g. ln -s /path/to/spdk third_party/spdk")
    endif()
    set(ENV{PKG_CONFIG_PATH} "${SPDK_PATH}/build/lib/pkgconfig")
    execute_process(COMMAND pkg-config --libs --static spdk_nvme spdk_env_dpdk OUTPUT_VARIABLE SPDK_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND pkg-config --libs --static spdk_syslibs OUTPUT_VARIABLE SPDK_SYSLIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(SPDK_LIBRARY UNIX_COMMAND "${SPDK_LIBRARY}")
    separate_arguments(SPDK_SYSLIBS UNIX_COMMAND "${SPDK_SYSLIBS}")
    target_include_directories(${PROJECT_NAME} PUBLIC ${SPDK_PATH}/build/include)
    target_link_directories(${PROJECT_NAME} PUBLIC ${SPDK_PATH}/dpdk/build/lib)
    target_link_options(${PROJECT_NAME} PUBLIC
        -pthread
        "SHELL:-Wl,--whole-archive"
        "SHELL:-Wl,-Bstatic"
        ${SPDK_LIBRARY}
        "SHELL:-Wl,-Bdynamic"
        "SHELL:-Wl,--no-whole-archive"
        ${SPDK_SYSLIBS}
    )
    target_link_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/liburing/src)
    target_link_libraries(${PROJECT_NAME} PUBLIC uring)
elseif(IO_ENGINE STREQUAL "uring")
    target_link_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/liburing/src)
    target_link_libraries(${PROJECT_NAME} PUBLIC uring)
elseif(IO_ENGINE STREQUAL "aio")
    target_link_libraries(${PROJECT_NAME} PUBLIC aio)
endif()


target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)

# python module.
# pip install "pybind11[global]"
if(BUILD_PYTHON_INTERFACE)
    find_package(pybind11 REQUIRED)
    find_library(ROCKSDB_LIB rocksdb REQUIRED)
    file(GLOB PYBIND_CPP python/*.cpp)
    # The neutral collection engine is shared with the gRPC server; compile it
    # into the Python module so MilvusClient drives the same data model.
    list(APPEND PYBIND_CPP ${CMAKE_CURRENT_SOURCE_DIR}/server/collection_store.cpp)
    pybind11_add_module(C NO_EXTRAS ${PYBIND_CPP})
    target_include_directories(C PRIVATE ${PROJECT_SOURCE_DIR}/src/server)
    target_link_libraries(C PRIVATE pipeann ${ROCKSDB_LIB})
endif()
