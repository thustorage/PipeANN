cmake_minimum_required(VERSION 3.5)
project(pipeann)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(IO_ENGINE "uring" CACHE STRING "I/O engine: uring, aio, or spdk")
set_property(CACHE IO_ENGINE PROPERTY STRINGS uring aio spdk)
option(USE_TCMALLOC "Use tcmalloc allocator" ON)
option(BUILD_PYTHON_INTERFACE "Build Python interface (requires pybind11)" OFF)
option(BUILD_MILVUS_SERVER "Build Milvus-compatible gRPC server (requires grpc++)" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND CMAKE_VERSION VERSION_LESS "3.4")
	# CMake 3.0 does not support compiler launchers, so fall back to the global rule.
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
	message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
elseif(CCACHE_PROGRAM)
	set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
	set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
	message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
else()
	message(STATUS "ccache not found, building without compiler cache")
endif()

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
)

add_compile_options(
	-fopenmp 
	-fopenmp-simd
	-Wall
	-Wextra
	-Wfatal-errors
	-Wno-class-memaccess
	-Wno-unused-variable
	-Wno-unused-parameter
	-Wno-unused-but-set-variable
)
add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops)

message(STATUS "Additional definitions: " $ENV{ADDITIONAL_DEFINITIONS})
add_definitions(
	$ENV{ADDITIONAL_DEFINITIONS}
	-DNDEBUG
	# -DIN_PLACE_RECORD_UPDATE # disable design #1
	-DBG_IO_THREAD # enable opt #1.2

	# -DNO_POLLUTE_ORIGINAL

	# For read-only tests without updates, enable the following defs:
	# -DREAD_ONLY_TESTS
	# -DNO_MAPPING # page_search (Starling): must disable this!
)

# add_compile_options(-fsanitize=address,undefined -lasan -lubsan)
# add_link_options(-fsanitize=address,undefined)

# check if uring is available.
try_run(
	PIPEANN_IO_URING_RUN_RESULT
	PIPEANN_IO_URING_COMPILE_RESULT
	"${CMAKE_BINARY_DIR}"
	SOURCES "${PROJECT_SOURCE_DIR}/tests/test_uring.cpp"
)

if(IO_ENGINE STREQUAL "aio" OR NOT PIPEANN_IO_URING_COMPILE_RESULT OR NOT "${PIPEANN_IO_URING_RUN_RESULT}" STREQUAL "0")
	# AIO is specified or uring is unavailable.
	set(IO_ENGINE "aio" CACHE STRING "I/O engine: uring, aio, or spdk" FORCE)
	message(STATUS "Using AIO for asynchronous I/O, compatible but slower.")
else()
	include_directories(${PROJECT_SOURCE_DIR}/third_party/liburing/src/include)
	link_directories(${PROJECT_SOURCE_DIR}/third_party/liburing/src)
	if(IO_ENGINE STREQUAL "spdk")
		add_definitions(-DUSE_SPDK)
		message(STATUS "Using SPDK for vector I/O & liburing for attr I/O, fastest but require user-space bdev.")
	elseif(IO_ENGINE STREQUAL "uring")
		add_definitions(-DUSE_URING)
		message(STATUS "Using liburing for asynchronous I/O, faster but requires kernel support.")
	else()
		message(FATAL_ERROR "Invalid IO_ENGINE=${IO_ENGINE}. Expected uring, aio, or spdk.")
	endif()
endif()

add_library(pipeann_tcmalloc INTERFACE)
if(USE_TCMALLOC)
	add_definitions(-DUSE_TCMALLOC)
	target_link_libraries(pipeann_tcmalloc INTERFACE
		-Wl,--no-as-needed tcmalloc -Wl,--as-needed)
	message(STATUS "Using tcmalloc allocator (C++ executables only).")
else()
	message(STATUS "Building without tcmalloc allocator.")
endif()

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
# test_x86_avx.cpp returns a bitmask: bit0=AVX2, bit1=AVX512F, bit2=AVX512VPOPCNTDQ
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
set(CPU_SUPPORTS_AVX512_VPOPCNTDQ FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
	try_run(
		RUN_RESULT
		COMPILE_RESULT
		"${CMAKE_BINARY_DIR}"
		SOURCES "${PROJECT_SOURCE_DIR}/tests/test_x86_avx.cpp"
		COMPILE_OUTPUT_VARIABLE COMPILE_OUT
		RUN_OUTPUT_VARIABLE RUN_OUT
	)
else()
	set(RUN_RESULT 0)
	set(COMPILE_RESULT FALSE)
endif()

# RUN_RESULT is a bitmask: bit0=AVX2, bit1=AVX512F, bit2=AVX512VPOPCNTDQ
if(COMPILE_RESULT)
	math(EXPR HAS_AVX2 "${RUN_RESULT} & 1")
	math(EXPR HAS_AVX512F "${RUN_RESULT} & 2")
	math(EXPR HAS_VPOPCNTDQ "${RUN_RESULT} & 4")
	if(HAS_AVX2)
		set(CPU_SUPPORTS_AVX2 TRUE)
		message(STATUS "CPU supports AVX2")
	endif()
	if(HAS_AVX512F)
		set(CPU_SUPPORTS_AVX512 TRUE)
		message(STATUS "CPU supports AVX512F")
	endif()
	if(HAS_VPOPCNTDQ)
		set(CPU_SUPPORTS_AVX512_VPOPCNTDQ TRUE)
		message(STATUS "CPU supports AVX512VPOPCNTDQ")
	endif()
else()
	message(STATUS "CPU feature detection failed")
endif()

if(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512_VPOPCNTDQ)
	message(STATUS "Enabling AVX512 + VPOPCNTDQ.")
	add_definitions(-DUSE_AVX512)
	add_definitions(-DUSE_AVX512_VPOPCNTDQ)
elseif(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU support -mavx512f. Enabling AVX512.")
	add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
	message(STATUS "Compiler and CPU support -mavx2. Enabling AVX2.")
	add_definitions(-DUSE_AVX2)
else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
endif()

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

	
find_package(BLAS)
if(BLAS_FOUND)
	message(STATUS "BLAS found ${BLAS_LIBRARIES}")
    if("${BLAS_LIBRARIES}" MATCHES "mkl")
		message(STATUS "BLAS is MKL, adding mkl_rt")
		set(BLAS_LIBRARIES mkl_rt)
    endif()
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)

if(BUILD_MILVUS_SERVER)
	add_subdirectory(src/server)
endif()
