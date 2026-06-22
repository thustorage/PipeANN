---
hide:
  - navigation
---

# Repository Layout

## Project Structure

```bash
PipeANN/
├── src/                          # Core implementation
│   ├── index.cpp                    # In-memory Vamana index
│   ├── ssd_index.cpp                # On-disk index (load/save, graph mgmt, entry points)
│   ├── search/                   # Search algorithms
│   │   ├── pipe_search.cpp          # 🌟 PipeANN pipelined search
│   │   ├── pipe_search_common.h     # Shared helpers for pipelined search
│   │   ├── spec_filter_search.cpp   # 🌟 Speculative filtered search
│   │   ├── beam_search.cpp          # DiskANN best-first search
│   │   ├── page_search.cpp          # Starling page-based search
│   │   └── coro_search.cpp          # Coroutine-based multi-query search
│   ├── update/                   # Update operations
│   │   ├── direct_insert.cpp        # 🌟 OdinANN direct insert
│   │   └── delete_merge.cpp         # Delete and merge logic
│   ├── python/                   # Python binding sources
│   │   ├── pybind.cpp               # pybind11 module definitions
│   │   └── pyindex.cpp              # IndexPipeANN C++ wrapper impl
│   └── utils/                    # Utilities
│       ├── distance.cpp             # Distance computation (L2/IP/cosine)
│       ├── linux_aligned_file_reader.cpp  # io_uring/AIO support
│       ├── index_build_utils.cpp    # Shared build helpers
│       ├── kmeans_utils.cpp         # K-means for PQ codebooks
│       ├── partition.cpp            # Dataset partitioning (PiPNN)
│       └── pipnn.cpp                # PiPNN graph construction
├── include/                      # Header files
│   ├── index.h                      # In-memory index interface
│   ├── ssd_index.h                  # On-disk index interface
│   ├── ssd_index_defs.h             # On-disk index constants & layout macros
│   ├── dynamic_index.h              # Dynamic index wrapper (search + update)
│   ├── pyindex.h                    # C++ wrapper exposed to Python
│   ├── nbr/                      # Neighbor storage & quantization
│   │   ├── abstract_nbr.h             # Base interface for neighbor codecs
│   │   ├── pq_nbr.h / pq_table.h      # Product-quantized neighbors
│   │   ├── rabitq_nbr.h / rabitq/     # RaBitQ 1-bit / multi-bit quantization
│   │   └── dummy_nbr.h                # No-op codec (debug)
│   ├── filter/                   # Speculative filtering
│   │   ├── attribute.h                # AttrIndex (label inverted, range, string-eq) + spmat decode helper
│   │   ├── selector.h                 # Selector tree (LabelOr/And, Range, StringEq, And/Or/Not)
│   │   ├── dsl_compiler.h             # SQL-like filter DSL → CompiledFilter; $$var binders
│   │   └── filter_utils.h             # Shared filter helpers
│   └── utils/                    # Headers for utilities, containers, logging
│       ├── pipnn.h, partition.h       # PiPNN build
│       ├── page_cache.h, journal.h    # SSD page cache & update journal
│       ├── lock_table.h, concurrent_queue.h  # Concurrency primitives
│       └── ...
├── tests/                        # Test programs & benchmarks
│   ├── build_disk_index.cpp         # Build on-disk index
│   ├── build_disk_index_filtered.cpp # Build on-disk index with attributes
│   ├── build_memory_index.cpp       # Build in-memory index
│   ├── search_disk_index.cpp        # Search benchmark (SSD)
│   ├── search_disk_index_mem.cpp    # Search benchmark (Load SSD index to RAM)
│   ├── search_disk_index_filtered.cpp # Filtered search benchmark
│   ├── test_insert_search.cpp       # Insert-search benchmark
│   ├── overall_performance.cpp      # Insert-delete-search benchmark (SSD)
│   ├── overall_perf_mem.cpp         # Insert-delete-search benchmark (in-memory)
│   ├── pad_partition.cpp            # Pad partition file (for Starling)
│   ├── normalize_data.cpp           # Normalize vectors (for cosine/MIPS)
│   ├── test_cpu.cpp                 # CPU/SIMD sanity check
│   ├── test_field_codec.cpp         # Scalar field codec unit test (server field_codec.h)
│   └── utils/                    # Data utilities (vecs_to_bin, gt_update, ...)
├── proto/                        # gRPC protocol definitions
│   └── milvus/                      # Milvus wire-protocol .proto files
├── src/server/                   # Milvus-compatible engine + C++ gRPC server
│   ├── main.cpp                     # Server entry point (CLI args, gRPC bootstrap)
│   ├── milvus_server.cpp            # Servicer: Milvus protobuf <-> CollectionStore
│   ├── milvus_server.h              # Servicer declarations
│   ├── collection_store.cpp         # CollectionStore engine (schema, build, insert/search/query)
│   ├── collection_store.h           # CollectionStore + Collection declarations
│   ├── field_codec.h                # Scalar field encode/decode (incl. order-preserving float)
│   ├── doc_store.h                  # RocksDB-backed (id, tag, document) store
│   └── search_worker_pool.h         # Per-thread search worker pool
├── tests_py/                     # Python examples & tests
│   ├── collection_example.py        # MilvusClient (in-process) smoke test
│   ├── index_example.py             # IndexPipeANN smoke test
│   ├── test_filter.py               # Python Selector pytest
│   ├── test_native_selector.py      # Native selector composition pytest
│   ├── test_hybrid_query.py         # Hybrid (vector + scalar) MilvusClient pytest
│   ├── test_milvus_quickstart.py    # Milvus-compatible API quickstart test
│   ├── test_grpc_server_e2e.py      # gRPC server end-to-end pytest (pymilvus client)
│   ├── test_insert_attrs.py         # Filtered build-vs-insert regression check
│   ├── test_insert_search.py        # Insert-then-search regression check
│   └── test_range_search.py         # Range-search smoke test
├── pipeann/                      # Python package
│   ├── __init__.py                  # Re-export public Python API
│   ├── filter.py                    # Attributes, AttrsVec, Selector
│   ├── index.py                     # IndexPipeANN, Metric, VALID_DATA_TYPES
│   └── milvus.py                    # Milvus-compatible MilvusClient (in-process, native CollectionStore)
├── scripts/                      # Evaluation scripts
└── third_party/                  # Dependencies (liburing, spdk)
```

## Evaluation Scripts

The `scripts/` directory reproduces the figures in our papers. Before running
anything, point the scripts at your data — they assume the dataset and index
layout below. **Edit the hard-coded paths in `eval_f.sh` / `fig*.sh`** or create
symlinks if your environment differs.

### Expected Dataset & Index Layout

```bash
/mnt/nvme/data/                  # Dataset Directory
├── bigann/
│   ├── 100M.bbin                # SIFT100M dataset
│   ├── 100M_gt.bin              # SIFT100M ground truth
│   ├── truth.bin                # SIFT1B ground truth
│   ├── bigann_200M.bbin         # SIFT200M (for updates)
│   └── bigann_query.bbin        # SIFT query
├── deep/
│   ├── 100M.fbin                # DEEP100M dataset
│   ├── 100M_gt.bin              # DEEP100M ground truth
│   └── queries.fbin             # DEEP query
└── SPACEV1B/
    ├── 100M.bin                 # SPACEV100M dataset
    ├── 100M_gt.bin              # SPACEV100M ground truth
    ├── query.bin                # SPACEV query
    └── truth.bin                # SPACEV1B ground truth

/mnt/nvme2/indices/              # Search-Only Indexes
├── bigann/100m                  # SIFT100M index prefix
├── deep/100M                    # DEEP100M index prefix
└── spacev/100M                  # SPACEV100M index prefix

/mnt/nvme/indices_upd/           # Search-Update Indexes
├── bigann/100M                  # SIFT100M index for updates
├── bigann_gnd_insert/           # GT for insert-search workload
└── bigann_gnd/                  # GT for insert-delete-search workload
```

### Script Layout

```
scripts/
├── tests-pipeann/                # PipeANN (OSDI'25) evaluation
│   ├── hello_world.sh               # Quick functionality test
│   ├── fig11.sh ~ fig18.sh          # Paper figure reproduction
│   ├── plotting.py                  # Generate figures
│   └── plotting.ipynb               # Jupyter notebook for plotting
├── tests-odinann/                # OdinANN (FAST'26) evaluation
│   ├── hello_world.sh               # Quick functionality test
│   ├── fig6.sh ~ fig12.sh           # Paper figure reproduction
│   └── plotting.ipynb               # Jupyter notebook for plotting
├── run_all_pipeann.sh               # Run all PipeANN experiments
├── bench_milvus_vs_pipeann.py       # SIFT1M benchmark: PipeANN gRPC vs Milvus (same MilvusClient)
└── validate_index_structure.py      # Index validation tool
```

### Running the Scripts

**Hello World (verify installation):**
```bash
# PipeANN search-only test (~1 min)
bash scripts/tests-pipeann/hello_world.sh

# OdinANN update test (~1 min)
bash scripts/tests-odinann/hello_world.sh
```

**Individual experiments:**

- **PipeANN (Search-Only)**:
    - `fig11.sh`: Latency vs Recall (100M datasets)
    - `fig12.sh`: Throughput vs Recall (100M datasets)
    - `fig13.sh`: Latency breakdown
    - `fig14.sh` ~ `fig18.sh`: Ablation, scalability, etc.
- **OdinANN (Search-Update)**:
    - `fig6.sh`: Insert-search on SIFT100M (~4d)
    - `fig7.sh`: Insert-search on DEEP100M (~4d)
    - `fig8.sh`: Insert-search on SIFT1B (~8d)
    - `fig12.sh`: Insert-delete-search (~6d)

**Plot results:**
```bash
cd scripts/tests-pipeann && python plotting.py
# Or open plotting.ipynb in Jupyter.
```
