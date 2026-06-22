# Changelog

## v0.4.0 (unreleased)

### Milvus-compatible collection engine

The Python document/collection layer was replaced by a native C++
`CollectionStore` (`src/server/collection_store.{h,cpp}`). It owns schema
metadata, scalar encoding, filter compilation, the SSD-backed graph index, and a
RocksDB-backed `(id, tag, document)` doc store. The same engine backs both
Milvus-compatible entry points:

- **In-process `MilvusClient`** (`pipeann/milvus.py`) is a thin row/column shim
  over `CollectionStore`; its URI is a local directory.
- **C++ gRPC server** (`src/server/milvus_server.cpp`) speaks the Milvus wire
  protocol, so stock Milvus SDKs can connect without PipeANN-specific client
  code.

#### New

- `CollectionStore` C++ engine and its `pipeann.C.CollectionStore` pybind
  binding (`src/python/pycollection.cpp`).
- Milvus-style collection CRUD, insert/upsert, search, scalar query, get,
  delete, count, flush, index creation, and load/release stubs through
  `MilvusClient` and the gRPC server.
- Native SQL-like filter strings for `search(filter=...)`, `query(filter=...)`,
  `delete(filter=...)`, and `count(filter=...)`. Supported syntax includes
  comparisons, `in` / `not in`, `between`, `like`, `array_contains`,
  `array_contains_all`, `array_contains_any`, `and`, `or`, `not`, and
  parentheses. Mongo-style filter dicts are not part of the Milvus-compatible
  API.
- Scalar field encoding for `VARCHAR`, integer, boolean, float/double, and
  integer-array fields. Float/double filter literals are rewritten to the
  order-preserving integer encoding used by range indexes.
- `StringPrefixSelector`, `StringSuffixSelector`, and `StringLikeSelector` for
  `LIKE` filtering, alongside `StringEqSelector`.
- `field_codec.h` scalar encode/decode coverage in `tests/test_field_codec.cpp`,
  Milvus quickstart coverage in `tests_py/test_milvus_quickstart.py`, and gRPC
  end-to-end coverage in `tests_py/test_grpc_server_e2e.py`.
- `pipeann-server` / `python -m pipeann.server` launcher for the bundled
  `pipeann_milvus_server` binary.

#### Low-level filter config

- `pipeann::dsl::compile(sql, schema)` now accepts the same SQL-like expression
  string used by the Milvus layer. Internally it is lowered to selector nodes;
  callers should not pass Mongo-style JSON.
- `pipeann::dsl::CompiledFilter` returns the selector, literal attribute
  template, `slot_map`, and `var_field_type`. Literal filters are ready to
  search immediately.
- `$$var` placeholders remain available for C++ benchmark and low-level
  `IndexPipeANN` workflows. `bind_row` and `bind_batch_per_var` fill those slots,
  including `.spmat` batch binding.
- `DynamicIndex::load_filter_from_json(config_path)` and
  `IndexPipeANN.load_filter_from_json(config_path)` consume
  `{attr_indexes, filter, bindings}` where `filter` is a SQL-like expression
  string, then return one `Attributes` row per bound query.

#### Fixed

- Benchmark throughput in `tests_py/bench_milvus_vs_pipeann.py` is now measured
  from each worker's barrier-synced search window instead of including process
  startup, pymilvus import, gRPC connection, and warmup time.
- Benchmark cold-cache artifacts were reduced: unfiltered sweeps discard round 0
  when `rounds > 1`, and filtered sweeps run a concurrent warmup pass before the
  `ef` sweep.

#### Breaking

- Removed the PipeANN-specific Python `Client` (`pipeann/client.py`) and
  `Collection` (`pipeann/collection.py`). Use `MilvusClient` in-process or any
  Milvus SDK against the gRPC server.
- Removed the SQLite-backed Python document/metadata layer. Collection metadata
  and documents now live in the native engine and RocksDB doc store.
- Removed the LangChain (`PipeANNVectorStore`) and Qdrant FastAPI integrations
  that depended on the deleted `Collection` layer.
- `Collection.compile_filter`, `Collection.hybrid_query`, and
  `Collection.filter_query` no longer exist. Use `MilvusClient.search` /
  `MilvusClient.query` with a SQL-like `filter=` string, or low-level
  `IndexPipeANN` selectors for direct index workflows.
- Removed `parse_selector_from_json` and `load_selector_from_config`
  (`include/filter/selector.h`) and `pipeann::load_base_attr_from_config`
  (`include/filter/attribute.h`). Use `load_filter_from_json` with the
  `{attr_indexes, filter, bindings}` config.
- The legacy `{"base": ..., "query": {...}}` filtered-search config no longer
  loads; `scripts/attr_config/*.json`, `tests/search_disk_index_filtered.cpp`,
  and `tests/utils/compute_groundtruth.cpp` use the new config schema.

## v0.3.0 (2026-05-18)

It adds new search capabilities first, then the Python-facing stack built on top of them, plus the refactors and tests needed to support updates and filtering consistently.

### Feature highlights

#### Speculative filtering

- Added speculative filtered ANNS for arbitrary attribute constraints.
- Uses lightweight in-memory probabilistic filters to explore a superset of valid vectors, then verifies final candidates exactly against SSD-resident attributes.
- A cost model chooses speculative pre-filtering, speculative in-filtering, or post-filtering per query.
- Supports label filters, range filters `[l, r)`, and Boolean combinations through native selectors.
- Added typed attributes (`Attributes`, `AttrsVec`), on-disk attribute indexes, dense-neighbor support (`range_dense`), and JSON filter config loading.

#### OOD refinement

- Added NGFix-style out-of-distribution graph refinement.
- Exposed `train_query_path`, `R_ood`, and `L_ood` in C++ build tools and `IndexPipeANN.build()`.
- Persisted OOD metadata in SSD index metadata.

#### Range search

- Added finite-threshold range search in C++ and Python.
- Results outside the threshold are filtered out and padded with `UINT32_MAX` / `inf`.
- Reuses the common pipelined traversal and result-copy path.

#### SPDK backend

- Added an optional SPDK I/O backend through `-DIO_ENGINE=spdk` for raw NVMe vector reads.
- Supports RAID-0-style striping across PCIe NVMe devices listed in `spdk_bdevs.json`, with one poller thread per device.
- Copies `{index_prefix}_disk.index` to the SPDK target on first open and reuses a marker to skip repeated copies.
- Keeps filtered-search attribute reads on `io_uring` while vector I/O uses SPDK.

#### Python API and integrations

- Added the current `IndexPipeANN(data_dim, data_type, metric)` API for build, load, search, insert, delete, save, filters, range search, and attribute-aware inserts.
- Added `Collection` and `Client` for SQLite-backed documents/payloads, vector CRUD, persistence, and collection auto-discovery.
- Added LangChain integration through `pipeann.langchain.PipeANNVectorStore`.
- Added a Qdrant-compatible FastAPI server with collection management, point upsert/query/scroll, payload indexes, filter delete, count, and save.
- Added `schema.json` persistence for collection config and attribute-index metadata.

### Code refactoring and implementation changes

- Merged dynamic search, insert, delete, merge, and save behavior into header-only `DynamicIndex<T>`.
- Unified update save/merge with same-prefix double-version replacement.
- Merged PiPNN and Vamana-style build paths behind `build_disk_index` and the shared SSD file format.
- Refactored pipelined top-k, range, and filtered search around `pipe_search_common.h` and `spec_filter_search.cpp`.
- Simplified metric/distance handling and updated PiPNN file layout.
- Added `IO_ENGINE` selection for `uring`, `aio`, and `spdk`, separate dense-node I/O sizing, CMake/CI cleanup, and package version `0.3.0`.

### Tests and examples

- Switched the project license from MIT to Apache License 2.0 and updated NOTICE attribution, including NGFix graph-refinement logic under MIT.
- Added Python insert/delete search tests, including assertions that results include inserted data and exclude deleted data.
- Added filtered insert regression tests comparing full build vs. build-plus-insert with attributes.
- Added native selector, range-search, collection, LangChain, and Qdrant server tests/examples.
- Added C++ filtered build/search and update test coverage.

### Breaking changes

- `DynamicSSDIndex` was removed; use `DynamicIndex<T>` in C++ or `IndexPipeANN` in Python.
- `dynamic_index.cpp` was removed; `DynamicIndex<T>` is now header-only.
- `filter/label.h` was removed; use `filter/attribute.h` and `filter/selector.h`.
- `build_pipnn_index` was removed; use `build_disk_index` with PiPNN-compatible parameters.
- The old schema-centered Python API was removed; construct `IndexPipeANN(data_dim, data_type, metric)` directly and let `Collection` / `Client` own collection persistence.
- `PyIndexInterface` construction changed from a Python dict to explicit `(dim, dtype, metric)` arguments.
- `IndexPipeANN.search()` now takes `selector`, `query_attrs`, and finite `range` keyword arguments.
- `IndexPipeANN.build()` now takes `attrs`, `range_dense`, `train_query_path`, `R_ood`, and `L_ood`; major build knobs auto-configure when left at `0`.
- `pipnn.h` / `pipnn.cpp` moved under the utils layout.
