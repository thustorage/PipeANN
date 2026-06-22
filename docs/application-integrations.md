---
hide:
  - navigation
---

# Application Integrations

PipeANN speaks the **Milvus API**, so an existing Milvus app can switch to PipeANN with almost no code changes. There are two ways to run it:

- **In-process `MilvusClient`** — a drop-in replacement for `pymilvus.MilvusClient` that runs inside your Python process, with no network in between. The URI points at a directory, not a `.db` file.
- **gRPC server** — a standalone C++ server that speaks the Milvus wire protocol. Any Milvus SDK (Python, Go, Java) can connect to it.

Both are thin wrappers over the same C++ engine (`CollectionStore`), which handles the schema, scalar encoding, filter compilation, output fields, and the SSD-backed graph index. There is no separate PipeANN client class — you use `MilvusClient`, or any stock Milvus SDK against the server.

---

## 1. Interface

### 1.1 In-process MilvusClient

```python
import numpy as np

USE_PIPEANN = True

if USE_PIPEANN:
    from pipeann import MilvusClient
    client = MilvusClient(uri="./pipeann-data")   # directory
else:
    from pymilvus import MilvusClient
    client = MilvusClient(uri="./milvus-lite.db") # .db file

client.create_collection("demo", dimension=128, metric_type="L2")

data = [
    {"id": "v1", "vector": np.random.rand(128).tolist(), "color": "red", "price": 42},
    {"id": "v2", "vector": np.random.rand(128).tolist(), "color": "blue", "price": 99},
]
client.insert("demo", data)

# Vector search
results = client.search(
    "demo",
    data=[np.random.rand(128).tolist()],
    limit=5,
    search_params={"params": {"L": 64}},
    output_fields=["color", "price"],
)

# Vector search + scalar filter
results = client.search(
    "demo",
    data=[np.random.rand(128).tolist()],
    filter="price < 80",
    limit=5,
)

# Scalar query
rows = client.query("demo", filter="color == 'red'", output_fields=["color", "price"])

# Delete & persist
client.delete("demo", ids=["v1"])
client.flush("demo")
```

**Index construction:** `insert` buffers the vectors. The SSD-backed graph index
is built from that buffer on the first read (`search`/`query`/`count`), or when you
call `create_index`/`load_collection`. So a collection is searchable right after
`insert` with no extra call — the build just happens on first use. Inserts *after*
the index is built are added to the live graph directly.

#### Supported operations

| Operation | Methods |
|-----------|---------|
| Collection CRUD | `create_collection`, `drop_collection`, `list_collections`, `has_collection`, `describe_collection`, `count` |
| Write | `insert`, `upsert` |
| Vector search (optional filter) | `search` |
| Scalar query / get by ID | `query`, `get` |
| Delete | `delete` |
| Persist | `flush` |

#### Schema declaration

You can pass a Milvus-style schema dict to declare typed scalar fields for hybrid filtering:

```python
schema = {
    "auto_id": False,
    "fields": [
        {"name": "id", "dtype": "VARCHAR", "is_primary": True, "params": {"max_length": 64}},
        {"name": "vector", "dtype": "FLOAT_VECTOR", "params": {"dim": 128}},
        {"name": "category", "dtype": "VARCHAR", "params": {"max_length": 128}},
        {"name": "price", "dtype": "FLOAT"},
    ],
}
client.create_collection("products", schema=schema, metric_type="L2")
```

Scalar fields map automatically to PipeANN filter index types: `VARCHAR` →
string, `INT*` → range, `FLOAT`/`DOUBLE` → range (order-preserving IEEE-754
encoding), `BOOL` → range.

#### Filter expressions

`search(filter=...)` and `query(filter=...)` take a standard Milvus boolean
expression string. The engine compiles it natively — no per-row Python scan — and
you can only reference declared scalar fields. Supported syntax:

| Category | Operators / forms |
|----------|-------------------|
| Comparison | `==`, `!=` (`<>`), `>`, `>=`, `<`, `<=` |
| Membership | `field in [a, b, c]`, `field not in [...]` |
| Range | `field >= lo and field <= hi`, `field between lo and hi` |
| String match | `field like "foo%"` (prefix), `like "%bar"` (suffix), `like "%mid%"` (contains) |
| Array / label | `array_contains(tags, X)`, `array_contains_all(tags, [X, Y])`, `array_contains_any(tags, [X, Y])` |
| Boolean | `and`, `or`, `not`, parentheses |

```python
client.search("demo", data=[q], filter="price < 80 and color == 'red'", limit=5)
client.query("demo", filter="array_contains_all(tags, [7, 42])", output_fields=["tags"])
```

Filtering is fused into the graph traversal by PipeANN's speculative filtering, so
a selective filter does not turn into a slow post-scan.

### 1.2 gRPC Server

The PipeANN gRPC server is a C++ server that speaks the Milvus wire protocol. You
need no PipeANN-specific client — start the server and keep using
`pymilvus.MilvusClient`. Switching is just a different URI.

#### Build

The server target needs gRPC, Protobuf, and RocksDB on top of PipeANN's core
build dependencies:
```bash
# Ubuntu >= 22.04
sudo apt install libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev \
                 protobuf-compiler librocksdb-dev

cd build_server && cmake -DBUILD_MILVUS_SERVER=ON .. && make -j pipeann_milvus_server
```

#### Start

If PipeANN is installed (`pip install .`), use the bundled launcher, which finds
the `pipeann_milvus_server` binary automatically:

```bash
pipeann-server --data_dir ./data --port 19530 --threads 8
# equivalently:
python3 -m pipeann.server --data_dir ./data --port 19530 --threads 8
```

The launcher looks for the binary in `$PIPEANN_SERVER_BIN`, the installed package,
the in-repo `build/` tree, and finally `$PATH`. From a source checkout you can
also run the binary directly:

```bash
./build_server/src/server/pipeann_milvus_server --data_dir ./data --port 19530 --threads 8
```

#### Connect

```python
from pymilvus import MilvusClient

# Stock Milvus server (e.g. on its own host/port):
# client = MilvusClient(uri="http://localhost:19530")

# PipeANN server (started above on port 19530):
client = MilvusClient(uri="http://localhost:19530")

client.create_collection("demo", dimension=768)
client.insert("demo", [{"id": 1, "vector": [0.1] * 768, "text": "hello"}])
client.create_index("demo")        # build the SSD graph from buffered vectors
client.load_collection("demo")     # standard pymilvus flow (no-op if already built)
results = client.search("demo", data=[[0.1] * 768], limit=5)
```

**Index construction:** in server mode, `insert` writes vectors into an in-memory
buffer, and the SSD-backed graph index is built only on `create_index`. This is
the standard Milvus `create_index` → `load_collection` → `search` flow, so the
stock pymilvus client runs unchanged.

#### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `./data` | Collection storage directory |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `19530` | gRPC listen port |
| `--threads` | `0` (auto) | Number of server threads |

#### Supported gRPC operations

`CreateCollection`, `DropCollection`, `HasCollection`, `ShowCollections`,
`DescribeCollection`, `GetCollectionStatistics`, `Insert`, `Upsert`, `Delete`,
`Search`, `Query`, `Flush`, the index lifecycle calls (`CreateIndex`, `DropIndex`,
`DescribeIndex`, `GetIndexState`, `GetIndexBuildProgress`), the load lifecycle
calls (`LoadCollection`, `ReleaseCollection`, `GetLoadState`, `GetLoadingProgress`),
plus `Connect` / `GetVersion` for the SDK handshake.

---

## 2. PipeANN MilvusClient vs. Milvus Lite

Both are in-process local engines. They run the same benchmark code; only the URI
differs (a directory vs a `.db` file).

Benchmark: `tests_py/bench_milvus_lite_vs_pipeann_client.py`

### 2.1 SIFT1M — Unfiltered Vector Search

Dataset: BigANN 1M (1,000,000 × 128-dim uint8 vectors, L2, top-10), 10,000
queries. Throughput uses 64 concurrent single-query threads; latency is
sequential single-query search (3 rounds × 1,000 queries).

> **In-process throughput here is limited by the Python GIL, not the engine.**
> Both backends are driven from one Python interpreter via `ThreadPoolExecutor`,
> because the in-process engine keeps the index in memory and is not fork-safe.
> PipeANN's native `search()` releases the GIL, but the Python work around it
> (building the query array, turning the C++ result back into dicts) holds it. A
> scaling probe shows in-process QPS flattens at ~2.7× the single-thread rate after
> 4 threads and stays flat to 64 — the classic single-interpreter GIL ceiling. So
> the in-process QPS column below measures Python overhead, **not** the engine; it
> understates PipeANN's real throughput by about 8×. For true engine throughput see
> §3 (gRPC, 64 separate processes → ~12,000 QPS). The comparison is still fair as a
> *relative* number — both backends pay the same Python tax on the same code path —
> and the latency column (measured single-threaded, no GIL contention) is a clean
> engine-to-engine number.

> **Why Milvus Lite is one point, not a sweep.** Milvus Lite is the embedded
> single-file build. In local mode it accepts only `FLAT`, `IVF_FLAT`, and
> `AUTOINDEX` (it rejects `HNSW`), and its embedded segcore **ignores the
> search-time tuning parameter entirely**: recall and QPS are flat across `ef` from
> 8 to 512 (`AUTOINDEX`) and across `nprobe` from 1 to `nlist` (`IVF_FLAT`). Even
> `nprobe == nlist`, which should be exhaustive search, returns the same fixed
> result set instead of the exact FLAT answer. Build-time `nlist` moves nothing
> either. So Lite has exactly one operating point you can't tune from the client,
> and sweeping it just repeats that point. PipeANN, by contrast, sweeps `ef`/`L`
> and trades recall for throughput as expected.

| ef/L | Backend | Recall@10 | 64-thread QPS (GIL-bound) | p50 (ms) | p99 (ms) |
|------|---------|-----------|---------------------------|----------|----------|
| 32 | PipeANN MilvusClient | 0.9708 | 1,311 | 1.86 | 2.15 |
| 64 | PipeANN MilvusClient | 0.9934 | 1,322 | 2.32 | 2.74 |
| 128 | PipeANN MilvusClient | 0.9987 | 1,391 | 3.28 | 4.06 |
| 256 | PipeANN MilvusClient | 0.9998 | 1,358 | 5.13 | 6.07 |
| — (fixed) | Milvus Lite (AUTOINDEX) | 0.8973 | ~240 | 7.7–7.9 | 8.8–8.9 |

Milvus Lite's row is its single operating point; the small QPS spread (235–245
across runs) is measurement noise, not parameter sensitivity.

**Key observations:**

- The 64-thread QPS column is GIL-bound for both backends (see note above), so read
  it only as a relative number, not as engine throughput. On that basis PipeANN is
  **~5–6×** Milvus Lite (1,311 vs ~240 QPS at ef=32).
- PipeANN p50 latency is **1.86 ms** (ef=32) vs ~7.7 ms for Milvus Lite — about a
  **4× gap** on a clean single-threaded measurement.
- PipeANN trades recall for throughput across its `ef`/`L` sweep (0.971 → 0.9998);
  Milvus Lite sits at a fixed 0.897 that no client parameter can move.
- For PipeANN's real concurrent throughput, see §3.1: the same engine behind a
  gRPC server with 64 separate client processes does ~12,000 QPS — ~9× the
  GIL-bound in-process figure here.

---

## 3. PipeANN gRPC Server vs. Milvus Server

Both are reached through `pymilvus.MilvusClient` on the same client code path; only
the server URI differs.

Benchmark: `tests_py/bench_milvus_vs_pipeann.py`

### 3.1 SIFT1M — Unfiltered Vector Search

Same dataset as above. Throughput uses 64 concurrent processes (one independent
gRPC connection each), so unlike §2 there is no shared-interpreter GIL ceiling —
this is real engine throughput. Latency is sequential single-query search.

- **PipeANN gRPC** — `pipeann_milvus_server`, 32 search workers.
- **Milvus 2.5.0** — standalone container (embedded etcd), HNSW (`M=16`, `efConstruction=200`).

| ef/L | Backend | Recall@10 | 64-client QPS | p50 (ms) | p99 (ms) |
|------|---------|-----------|---------------|----------|----------|
| 32 | PipeANN gRPC | 0.9719 | 12,088 | 2.95 | 3.49 |
| 32 | Milvus 2.5.0 | 0.9114 | 23,287 | 4.51 | 5.47 |
| 64 | PipeANN gRPC | 0.9936 | 9,617 | 3.40 | 4.03 |
| 64 | Milvus 2.5.0 | 0.9698 | 19,327 | 5.04 | 5.73 |
| 128 | PipeANN gRPC | 0.9987 | 6,452 | 3.89 | 4.63 |
| 128 | Milvus 2.5.0 | 0.9921 | 16,495 | 5.55 | 6.40 |
| 256 | PipeANN gRPC | 0.9998 | 3,690 | 5.50 | 6.93 |
| 256 | Milvus 2.5.0 | 0.9985 | 12,145 | 6.58 | 7.91 |

**Key observations:**

- **Milvus has higher raw QPS on unfiltered SIFT1M** (23,287 vs 12,088 at ef=32).
  That's expected: Milvus serves this from an **in-memory** HNSW graph, while
  PipeANN is a **disk-resident** index that pages graph and vector data from SSD
  per query. Trading peak throughput for a much smaller memory footprint is the
  whole point of an on-disk design — PipeANN holds 1M vectors in tens of MB of RAM
  where Milvus holds the full graph in memory.
- **PipeANN's per-query latency is lower at matched-or-better recall.** At the same
  ef, PipeANN p50 is below Milvus at every point (2.95 vs 4.51 ms at ef=32). At the
  same recall the gap holds: PipeANN ef=32 (recall 0.972, p50 2.95 ms) beats Milvus
  ef=64 (recall 0.970, p50 5.04 ms). The SSD reads sit on a deep io_uring pipeline,
  so they don't inflate single-query latency.
- **PipeANN reaches higher recall per `ef`.** At ef=32 PipeANN is at 0.972 where
  Milvus HNSW is at 0.911; Milvus needs ef≈64 to match.
- Ingest: PipeANN gRPC 63K vec/s vs Milvus 33K vec/s.

In short: if the dataset fits comfortably in RAM and peak QPS is all you want,
in-memory Milvus wins on throughput. PipeANN is for serving **larger-than-RAM**
indexes at **low, predictable latency** and **high recall** — and §3.2 shows that
becomes a decisive advantage once you add a filter.

### 3.2 YFCC10M — Filtered Vector Search

Dataset: YFCC10M filtered ANNS benchmark (10,000,000 × 192-dim uint8 vectors, L2,
top-10). Filter: `array_contains_all(tags, [X, Y])` — an AND over two label
fields. Each base vector carries ~11 labels on average. Ground truth is computed
on the full 10M set, so recall is directly comparable.

Throughput uses 64 concurrent processes. Both backends ran the same 10,000
queries at every `ef`. A concurrent warmup pass runs first so the first `ef` is
not penalized by a cold page cache.

| ef/L | Backend | Recall@10 | 64-client QPS | p50 (ms) | p99 (ms) |
|------|---------|-----------|---------------|----------|----------|
| 32 | PipeANN gRPC | 0.9443 | 4,893 | 7.78 | 33.49 |
| 64 | PipeANN gRPC | 0.9809 | 3,640 | 8.07 | 40.84 |
| 128 | PipeANN gRPC | 0.9880 | 2,807 | 8.23 | 53.61 |
| 256 | PipeANN gRPC | 0.9900 | 1,744 | 9.18 | 83.08 |
| 32 | Milvus 2.5.0 | 0.9887 | 38 | 380.2 | 805.2 |
| 64 | Milvus 2.5.0 | 0.9900 | 38 | 379.7 | 830.2 |
| 128 | Milvus 2.5.0 | 0.9905 | 39 | 360.7 | 685.2 |
| 256 | Milvus 2.5.0 | 0.9908 | 38 | 380.7 | 866.0 |

**Key observations:**

- PipeANN filtered search delivers **~130× the throughput** of Milvus (4,893 vs
  38 QPS at ef=32) and holds that gap across the whole sweep.
- PipeANN p50 is **7.8–9.2 ms** vs **~380 ms** for Milvus — a **~40–50× latency gap**.
- **Milvus's filtered throughput is flat (~38 QPS) at every `ef`.** Its cost is
  dominated by scanning the `array_contains_all` predicate over the 10M set, not by
  the vector search, so the `ef` knob barely moves it. The in-memory-HNSW advantage
  from §3.1 is fully erased once a low-selectivity filter is in play.
- PipeANN uses speculative filtering — vector and filter are fused in the C++ graph
  traversal, so the filter prunes the traversal instead of running as a separate
  scan. That's why the on-disk engine that *lost* on raw unfiltered QPS (§3.1)
  *wins by ~130×* here.
- Milvus reaches slightly higher recall (~0.99 vs PipeANN's 0.944 at ef=32) because
  it applies the filter exactly. PipeANN closes to 0.99 by ef=256 while still
  serving ~1,744 QPS — ~46× Milvus.
- Ingest (full 10M, with label indexing): PipeANN 26,824 vec/s (372.8 s); Milvus 18,522 vec/s (539.9 s).

### 3.3 Where the latency comes from

The gRPC envelope (pymilvus encode + localhost TCP + decode) is a flat **~1.5 ms**
and does not amplify the tail. Breaking down the server-side cost of an unfiltered
SIFT query under load (ef=32, 32 workers, 64 clients):

- **native graph+SSD search (`exec`)** — p50 ~2.1 ms, the dominant term;
- **worker-pool queue wait** — p50 ~0.5 ms (requests briefly queue when 64 clients
  outnumber 32 search workers; this is normal back-pressure, not a lock problem — it
  grows under heavier filtered load, but the pool is what keeps the io_uring ring
  count bounded);
- **doc-store id resolution** — p50 ~0.1 ms.

P99 comes from the native `DynamicIndex::search` SSD I/O tail: low-selectivity
filters force more candidate vector reads, and the per-query variance in I/O count
drives the tail (visible as the rising p99 across the YFCC `ef` sweep).

### 3.4 Memory footprint

This is where the on-disk design pays off most. Both servers were loaded with the
**same two collections at once** — SIFT1M (1M × 128d) + YFCC10M (10M × 192d), 11M
vectors total — and peak resident memory was sampled while serving 64 concurrent
clients.

| Server | Peak RSS (both collections resident, 64 clients) |
|--------|--------------------------------------------------|
| **PipeANN gRPC** | **1.69 GB** (idle 1.29 GB) |
| **Milvus** | **12.4 GB** (steady 11.2 GB) |

PipeANN serves the same data in **~6.6× less memory**. Only the PQ-compressed
vectors (~512 MB for both collections), tags, filter bitmaps, and per-thread I/O
scratch are resident; Milvus holds the entire HNSW graph plus raw vectors in RAM.
Under load PipeANN grows only ~0.4 GB over its idle baseline.

> The C++ server links **tcmalloc** (`USE_TCMALLOC=ON`, default), which returns
> freed per-query scratch to the OS. With glibc's default allocator the same
> 64-client run held on to ~2.4 GB of extra per-thread malloc arenas, pushing peak
> RSS to ~3.7 GB — bounded and reused, but avoidable.

**The overall picture (§3.1–3.4):** PipeANN's per-query **latency is consistently
lower** at matched recall, its **throughput is competitive** (it trails in-memory
Milvus on raw unfiltered QPS but wins by ~130× once you add a filter), and its
**memory footprint is dramatically smaller**. For larger-than-RAM, filtered
workloads — the common production case — that combination is decisive.

---

> **Environment:** single node, 112-core x86-64, 512 GB RAM. PipeANN gRPC server:
> 32 search workers. Milvus v2.5.0: standalone container with embedded etcd, local
> storage. Both on the same host. Benchmarks driven by
> `tests_py/bench_milvus_vs_pipeann.py` (gRPC, §3) and
> `tests_py/bench_milvus_lite_vs_pipeann_client.py` (in-process, §2).
