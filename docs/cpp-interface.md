# C++ Interface

The C++ interface is the fast path: it exposes the full feature set (search, updates, filtered search, OOD, in-memory workloads) and matches the numbers reported in the papers. For best raw performance, see also the [SPDK backend](cpp-interface-spdk.md).

This guide walks through:

1. [Build configuration](#build-configuration) — pick search-only or search+update at compile time.
2. [Prepare datasets](#prepare-datasets) — download, format conversion, ground truth.
3. [Build the index](#build-the-index) — Vamana or PiPNN, plus the optional in-memory entry point.
4. [Search](#search) — basic search and search modes.
5. [Update (insert / delete)](#update-insert-delete) — concurrent insert/search/delete workloads.
6. [In-memory workloads](#in-memory-workloads) — load the SSD index entirely into DRAM.
7. [Filtered search](#filtered-search) — attribute-constrained ANNS via speculative filtering.
8. [OOD search](#out-of-distribution-ood-search) — NGFix refinement for cross-modal workloads.

## Build Configuration

Two flags in `CMakeLists.txt` control the build profile:

| Flag | Effect |
|------|--------|
| `-DREAD_ONLY_TESTS` | Disables update paths; higher search throughput. |
| `-DNO_MAPPING` | Disables the tag↔ID mapping table; required together with `-DREAD_ONLY_TESTS` for search-only. |

* **Search-only** (best search performance): enable both flags.
* **Search+Update**: disable both flags.

Re-run `bash ./build.sh` after toggling.

## Prepare Datasets

**1. Download.** [SIFT](http://corpus-texmex.irisa.fr/), [DEEP1B](https://github.com/matsui528/deep1b_gt), [SPACEV](https://github.com/microsoft/SPTAG). If the originals are unavailable, [Big ANN benchmarks](https://big-ann-benchmarks.com/neurips21.html) mirrors them.

SPACEV1B may ship as several sub-files. Concatenate them and save the numpy array as `bin`:

```py
# bin format:
# | 4 bytes num_vecs | 4 bytes dim | flattened vectors |
def bin_write(vectors, filename):
    with open(filename, 'wb') as f:
        num_vecs, vector_dim = vectors.shape
        f.write(struct.pack('<i', num_vecs))
        f.write(struct.pack('<i', vector_dim))
        f.write(vectors.tobytes())

def bin_read(filename):
    with open(filename, 'rb') as f:
        num_vecs = struct.unpack('<i', f.read(4))[0]
        vector_dim = struct.unpack('<i', f.read(4))[0]
        data = f.read(num_vecs * vector_dim * 4)  # 4 bytes per float
        vectors = np.frombuffer(data, dtype=np.float32).reshape((num_vecs, vector_dim))
    return vectors
```

The dataset should include a ground truth file for the full set. Some datasets also include ground truth for subsets (first $k$ vectors) — e.g., SIFT100M's GT lives in `idx_100M.ivecs` inside the SIFT1B archive.

**2. Convert format (if needed):**

```bash
# convert .vecs to .bin
build/tests/utils/vecs_to_bin uint8 bigann_base.bvecs bigann.bin # for int8/uint8 vecs (SIFT)
build/tests/utils/vecs_to_bin float base.fvecs deep.bin # for float vecs (DEEP)
build/tests/utils/vecs_to_bin int32 idx_1000M.ibin # for int32/uint32 vecs (groundtruth)

# Generate 100M subsets (e.g., for SIFT and DEEP).
build/tests/utils/change_pts uint8 bigann.bin 100000000 # bigann.bin -> bigann.bin100000000
mv bigann.bin100000000 bigann_100M.bin
build/tests/utils/change_pts float deep.bin 100000000
mv deep.bin100000000 deep_100M.bin

# Compute ground truth for the 100M subset (SIFT100M example).
# compute_groundtruth <type> <metric> <data> <query> <topk> <output> null null
build/tests/utils/compute_groundtruth uint8 l2 bigann_100M.bin query.bin 1000 100M_gt.bin null null
```

## Build the Index

PipeANN supports two on-disk graph builders with the same file format:

* [**Vamana**](https://github.com/microsoft/diskann) (recommended) — DiskANN-style builder. Alpha-RNG pruning, one-by-one vector insertion.
* [**PiPNN**](http://arxiv.org/abs/2602.21247) (experimental) — partitions the dataset into overlapping sub-problems and leverages dense matrix multiplication kernels. `L1 * L2` should be comparable to Vamana's `L`.

Same command for both:

```bash
# build_disk_index <type> <data> <prefix> <R> <L_or_L1> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type> [L2]
# Vamana: omit L2, or pass 0.
build/tests/build_disk_index uint8 data.bin index 96 128 32 256 112 l2 pq

# PiPNN: pass L1 in L_or_L1, and L2 as the last argument.
build/tests/build_disk_index uint8 data.bin index 96 9 32 256 112 l2 pq 10
```

**Parameters:**

| Parameter | Meaning |
|-----------|---------|
| `R` | Maximum out-neighbors. |
| `L_or_L1` | Vamana: build-time candidate pool `L`. PiPNN: `L1`. |
| `L2` | `0` or omitted → Vamana. `L2 > 0` → PiPNN. Typically `L1 * L2 ≈ L`. |
| `PQ_bytes` | Bytes per PQ vector. `32` is a good default; raise if accuracy is low. |
| `M_GB` | Max memory (GB). PiPNN currently ignores this budget. |
| `nbr_type` | `pq` (supports update), `rabitq` (1-bit, search-only), `rabitq{3-5}` (3–5-bit, search-only). |

**Recommended Vamana parameters:**

| Dataset | Type | R | L | PQ_bytes | Memory | Threads |
|---------|------|---|---|----------|--------|---------|
| 100M subsets | uint8/float/int8 | 96 | 128 | 32 | 256GB | 112 |
| SIFT1B | uint8 | 128 | 200 | 32 | 500GB | 112 |
| SPACEV1B | int8 | 128 | 200 | 32 | 500GB | 112 |

Expect ~5h for 100M datasets and ~1d for billion-scale.

### In-Memory Entry-Point Index (optional)

An in-memory index optimizes the entry point. Skip it by setting `mem_L=0` at search time.

```bash
build/tests/utils/gen_random_slice uint8 data.bin index_SAMPLE_RATE_0.01 0.01
build/tests/build_memory_index uint8 index_SAMPLE_RATE_0.01_data.bin index_SAMPLE_RATE_0.01_ids.bin index_mem.index 32 64 1.2 $(nproc) l2
```

The output lives in two files: `index_mem.index` and `index_mem.index.tags`.

This index boosts performance for 100-dimensional datasets (SIFT, DEEP, and SPACEV) but may degrade performance for higher-dimensional datasets (e.g., Wiki).

!!! note
    PipeANN uses the same SSD layout for the in-memory and on-SSD indexes. It is **not** compatible with DiskANN's or old-version PipeANN's in-memory index format.

## Search

```bash
# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index uint8 index_prefix 1 32 query.bin gt.bin 10 l2 pq 2 10 10 20 30 40
```

**Search modes (`mode`):**

| Mode | Algorithm |
|------|-----------|
| `0` | DiskANN best-first search. |
| `1` | Starling page-reordered search. Requires a reordered index produced by the original Starling code; align the partition file via `build/tests/pad_partition`. |
| `2` | **PipeANN pipelined search (recommended).** |
| `3` | CoroSearch — coroutine-based inter-query parallel search. |

Example output:
```
Search parameters: #threads: 1,  beamwidth: 32
... some outputs during index loading ...
     L   I/O Width         QPS  AvgLat(us)     P99 Lat    Mean IOs   Recall@10
=============================================================================
    10          32     1871.92      512.01      939.00       23.24       67.40
    20          32     1678.32      560.96      926.00       32.22       84.76
    30          32     1551.03      601.63      945.00       41.19       91.13
    40          32     1420.42      654.29     1007.00       50.11       94.28
```

### Search a DiskANN Index

If you already have a DiskANN on-disk index, you can search it directly with PipeANN. Just build an in-memory entry-point index from a 1% sample first:

```bash
export INDEX_PREFIX=/mnt/nvme2/indices/bigann/100m # on-disk index filename is 100m_disk.index
export DATA_PATH=/mnt/nvme/data/bigann/100M.bbin

# Build in-memory entry point index (~10min for 1B vectors)
build/tests/utils/gen_random_slice uint8 ${DATA_PATH} ${INDEX_PREFIX}_SAMPLE_RATE_0.01 0.01
build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin ${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 32 64 1.2 $(nproc) l2

# Search with PipeANN
build/tests/search_disk_index uint8 ${INDEX_PREFIX} 1 32 query.bin gt.bin 10 l2 pq 2 10 10 20 30 40
```

## Update (Insert / Delete)

Update support requires `-DREAD_ONLY_TESTS` and `-DNO_MAPPING` to be **disabled** in `CMakeLists.txt`.

**1. Generate ground truths for updates.**

Computing exact ground truth at every insertion step is costly. PipeANN uses a shortcut: select top-10 vectors per interval from the top-1000 (or larger) of the full dataset.

```bash
# gt_update <gt_file> <index_pts> <total_pts> <batch_pts> <topk> <output_dir> <insert_only>
# Insert 100M vectors (batch=1M) into 100M index; truth.bin contains top-1000 of the 200M dataset.
build/tests/utils/gt_update truth.bin 100000000 200000000 1000000 10 /path/to/gt 1
# Insert 100M and delete the original 100M.
build/tests/utils/gt_update truth.bin 100000000 200000000 1000000 10 /path/to/gt 0
```

**2. Search-insert workload (`test_insert_search`).** Inserts vectors while concurrently searching.

```bash
# test_insert_search <type> <data> <L_disk> <step_size> <steps> <ins_thds> <srch_thds> <mode> ...
build/tests/test_insert_search uint8 data_200M.bin 128 1000000 100 10 32 2 index_prefix query.bin /path/to/gt 0 10 4 32 10 20 30 40 50
```

**3. Search-insert-delete workload (`overall_performance`).** Sliding window — inserts new and deletes old.

```bash
# overall_performance <type> <data> <L_disk> <index> <query> <gt> <recall> <beam> <steps> <Ls...>
build/tests/overall_performance uint8 data_200M.bin 128 index_prefix query.bin /path/to/gt 10 4 100 20 30
```

**Notes:**

- The index is **not crash-consistent** during updates; call `save()` for consistent snapshots.
- PipeSearch is used for both search and insert. Defaults: `W=8` for insert, `W=32` for search.
- The in-memory entry-point index is immutable during updates but still useful for entry-point optimization.

## In-Memory Workloads

PipeANN can load the entire SSD index into DRAM as an in-memory baseline (e.g., for comparison against Vamana).

**Search-only (`search_disk_index_mem`).** Same CLI as `search_disk_index`, but loads the index into RAM first.

```bash
build/tests/search_disk_index_mem uint8 index_prefix 1 32 query.bin gt.bin 10 l2 pq 2 10 10 20 30 40
```

**Search-insert-delete (`overall_perf_mem`).** Same CLI as `overall_performance`, in-memory.

```bash
build/tests/overall_perf_mem uint8 data_200M.bin 128 index_prefix query.bin /path/to/gt 10 4 100 20 30
```

## Filtered Search

PipeANN supports filtered ANNS with arbitrary attribute constraints via **speculative filtering** — both **memory-efficient** (only lightweight probabilistic filters live in RAM, not full attributes) and **high-performance**.

**How it works.** Speculative filtering explores a *superset* of valid vectors using in-memory probabilistic structures (Bloom filters, quantized values). Once a candidate set is found, exact attribute verification runs against full attributes stored alongside vectors on SSD. A cost model routes each query to the best strategy (speculative pre-filter / speculative in-filter / post-filter).

**Supported attributes.** Label filtering (OR/AND) and range filtering `[l, r)`, plus their Boolean combinations (AND/OR/NOT). Custom attribute types can be added by implementing `AttrIndex` and `Selector`.

### Example: YFCC10M LabelAnd

From the [NeurIPS'23 BigANN benchmark](https://big-ann-benchmarks.com/neurips23.html). Dataset: 10M 192-dim uint8 vectors, each with 1–1517 labels. Query: find vectors containing *all* query labels.

**1. Build the filtered index:**

```bash
# build_disk_index_filtered <type> <data> <prefix> <R> <R_dense> <L> <PQ_bytes> <M_GB> <threads>
#   <metric> <nbr_type> <label_type_1> <label_file_1> ...
build/tests/build_disk_index_filtered uint8 base.10M.u8bin yfcc10M 48 1500 72 64 500 112 l2 pq label_spmat base.metadata.10M.spmat
```

Output:
1. The graph index on SSD — each record stores `[vector | neighbors | attributes | 2-hop neighbors]` (attributes are used for exact verification).
2. Separate attribute index files on SSD (e.g., `yfcc10M.label.0` is the inverted label index), used for pre-filter scans. Only PQ-compressed vectors and lightweight probabilistic filters live in memory.

**2. Configure the query.** The config has three sections: `attr_indexes` (base attribute indexes), `filter` (a SQL-like query template), and `bindings` (per-`$$var` query-attribute `.spmat` files). Placeholders use the `$$varName` convention; each placeholder gets one row per query from its bound `.spmat`.

```json
{
    "attr_indexes": [
        { "name": "tags", "key": 0, "type": "label", "file": "yfcc10M.label.0" }
    ],
    "filter": "array_contains_all(tags, $$query_tags)",
    "bindings": {
        "query_tags": "query.metadata.public.100K.spmat"
    }
}
```

**3. Search with filter:**

```bash
# search_disk_index_filtered <type> <prefix> <threads> <beamwidth> <query> <gt> <topk>
#   <metric> <nbr_type> <config.json> <mem_L> <Ls...>
build/tests/search_disk_index_filtered uint8 yfcc10M 32 32 query.public.100K.u8bin GT.public.ibin 10 l2 pq config.json 0 10 15 20 30 40 50
```

Abbreviated result on YFCC10M (LabelAnd):
```
   L  BW       QPS Avg(us)      #Pre       #In   EstIO   #Post   AvgIO Recall@10
====================================================
  10  32   11355.8  2746.0   41738.0   58262.0   174.6     0.0    70.7      74.3
  20  32    9605.9  3273.3   49604.0   50396.0   251.4     0.0    93.5      89.8
  30  32    8093.9  3897.8   54081.0   45919.0   313.3     0.0   118.5      93.9
  50  32    6144.2  5151.0   59951.0   40049.0   405.0     0.0   171.0      97.3
```

### JSON Config Reference

| Key | Type | Description |
|-----|------|-------------|
| `attr_indexes` | array | Base attribute index definitions built at index time. |
| `filter` | string | SQL-like filter expression with `$$var` placeholders. |
| `bindings` | object | Maps each `$$var` to a query-attribute `.spmat` file. |

**`attr_indexes` array item:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Field name referenced by `filter`. |
| `key` | uint32 | Slot key in the on-disk vector record; assigned at build time. |
| `type` | string | `"label"` (inverted label index), `"range"` (numeric range index), or `"string"`. |
| `file` | string | Path to the attribute index file (output of `build_disk_index_filtered`). |

**`filter` operators.** SQL-like: `=`, `==`, `!=`, `<>`, `>`, `>=`, `<`, `<=`, `in`, `not in`, `between`, `like`, `array_contains`, `array_contains_all`, `array_contains_any`, `and`, `or`, `not`, and parentheses. Placeholders (`$$var`) can appear anywhere a literal value is accepted, including as the second argument to `array_contains_all` / `array_contains_any`.

**`bindings`.** Each `.spmat` file decodes one Attribute per query row (label: indices where `data != 0`; range: pushes the row's `[lo, hi)` interval). All bound files must share the same row count.

**Complex example — LabelOr OR Range:**

```json
{
    "attr_indexes": [
        { "name": "tags",  "key": 0, "type": "label", "file": "100M.label.0" },
        { "name": "width", "key": 1, "type": "range", "file": "100M.label.1" }
    ],
    "filter": "tags = $$query_tags or width = $$query_width",
    "bindings": {
        "query_tags":  "metadata_query.spmat",
        "query_width": "metadata_width_query.spmat"
    }
}
```

Updates are supported, but attribute index updates are currently sub-optimal (in-memory only).

## Out-of-Distribution (OOD) Search

For OOD workloads — queries and base vectors come from different distributions (e.g., text queries against image embeddings) — PipeANN supports [**NGFix**](https://dl.acm.org/doi/abs/10.1145/3769783) refinement. A fraction of each node's out-edges (`R_ood`) is replaced with "refine" edges selected from real training-query traversals. Total out-degree stays `R = R_base + R_ood`, so **disk layout, memory footprint, and the search algorithm are unchanged** — only graph topology differs.

**When to use.** Queries are noticeably OOD (text-to-image, cross-modal retrieval, multi-modal embeddings such as LAION). For in-distribution workloads, the extra build time is not worth it.

**1. Prepare training queries.** A `.bin` file in the same dtype and dimension as the base vectors. NGFix recommends a training set comparable in size to the base set. Held-out historical queries work best; public OOD benchmarks (Text-to-Image, LAION) ship with dedicated `query.train.*` files.

**2. Build with OOD refinement.** Example on Text-to-Image 10M (200-dim float IP), using the 50M learn-query split as training data:

```bash
# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M> <T> <metric> <nbr_type>
#   [L2] [train_query_path] [R_ood] [L_ood]
#
# R is the total out-degree (R_base + R_ood). R_ood is the number of refine edges per node.
# L_ood is the beam width used when computing AKNN for each training query (default 1500).
build/tests/build_disk_index float \
    /mnt/nvme/data/text2image/10M.bin \
    /mnt/nvme/indices/text2image/10M \
    96 128 64 256 112 mips pq 0 \
    /mnt/nvme/data/text2image/query.learn.50M.fbin 48 1500
```

Recommended parameters:

| Field | Recommendation |
|-------|----------------|
| `R_ood` | `R / 2` (e.g., `R=96 → R_ood=48`). Must be `< R`. |
| `L_ood` | `1500` (default). Larger → more accurate refine AKNN but slower build. |
| `train_query_path` | Comparable size, same dtype/dim as base. |

**3. Search.** OOD metadata is embedded in the graph, so search uses the same command as a regular index:

```bash
build/tests/search_disk_index float /mnt/nvme/indices/text2image/10M \
    1 32 /mnt/nvme/data/text2image/query.public.100K.fbin \
    /mnt/nvme/data/text2image/t2i_new_groundtruth.public.100K.bin \
    10 mips pq 2 10 10 20 30 40
```

!!! tip
    NGFix is compatible with filtered search — combine `R_ood` with `range_dense` / attribute indexes as needed.
