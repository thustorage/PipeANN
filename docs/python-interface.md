---
hide:
  - navigation
---

# Python Interface

PipeANN provides a Python interface via `IndexPipeANN`. The sections below walk
through the common operations; see `tests_py/index_example.py` for an end-to-end
script.

## Create an Index

```python
from pipeann import IndexPipeANN, Metric

idx = IndexPipeANN(data_dim=128, data_type='float32', metric=Metric.L2)
idx.omp_set_num_threads(32)         # search/insert/delete thread count
idx.set_index_prefix(index_prefix)  # stored at {index_prefix}_disk.index
```

## Build or Insert

There are two ways to populate the index:

- **`idx.add(vectors, tags)`** — streaming inserts. PipeANN keeps inserts in
  memory and auto-converts to an on-disk index once the count exceeds 100K. The
  on-disk index built this way has a fixed out-degree of 64.
- **`idx.build(data_path, index_prefix)`** — recommended for ≥10M vectors.
  Builds the on-disk index directly with full control over graph parameters.
  Pass `attrs=AttrsVec(...)` to embed per-vector attributes into disk nodes for
  later filtered search.

```python
# Streaming insert.
idx.add(vectors, tags)

# Or build from a pre-existing dataset file.
# idx.build(data_path, index_prefix)
# idx.load(index_prefix)

# Or load a pre-built on-disk index.
# idx.load(index_prefix)
```

## Search

`idx.search` runs PipeANN on the on-disk index (or best-first search if the index is still in memory):

```python
ids, dists = idx.search(queries, topk=10, L=50)
```

## Update and Save

```python
idx.add(vectors, tags)   # insert vectors
idx.remove(tags)         # delete vectors by tag
idx.save(index_prefix)   # persist after updates
```

Run the end-to-end example (edit hard-coded paths first):
```bash
cd tests_py && python index_example.py
```

Example output:
```bash
(2000000, 128)
Building index with prefix /mnt/nvme/indices/bigann/1M...
Inserting the first 1M points 0 to 10000 ...
# ...
Inserting the first 1M points 990000 to 1000000 ...
Loading index with prefix /mnt/nvme/indices/bigann/1M...
Searching for 10 nearest neighbors with L=10...
Search time: 0.6878 seconds for 10000 queries, throughput: 14539.948333885677 QPS.
Recall@10 with L=10: 0.7486
# ...
Searching for 10 nearest neighbors with L=50...
Search time: 0.8825 seconds for 10000 queries, throughput: 11331.721143151088 QPS.
Recall@10 with L=50: 0.9800
Inserting 1M new vectors to the index ...
Inserting data points 1000000 to 1010000 ...
# ...
Inserting data points 1990000 to 2000000 ...
Deleting the first 1M vectors from the index ...
Searching for 10 nearest neighbors with L=10...
Search time: 0.6270 seconds for 10000 queries, throughput: 15948.851777308719 QPS.
Recall@10 with L=10: 0.7423
# ...
Searching for 10 nearest neighbors with L=50...
Search time: 0.8633 seconds for 10000 queries, throughput: 11583.198564825861 QPS.
Recall@10 with L=50: 0.9653
```

## Range Search

Range search reuses `idx.search()` with a finite `range` threshold (an on-disk
index is required). Only neighbors within the threshold are returned; unused
result slots are padded with `UINT32_MAX` / `inf`.

`range` is interpreted in the user-facing metric:

- `Metric.L2` — maximum L2 distance. `range=0.2` keeps vectors within L2 ≤ 0.2.
- Cosine — maximum cosine distance (`1 − similarity`). `range=0.1` keeps
  vectors with similarity ≥ 0.9.

```python
idx.load(index_prefix)

# L2: L2 distance <= 0.2.
ids, dists = idx.search(queries, topk=10, L=200, range=0.2)

# Cosine (when the index was built with the cosine metric): similarity >= 0.9.
ids, dists = idx.search(queries, topk=10, L=200, range=0.1)
```

Internally this uses the same pipelined traversal as top-k search, with a
range-aware early-stop heuristic.

Smoke test:

```bash
cd tests_py && python test_range_search.py
```

## Filtered Search

Filtered search in Python shares the C++ filtered-search machinery — same
indexes, same selector semantics. See [C++ Interface — Filtered Search](cpp-interface.md#filtered-search)
for the index-build path, attribute file format, JSON config schema, and
selector reference.

Two ways to set up a filter:

**1. Compile a SQL-like filter (recommended for per-query workflows):**
```python
idx.load(index_prefix)
tag_index = idx.load_attr_index_from_file(0, "base.label.0", "label")
range_index = idx.load_attr_index_from_file(1, "base.range.1", "range")

# Schema: field name → (key, type, attr_index).
schema = {
    "tags":  (0, "label", tag_index),
    "width": (1, "range", range_index),
}
selector, attrs, slot_map, var_field_type = idx.compile_filter(
    "tags = 7 and width > 100 and width < 500",
    schema,
)
ids, dists = idx.search(queries, topk=10, L=50, selector=selector, query_attrs=[attrs])
```
For literal filters `slot_map` is empty and `attrs` can be passed directly. Use `$$var` placeholders in the filter string to leave slots open for late binding; the easiest way to bind from `.spmat` files is the unified config loader below.

**1b. Load the unified config end-to-end (batch `.spmat` binding):**
```python
idx.load(index_prefix)
selector, query_attrs = idx.load_filter_from_json("filter.json")
ids, dists = idx.search(queries, topk=10, L=50, selector=selector, query_attrs=query_attrs)
```
The config is the same `{attr_indexes, filter, bindings}` schema documented in
[C++ Interface — Filtered Search](cpp-interface.md#filtered-search). Each declared
`attr_indexes` entry is loaded into the index (so subsequent `add()` calls route
attributes to the right place), and `query_attrs` has one row per query row in
the bound `.spmat` files.

**2. Compose a native selector in Python:**
```python
from pipeann import AndSelector, AttrsVec, LabelOrSelector, RangeSelector

idx.load(index_prefix)
tag_index = idx.load_attr_index_from_file(0, "base.label.0", "label")
range_index = idx.load_attr_index_from_file(1, "base.range.1", "range")

selector = AndSelector(
    LabelOrSelector(key=0, base_key=0, attr_index=tag_index),
    RangeSelector(key=1, base_key=1, attr_index=range_index),
)

# Query attrs use the same row-oriented container as build attrs.
query_attrs = AttrsVec()
query_attrs.load_from_file(0, "label", "tag_query.spmat")
query_attrs.load_from_file(1, "range", "range_query.spmat")
ids, dists = idx.search(queries, topk=10, L=50, selector=selector, query_attrs=query_attrs)
```

You can also subclass `Selector` to implement a selector in Python — but each
callback crosses the C++/pybind/Python boundary, so this path is much slower.
Prefer native selectors for performance-critical workloads.

See `tests_py/test_filter.py` (Python callback) and `tests_py/test_native_selector.py`
(native composition) for runnable examples:

```bash
cd tests_py && python test_native_selector.py
cd tests_py && python test_filter.py
```

## OOD Search

Pass `R_ood` to enable NGFix refine. The search path is unchanged.

```python
idx = IndexPipeANN(data_dim=200, data_type="float32", metric=Metric.INNER_PRODUCT)
idx.build(
    "/mnt/nvme/data/text2image/10M.bin",
    "/mnt/nvme/indices/text2image/10M",
    max_nbrs=48,
    train_query_path="/mnt/nvme/data/text2image/query.learn.50M.fbin",
    R_ood=32,
    # L_ood defaults to 1500; build_L/PQ_bytes default to auto.
)
# Search path is unchanged.
ids, dists = idx.search(queries, topk=10, L=50)
```
