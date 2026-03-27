# PipeANN

**PipeANN** is a **low-latency**, **billion-scale**, and **updatable** graph-based vector store on SSD.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **Ultra-Low Latency** | <1ms for 1 billion vectors (top-10, 90% recall), only 1.14x-2.02x of in-memory index |
| 📈 **High Throughput** | 20K QPS for 1 billion vectors, outperforming DiskANN and SPANN |
| 🔄 **Efficient Updates** | Insert/delete with minimal search interference (1.07x fluctuation) |
| 🎯 **User-Defined Filters** | Supports arbitrary filtered ANNS via user-defined `Label` and `Selector` |
| 💾 **Memory Efficient** | >10x less memory than in-memory indexes (~40GB for 1B vectors) |
| 🐍 **Easy-to-Use** | Both Python (`faiss`-like) and C++ interfaces supported |

## 📊 Performance Comparison

PipeANN is suitable for both **large-scale** and **memory-constraint** scenarios.


| Dataset | Dimension | Memory | Latency | QPS | PipeANN | HNSW | DiskANN |
|---------|-----|--------|---------|-----|---------|-------------| -------- |
| 1B (SPACEV) | 100 | 40GB | 2ms | 5K | ✅ | ❌ 1TB mem | ❌ 6ms |
| 80M (Wiki) | 768 | 10GB | 1.5ms | 5K | ✅ | ❌ 300GB mem | ❌ 4ms |
| 10M (SIFT) | 128 | 550MB | <1ms | 10K | ✅ | ❌ 4GB mem | ❌ 3ms |

> Recall@10 = 0.99, Samsung PM9A3 SSD, 32B PQ-compressed vectors (128B for Wiki).

---

## 📰 Updates

- **Mar 27, 2026**: [PiPNN](http://arxiv.org/abs/2602.21247) indexing algorithm supported
- **Dec 4, 2025**: Inner product and filtered ANNS (*arbitrary filter*) supported
- **Oct 14, 2025**: [RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library) (1-bit and multi-bit quantization) supported
- **Sep 29, 2025**: Python interface released
- **Jul 16, 2025**: Vector update (insert/delete) supported

---

## 🔧 Prerequisites

### Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **CPU** | x86 or ARM with SIMD (AVX2/AVX512 recommended) |
| **DRAM** | ~40GB (search) or ~90GB (search+update) per billion vectors |
| **SSD** | ~700GB for 1B SIFT, ~900GB for 1.4B SPACEV |

### Software Requirements

- **OS**: Linux with `io_uring` (fast) or `libaio` (compatible) support, Ubuntu 22.04 recommended
- **Compiler**: C++17 support required
- **Constraints**: <2B vectors to avoid integer overflow

### Install Dependencies

```bash
# Ubuntu >= 22.04
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev \
                 clang-format libmkl-full-dev libeigen3-dev

# For Python interface
pip3 install "pybind11[global]"
```

`libmkl` could be replaced by other BLAS libraries (e.g., `openblas`).

---

## 🏗️ Build PipeANN

### Step 1: Build liburing

```bash
cd third_party/liburing
./configure && make -j
cd ../..
```

### Step 2: Build PipeANN

**For Python interface:**
```bash
python setup.py install  # Installs `pipeann` wheel
```

**For C++ interface:**
```bash
bash ./build.sh  # Binaries in build/
```

For performance-critical scenarios, we recommend using C++ interface.

---

## 🚀 Quick Start

### 🐍 Python Interface

```python
from pipeann import IndexPipeANN, Metric

# Create index
idx = IndexPipeANN(data_dim=128, data_type='float32', metric=Metric.L2)
idx.omp_set_num_threads(32) # the number of search/insert/delete threads.
idx.set_index_prefix(index_prefix) # the index is stored to {index_prefix}_disk.index

# Insert vectors in memory (auto-converts to disk index when >100K vectors)
idx.add(vectors, tags)

# For SSD index initialized using idx.add, out-neighbor number is fixed to 64.
# For large-scale datasets (>= 10M), we recommend using idx.build for initialization.
# idx.build(data_path, index_prefix)
# idx.load(index_prefix) # load the pre-built index from disk.

# Search using PipeSearch (on-SSD) or best-first search (in-memory)
results = idx.search(queries, topk=10, L=50)

idx.remove(tags) # remove vectors from the index with corresponding tags.
# The index should be saved after updates.
idx.save(index_prefix) # save the index.
```

Run an example (hard-coded paths should be modified):
```bash
cd tests_py && python index_example.py
```

Example result:
```bash
python setup.py install
cd tests_py
# Please modify the hard-coded paths first!
python index_example.py
```

It runs like this:
```bash
# Insert the first 100K vectors using in-memory index.
[index.cpp:68:INFO] Getting distance function for metric: l2
Building index with prefix /mnt/nvme/indices/bigann/1M...
# ...
Inserting the first 1M points 100000 to 110000 ...
# Transform the in-memory index to SSD index.
[pyindex.h:100:INFO] Transform memory index to disk index.
# ...
[pyindex.h:109:INFO] Transform memory index to disk index done.
# Insert the remaining 900K vectors, save, and reload the SSD index.
Inserting the first 1M points 110000 to 120000 ...
# ...
[ssd_index.cpp:206:INFO] SSDIndex loaded successfully.
# The first search in the SIFT1M dataset.
Searching for 10 nearest neighbors with L=10...
Search time: 0.6290 seconds for 10000 queries, throughput: 15897.957218870273 QPS.
Recall@10 with L=10: 0.7397
# ...
Searching for 10 nearest neighbors with L=50...
Search time: 0.8746 seconds for 10000 queries, throughput: 11433.789824882691 QPS.
Recall@10 with L=50: 0.9784
# Insert the second 1M vectors, save and reload.
Inserting 1M new vectors to the index ...
# ...
[ssd_index.cpp:206:INFO] SSDIndex loaded successfully.
# The second search in the SIFT2M dataset.
Searching for 10 nearest neighbors with L=10...
Search time: 0.6461 seconds for 10000 queries, throughput: 15477.096553625139 QPS.
Recall@10 with L=10: 0.7181
# ...
Searching for 10 nearest neighbors with L=50...
Search time: 0.8907 seconds for 10000 queries, throughput: 11227.508131590563 QPS.
Recall@10 with L=50: 0.9720
```

### ⚡ C++ Interface (Search-Only)

Enable `-DREAD_ONLY_TESTS` and `-DNO_MAPPING` in `CMakeLists.txt`. This disables updates but achieves higher search performance.

**For DiskANN users** (existing on-disk index):
```bash
# Build in-memory entry point index (~10min for 1B vectors)
export INDEX_PREFIX=/mnt/nvme2/indices/bigann/100m # on-disk index filename is 100m_disk.index
export DATA_PATH=/mnt/nvme/data/bigann/100M.bbin

build/tests/utils/gen_random_slice uint8 ${DATA_PATH} ${INDEX_PREFIX}_SAMPLE_RATE_0.01 0.01
build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin \
    ${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 32 64 1.2 $(nproc) l2

# Search with PipeANN
build/tests/search_disk_index uint8 ${INDEX_PREFIX} 1 32 query.bin gt.bin 10 l2 pq 2 10 10 20 30 40
```

Example results:
```
Search parameters: #threads: 1,  beamwidth: 32
... some outputs during index loading ...
[search_disk_index.cpp:216:INFO] Use two ANNS for warming up...
[search_disk_index.cpp:219:INFO] Warming up finished.
     L   I/O Width         QPS    Mean Lat     P99 Lat   Mean Hops    Mean IOs   Recall@10
=========================================================================================
    10          32     1952.03      490.99     3346.00        0.00       22.28       67.11
    20          32     1717.53      547.84     1093.00        0.00       31.11       84.53
    30          32     1538.67      608.31     1231.00        0.00       41.02       91.04
    40          32     1420.46      655.24     1270.00        0.00       52.50       94.23
```

**Starting from scratch:**

**1. Download datasets**: [SIFT](http://corpus-texmex.irisa.fr/), [DEEP1B](https://github.com/matsui528/deep1b_gt), [SPACEV](https://github.com/microsoft/SPTAG). 

If the links are not available, you could get the datasets from [Big ANN benchmarks](https://big-ann-benchmarks.com/neurips21.html).

SPACEV1B may comprises several sub-files. To concatenate them, save the dataset's numpy `array` to `bin` format (the following Python 
code might be used).

```py
# bin format:
# | 4 bytes for num_vecs | 4 bytes for vector dimension (e.g., 100 for SPACEV) | flattened 
vectors |
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

The dataset should contain a ground truth file for its full set.
Some datasets also contain the ground truth of subsets (first $k$ vectors). For example, 
SIFT100M's (the first 100M vectors of SIFT1B) ground truth could be found in `idx_100M.ivecs` of 
SIFT1B dataset.

**2. Convert format** (if needed):
```bash
# convert .vecs to .bin
build/tests/utils/vecs_to_bin uint8 bigann_base.bvecs bigann.bin # for int8/uint8 vecs (SIFT)
build/tests/utils/vecs_to_bin float base.fvecs deep.bin # for float vecs (DEEP)
build/tests/utils/vecs_to_bin int32 idx_1000M.ibin # for int32/uint32 vecs (groundtruth) 

# Generate 100M subsets (e.g., for SIFT and DEEP).
build/tests/utils/change_pts uint8 bigann.bin 100000000 # bigann.bin -> bigann.bin100000000
mv bigann.bin100000000 bigann_100M.bin
build/tests/utils/change_pts float deep.bin 100000000 # deep.bin -> deep.bin100000000
mv deep.bin100000000 deep_100M.bin

# Calculate Ground Truth for 100M subsets (SIFT100M example)
# compute_groundtruth <type> <metric> <data> <query> <topk> <output> null null
build/tests/utils/compute_groundtruth uint8 l2 bigann_100M.bin query.bin 1000 100M_gt.bin null null
```
  
**3 Build on-disk index**:

PipeANN now supports two indexing building algorithms (with the same file format):
  
**3.1 Build on-disk index (Vamana)**:

To build a Vamana (DiskANN-style) index, run:

```bash
# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index uint8 data.bin index 96 128 32 256 112 l2 pq
```

**Parameter explanation:**
- `R`: Maximum out-neighbors.
- `L`: Candidate pool size during build.
- `PQ_bytes`: Bytes per PQ vector (32 recommended, use a larger value if accuracy is low).
- `M`: Max memory (GB).
- `nbr_type`: `pq` (product quantization, supports update), `rabitq` (1-bit quantization, search-only), `rabitq{3-5}` (3-bit to 5-bit quantization, search-only).

**Recommended Parameters:**

| Dataset | Type | R | L | PQ_bytes | Memory | Threads |
|---------|------|---|---|----------|--------|---------|
| 100M subsets | uint8/float/int8 | 96 | 128 | 32 | 256GB | 112 |
| SIFT1B | uint8 | 128 | 200 | 32 | 500GB | 112 |
| SPACEV1B | int8 | 128 | 200 | 32 | 500GB | 112 |

This requires ~5h for 100M-scale datasets, and ~1d for billion-scale datasets.

**3.2 Build on-disk index (PiPNN)**

[PiPNN](http://arxiv.org/abs/2602.21247) (Pick-in-Partitions Nearest Neighbors) is a graph construction algorithm. It partitions the dataset into overlapping sub-problems and leverages dense matrix multiplication kernels.

```bash
# build_pipnn_index <type> <data> <prefix> <R> <L1fanout> <L2fanout> <PQ_bytes> <M_GB (unused)> <threads> <metric> <nbr_type>
build/tests/build_pipnn_index uint8 data.bin index 96 9 10 32 256 112 l2 pq
```

Most parameters are similar to Vamana, except:
- `L1fanout` and `L2fanout`: Control the replication factor during construction. Typically `L1fanout * L2fanout` works similarly to `L` in Vamana.
- `M_GB`: Memory limits are not supported for PiPNN currently, so this option is unused.


**4. Build in-memory index** (optional but recommended):

An in-memory index optimizes the entry point. Skip it by setting `mem_L=0` in search.

```bash
build/tests/utils/gen_random_slice uint8 data.bin index_SAMPLE_RATE_0.01 0.01
build/tests/build_memory_index uint8 index_SAMPLE_RATE_0.01_data.bin \
    index_SAMPLE_RATE_0.01_ids.bin index_mem.index 32 64 1.2 $(nproc) l2
```

The output in-memory index should reside in three files: `index_mem.index`, `index_mem.index.data`, and 
`index_mem.index.tags`.

**5. Search**:
```bash
# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index uint8 index_prefix 1 32 query.bin gt.bin 10 l2 pq 2 10 10 20 30 40
```

**Search Modes (`mode`):**
- `0` (DiskANN): Best-first search.
- `1` (Starling): Page-reordered search. Requires reordered index using the original Starling code and use `build/tests/pad_partition` to align the generated partition file.
- `2` (PipeANN): Pipelined search (**Recommended**).
- `3` (CoroSearch): Coroutine-based inter-query parallel  search.

### 🔄 Search + Update

Disable `-DREAD_ONLY_TESTS` and `-DNO_MAPPING` flags in `CMakeLists.txt` for update support.

**1. Prepare Tags (Optional)**

Each vector corresponds to one tag. PipeANN uses identity mapping (ID -> tag) by default. Use `gen_tags` to generate explicit mapping (necessary for FreshDiskANN).

```bash
# gen_tags <type> <data> <output_prefix>
build/tests/utils/gen_tags uint8 data.bin index_prefix
```

**2. Generate Ground-Truths for Updates**

Calculating exact ground truth for every insertion step is costly. We use a tricky approach: select top-10 vectors for each interval from the top-1000 (or more) of the whole dataset (or a larger subset).

```bash
# gt_update <gt_file> <index_pts> <total_pts> <batch_pts> <topk> <output_dir> <insert_only>
# Example: Insert 100M vectors (batch=1M) into 100M index. 
# truth.bin contains top-1000 for the 200M dataset.
build/tests/utils/gt_update truth.bin 100000000 200000000 1000000 10 /path/to/gt 1
# Example: Insert 100M vectors and delete the original 100M vectors.
build/tests/utils/gt_update truth.bin 100000000 200000000 1000000 10 /path/to/gt 0
```

**3. Run Benchmarks**

**Search-Insert Workload (`test_insert_search`):**
Inserts vectors while concurrently searching.

```bash
# Usage: test_insert_search <type> <data> <L_disk> <step_size> <steps> <ins_thds> <srch_thds> <mode> ...
build/tests/test_insert_search uint8 data_200M.bin 128 1000000 100 10 32 2 \
    index_prefix query.bin /path/to/gt 0 10 4 32 10 20 30 40 50
```

**Search-Insert-Delete Workload (`overall_performance`):**
Inserts new vectors and deletes old ones (sliding window).

```bash
# Usage: overall_performance <type> <data> <L_disk> <index> <query> <gt> <recall> <beam> <steps> <Ls...>
build/tests/overall_performance uint8 data_200M.bin 128 index_prefix query.bin \
    /path/to/gt 10 4 100 20 30
```

**Notes:**
- Index is **not crash-consistent** after updates; use `final_merge` for consistent snapshots
- For update workloads, use `search_mode=2` (PipeANN) with `search_beam_width=32` for best performance
- In-memory index is immutable during updates but still useful for entry point optimization

### 🧠 In-Memory Workloads (Load SSD Index to DRAM)

PipeANN supports loading the entire SSD index into DRAM to use as an in-memory baseline (e.g., Vamana).

**Search-Only (`search_disk_index_mem`)**

Usage is identical to `search_disk_index`, but loads the index to memory first.

```bash
# search_disk_index_mem <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index_mem uint8 index_prefix 1 32 query.bin gt.bin 10 l2 pq 2 10 10 20 30 40
```

**Search-Insert-Delete (`overall_perf_mem`)**

Usage is identical to `overall_performance`, but operates entirely in memory.

```bash
# overall_perf_mem <type> <data> <L_disk> <index> <query> <gt> <recall> <beam> <steps> <Ls...>
build/tests/overall_perf_mem uint8 data_200M.bin 128 index_prefix query.bin \
    /path/to/gt 10 4 100 20 30
```

### 🏷️ Filtered Search

PipeANN supports filtered search using post-filtering.

To achieve this, two new classes are introduced:
* `AbstractLabel` class stores the labels for each data, used for filtering.
* `AbstractSelector` class filters the labels given a query label and a target label (as well as the target ID), using `is_member` function.

We implemented some example `Label`s and `Selector`s, including spmat `Label` and (range) filtered `Selectors`.
If arbitrary label or selector is required, you could implement them by deriving from the `Abstract` classes.

The labels are directly stored at the end of each record, so in the graph:
* Each record contains `[ Vector | R | R neighbors | labels ]`. 
* The total size is fixed to `max_node_len`, which may be larger than `vector_size` + (1 + R) * `sizeof(uint32_t)`.
* A new metadata, `label_size`, is introduced to the metadata page.

`build_disk_index` and `pipe_search` are extended to support building and searching a filtered graph index. An example for [YFCC10M in NIPS'23 BigANN benchmark](https://big-ann-benchmarks.com/neurips23.html):

**1. Build Filtered Index**

Two new arguments: `label_type` (e.g., `spmat`) and `label_file`.

```bash
# Build index with labels
# For yfcc10M, the labels should be with filename base.metadata.10M.spmat
build/tests/build_disk_index uint8 data.bin index 64 96 32 500 112 l2 pq spmat labels.spmat
```

**2. Search with Filter**

Use `search_disk_index_filtered`. Requires `selector_type` (e.g., `subset`) and `query_label_file`.

```bash
# Search with filter
# For yfcc10M, the query_labels should be with filename query.metadata.public.100K.spmat.
build/tests/search_disk_index_filtered uint8 index 16 32 query.bin gt.bin 10 l2 pq \
    subset query_labels.spmat 0 0 20 50 100 200 300
```

Example result on YFCC10M:
```
     L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean Hops    Mean IOs   Recall@10
==========================================================================================
    20          32     8836.26     1777.56     3402.00        0.00       49.59       13.87
    50          32     6357.16     2465.66     4110.00        0.00       77.99       21.00
   100          32     4164.85     3758.77     5982.00        0.00      126.23       27.96
   200          32     2423.22     6490.44     9512.00        0.00      223.67       36.13
   300          32     1697.97     9250.71    12881.00        0.00      321.85       41.19
```

---

## 📁 Code Structure

```bash
PipeANN/
├── src/                          # Core implementation
│   ├── index.cpp                    # In-memory Vamana index
│   ├── ssd_index.cpp                # On-disk index (search-only)
│   ├── search/                   # Search algorithms
│   │   ├── pipe_search.cpp          # 🌟 PipeANN search (main algorithm)
│   │   ├── beam_search.cpp          # DiskANN best-first search
│   │   ├── page_search.cpp          # Starling page-based search
│   │   └── coro_search.cpp          # Coroutine-based multi-query search
│   ├── update/                   # Update operations
│   │   ├── direct_insert.cpp        # 🌟 OdinANN direct insert
│   │   ├── delete_merge.cpp         # Delete and merge logic
│   │   └── dynamic_index.cpp        # Dynamic index wrapper (search-update)
│   └── utils/                    # Utilities
│       ├── distance.cpp             # Distance computation (L2/IP/cosine)
│       ├── linux_aligned_file_reader.cpp  # io_uring/AIO support
│       └── ...
├── include/                      # Header files
│   ├── index.h                      # In-memory index interface
│   ├── ssd_index.h                  # On-disk index interface
│   ├── dynamic_index.h              # Dynamic index interface
│   └── filter/                   # Filtered search support
├── tests/                        # Test programs & benchmarks
│   ├── build_disk_index.cpp         # Build on-disk index
│   ├── build_memory_index.cpp       # Build in-memory index
│   ├── search_disk_index.cpp        # Search benchmark (SSD)
│   ├── search_disk_index_mem.cpp    # Search benchmark (Load SSD index to RAM)
│   ├── search_disk_index_filtered.cpp # Filtered search benchmark
│   ├── test_insert_search.cpp       # Insert-search benchmark
│   ├── overall_performance.cpp      # Insert-delete-search benchmark
│   ├── pad_partition.cpp            # Pad partition file (for Starling)
│   └── utils/                    # Data utilities
├── tests_py/                     # Python examples
├── pipeann/                      # Python package
├── scripts/                      # Evaluation scripts
└── third_party/                  # Dependencies (liburing)
```

## 📂 Directory Structure & Path Assumptions

The provided scripts in `scripts/` assume a specific directory structure for datasets and indexes. 
**Please modify the hard-coded paths in the scripts** (or create symlinks) if your environment differs.

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

---

## 📜 Scripts Reference

The scripts are designed to reproduce the figures in our papers.
> **Note**: Before running, ensure your data paths match the [Directory Structure](#-directory-structure--path-assumptions) above, or edit the scripts (`eval_f.sh`, `fig*.sh`) to point to your locations.

### Directory Structure

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
└── validate_index_structure.py      # Index validation tool
```

### Usage Examples

**Hello World (verify installation):**
```bash
# PipeANN search-only test (~1 min)
bash scripts/tests-pipeann/hello_world.sh

# OdinANN update test (~1 min)
bash scripts/tests-odinann/hello_world.sh
```

**Run individual experiments:**

*   **PipeANN (Search-Only)**:
    *   `fig11.sh`: Latency vs Recall (100M datasets)
    *   `fig12.sh`: Throughput vs Recall (100M datasets)
    *   `fig13.sh`: Latency breakdown
    *   `fig14.sh` ~ `fig18.sh`: Other evaluations (ablation, scalability, etc.)

*   **OdinANN (Search-Update)**:
    *   `fig6.sh`: Insert-search on SIFT100M (~4d)
    *   `fig7.sh`: Insert-search on DEEP100M (~4d)
    *   `fig8.sh`: Insert-search on SIFT1B (~8d)
    *   `fig12.sh`: Insert-delete-search (~6d)

**Plot results:**
```bash
cd scripts/tests-pipeann && python plotting.py
# Or use Jupyter: plotting.ipynb
```


---

## 📖 Citation

If you use PipeANN in your research, please cite our papers:

```bibtex
@inproceedings{fast26odinann,
  author    = {Hao Guo and Youyou Lu},
  title     = {OdinANN: Direct Insert for Consistently Stable Performance 
               in Billion-Scale Graph-Based Vector Search},
  booktitle = {24th USENIX Conference on File and Storage Technologies (FAST 26)},
  year      = {2026},
  address   = {Santa Clara, CA},
  publisher = {USENIX Association}
}

@inproceedings{osdi25pipeann,
  author    = {Hao Guo and Youyou Lu},
  title     = {Achieving Low-Latency Graph-Based Vector Search via 
               Aligning Best-First Search Algorithm with SSD},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year      = {2025},
  address   = {Boston, MA},
  pages     = {171--186},
  publisher = {USENIX Association}
}
```

---

## 🙏 Acknowledgments

PipeANN is based on [DiskANN and FreshDiskANN](https://github.com/microsoft/DiskANN/tree/diskv2). We sincerely appreciate their excellent work!
