---
hide:
  - navigation
---

<h1 style="display: none;">PipeANN</h1>

<p align="center">
  <img src="assets/logo-wordmark.svg" alt="PipeANN" width="50%">
</p>

**PipeANN** is a **low-latency**, **billion-scale**, and **updatable** graph-based vector store on SSD.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **Ultra-Low Latency** | <1ms for 1 billion vectors (top-10, 90% recall), only 1.14x-2.02x of in-memory index |
| 📈 **High Throughput** | 20K QPS for 1 billion vectors, outperforming DiskANN and SPANN |
| 🔄 **Efficient Updates** | Insert/delete with minimal search interference (1.07x fluctuation) |
| 🎯 **Speculative Filtering** | 3K QPS & 6ms latency for attribute-filtered ANNS on 100 million vectors |
| 💾 **Memory Efficient** | >10x less memory than in-memory indexes (~40GB for 1B vectors) |
| 🐍 **Easy-to-Use** | Both Python (`faiss`-like) and C++ interfaces supported |
| 🔌 **Seamless Integration** | Milvus-compatible API (drop-in MilvusClient + gRPC server) |
| 🗄️ **Multi-SSD Scaling** | Scales to 70K QPS & 2ms tail latency on 1B vectors (4 SSDs, SPDK backend) |

## 📊 Performance Comparison

PipeANN is suitable for both **large-scale** and **memory-constraint** scenarios.


| Dataset | Dimension | Memory | Latency | QPS | PipeANN | HNSW | DiskANN |
|---------|-----|--------|---------|-----|---------|-------------| -------- |
| 1B (SPACEV) | 100 | 40GB | 2ms | 5K | ✅ | ❌ 1TB mem | ❌ 6ms |
| 80M (Wiki) | 768 | 10GB | 1.5ms | 5K | ✅ | ❌ 300GB mem | ❌ 4ms |
| 10M (SIFT) | 128 | 550MB | <1ms | 10K | ✅ | ❌ 4GB mem | ❌ 3ms |

> Recall@10 = 0.99, Samsung PM9A3 SSD, 32B PQ-compressed vectors (128B for Wiki).

---

## 🚀 Quick Start

### Requirements

| Component | Requirement |
|-----------|-------------|
| **CPU** | x86 or ARM with SIMD (AVX2/AVX512 recommended) |
| **DRAM** | ~40GB (search) or ~90GB (search+update) per billion vectors |
| **SSD** | ~700GB for 1B SIFT, ~900GB for 1.4B SPACEV |

- **OS**: Linux with `io_uring` (fast) or `libaio` (compatible) support, Ubuntu 22.04 with Kernel 6.8 recommended
- **Compiler**: C++17 support required
- **Constraints**: <2B vectors to avoid integer overflow
- **RAID users**: Use Kernel > 6.1 for better `io_uring` performance ([liburing#792](https://github.com/axboe/liburing/issues/792)), or use [SPDK backend](cpp-interface-spdk.md).

### Build

Install dependencies:

```bash
# Ubuntu >= 22.04
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev \
                 clang-format libmkl-full-dev libeigen3-dev

# For Python interface
pip3 install "pybind11[global]"
```

`libmkl` could be replaced by other BLAS libraries (e.g., `openblas`).

Build liburing:

```bash
cd third_party/liburing
./configure && make -j
cd ../..
```

Build PipeANN:

```bash
# For Python users: editable install of the Python package
pip install -e .

# For C++ users: build binaries under build/
bash ./build.sh
```

For the optional **Milvus-compatible gRPC server**, install gRPC, Protobuf, and
RocksDB, then build the server target:

```bash
sudo apt install libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev \
                 protobuf-compiler librocksdb-dev

cd build_server && cmake -DBUILD_MILVUS_SERVER=ON .. && make -j pipeann_milvus_server
```

### C++ Interface

```bash
# Search an existing on-disk index with PipeANN pipelined search
build/tests/search_disk_index uint8 index_prefix 1 32 query.bin gt.bin 10 l2 pq 2 10 10 20 30 40
```

See [C++ Interface](cpp-interface.md) for index building, updates, and filtered / OOD search,
and [SPDK Backend](cpp-interface-spdk.md) to use SPDK as I/O engine (fastest).

### Python Interface

```python
from pipeann import IndexPipeANN, Metric

idx = IndexPipeANN(data_dim=128, data_type='float32', metric=Metric.L2)
idx.omp_set_num_threads(32)
idx.set_index_prefix(index_prefix)
idx.add(vectors, tags)                          # insert (auto disk-convert at 100K)
ids, dists = idx.search(queries, topk=10, L=50) # search
idx.save(index_prefix)                          # persist
```

See [Python Interface](python-interface.md) for the full API, including filtered / OOD search and example output.

### Application Integrations

PipeANN speaks the **Milvus API** — both an in-process `MilvusClient` and a
high-performance C++ gRPC server — so existing Milvus applications can switch to
PipeANN in one line:

```python
# In-process, drop-in for pymilvus.MilvusClient (URI is a directory):
from pipeann import MilvusClient
client = MilvusClient(uri="./pipeann-data")

# Or connect the stock pymilvus client to the PipeANN gRPC server:
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
```

See [Application Integrations](application-integrations.md) for details and
Milvus-vs-PipeANN benchmarks.

---

## 📰 Updates

- **May 18, 2026**: SPDK backend supported, stable tail latency with better multi-SSD scalability
- **May 18, 2026**: Filtered Search (Speculative Filtering), OOD search ([NGFix](https://dl.acm.org/doi/abs/10.1145/3769783)) & range search supported
- **May 18, 2026**: PipeANN is integrated into OdinANN (search + insert), higher performance with less threads
- **Mar 27, 2026**: [PiPNN](http://arxiv.org/abs/2602.21247) indexing algorithm supported
- **Dec 4, 2025**: Inner product and filtered ANNS (*arbitrary filter*) supported
- **Oct 14, 2025**: [RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library) (1-bit and multi-bit quantization) supported
- **Sep 29, 2025**: Python interface released
- **Jul 16, 2025**: Vector update (insert/delete) supported

---

## 📖 Citation

If you use PipeANN in your research, please cite our papers:

```bibtex
@misc{arxiv26pipeannfilter,
      title={PipeANN-Filter: An Efficient Filtered Vector Search System on SSD}, 
      author={Hao Guo and Jiwu Shu and Youyou Lu},
      year={2026},
      eprint={2605.17992},
      archivePrefix={arXiv},
      primaryClass={cs.OS},
      url={https://arxiv.org/abs/2605.17992}, 
}

@inproceedings{fast26odinann,
  author    = {Hao Guo and Youyou Lu},
  title     = {OdinANN: Direct Insert for Consistently Stable Performance 
               in Billion-Scale Graph-Based Vector Search},
  booktitle = {24th USENIX Conference on File and Storage Technologies (FAST 26)},
  year      = {2026},
  address   = {Santa Clara, CA},
  pages     = {133--147},
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

PipeANN is based on [DiskANN and FreshDiskANN](https://github.com/microsoft/DiskANN/tree/diskv2).
We sincerely appreciate their excellent work.

