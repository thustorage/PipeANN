#!/usr/bin/env python3
"""Benchmark: PipeANN Milvus gRPC server vs Milvus server.

Supports two modes:
  1. Unfiltered (SIFT1M / BigANN): pure vector search
  2. Filtered (YFCC10M): vector search with label AND filter

Same pymilvus MilvusClient code path for both servers — only the URI differs.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as _mp
_mp_ctx = _mp.get_context("fork")
Process = _mp_ctx.Process
Queue = _mp_ctx.Queue
Barrier = _mp_ctx.Barrier
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def read_bbin(path: str, n: int) -> np.ndarray:
    """Read .u8bin / .bbin vector file -> float32."""
    h = np.fromfile(path, dtype=np.uint32, count=2)
    n_file, dim = int(h[0]), int(h[1])
    n = min(n, n_file)
    v = np.fromfile(path, dtype=np.uint8, offset=8, count=n * dim)
    return v.reshape(n, dim).astype(np.float32)


def read_gt(path: str, nq: int, topk: int = 10) -> np.ndarray:
    """Read ground truth (n, k) header + uint32 ids."""
    h = np.fromfile(path, dtype=np.uint32, count=2)
    n_file, k = int(h[0]), int(h[1])
    ids = np.fromfile(path, dtype=np.uint32, offset=8, count=n_file * k).reshape(n_file, k)
    return ids[:nq, :topk]


def read_spmat_labels(path: str, n: int) -> list[list[int]]:
    """Read CSR spmat file -> per-row label lists (only column indices where data != 0)."""
    with open(path, "rb") as f:
        nrow, ncol, nnz = struct.unpack("<qqq", f.read(24))
        n = min(n, nrow)
        # Only read the first n+1 indptr entries
        indptr = np.frombuffer(f.read((n + 1) * 8), dtype=np.int64)
        # Skip remaining indptr
        if n < nrow:
            f.seek((nrow - n) * 8, 1)
        max_idx = int(indptr[n])
        indices = np.fromfile(f, dtype=np.int32, count=max_idx)
        data = np.fromfile(f, dtype=np.float32, count=max_idx)

    labels = []
    for i in range(n):
        start, end = int(indptr[i]), int(indptr[i + 1])
        row_labels = indices[start:end][data[start:end] != 0.0].tolist()
        labels.append(row_labels)
    return labels


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client(uri: str):
    """Create a pymilvus MilvusClient for server benchmarks."""
    from pymilvus import MilvusClient, DataType
    return MilvusClient(uri=uri), DataType


def percentile(xs, p):
    return float(np.percentile(np.asarray(xs), p))


def collection_ready(client, name: str, expected_count: int) -> bool:
    try:
        if not client.has_collection(name):
            return False
    except Exception:
        return False
    try:
        n = int(client.count(name))
    except Exception:
        n = int(client.get_collection_stats(name)["row_count"])
    return n == expected_count


def add_vector_index(index_params, uri: str):
    if uri.endswith(".db"):
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
    else:
        index_params.add_index(field_name="vector", index_type="HNSW", metric_type="L2",
                               params={"M": 16, "efConstruction": 200})


# ---------------------------------------------------------------------------
# Unfiltered benchmark
# ---------------------------------------------------------------------------

def bench_unfiltered(uri: str, label: str, base: np.ndarray, queries: np.ndarray,
                     gt: np.ndarray, ef_list: list[int], nq_lat: int, n_rounds: int,
                     topk: int, threads: int) -> list[dict]:
    """Benchmark unfiltered vector search."""
    dim = base.shape[1]
    nq = len(queries)
    is_grpc = uri.startswith("http://") or uri.startswith("https://")

    print(f"\n===== {label} ({uri}) =====")
    client, DataType = make_client(uri)
    name = "bench_unfiltered"

    if collection_ready(client, name, len(base)):
        print(f"  reuse existing collection: {len(base)} vecs")
    else:
        print("  build collection")
        try:
            client.drop_collection(name)
        except Exception:
            pass

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        client.create_collection(name, schema=schema, metric_type="L2")

        # Insert
        t0 = time.perf_counter()
        batch = 5000
        for s in range(0, len(base), batch):
            e = min(s + batch, len(base))
            rows = [{"id": str(i), "vector": base[i]} for i in range(s, e)]
            client.insert(name, rows)
        insert_s = time.perf_counter() - t0
        print(f"  insert {len(base)} vecs: {insert_s:.1f}s ({len(base)/insert_s:,.0f} vec/s)")
        client.flush(name)

        # create_index + load (real for Milvus, no-op for PipeANN)
        try:
            ip = client.prepare_index_params()
            add_vector_index(ip, uri)
            client.create_index(name, ip)
        except Exception:
            pass
    try:
        client.load_collection(name)
    except Exception:
        pass

    # Set search threads for PipeANN standalone
    if hasattr(client, '_col'):
        try:
            col = client._col(name)
            col._index.omp_set_num_threads(threads)
        except Exception:
            pass

    # Warmup
    client.search(name, data=queries[:100].tolist(), limit=topk,
                  search_params={"params": {"L": 64}})

    # Close parent client before multiprocessing to avoid gRPC fork issues
    if is_grpc:
        client.close()
        del client

    # Search sweep
    results = []
    for ef in ef_list:
        sp = {"params": {"ef": ef, "L": ef}}

        # Throughput: concurrent single-query search
        # For gRPC backends, use multiprocessing to bypass GIL; for
        # in-process clients keep ThreadPoolExecutor (not fork-safe).
        qps_samples = []
        all_res = None
        for rnd in range(n_rounds):
            if is_grpc:
                round_qps, round_res = _mp_unfiltered_throughput(
                    uri, name, queries, topk, sp, threads)
            else:
                def _search_one(i):
                    return i, client.search(name, data=[queries[i].tolist()], limit=topk,
                                            search_params=sp)

                round_res = [None] * nq
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    futs = [pool.submit(_search_one, i) for i in range(nq)]
                    for f in as_completed(futs):
                        i, r = f.result()
                        round_res[i] = r[0]
                total_s = time.perf_counter() - t0
                round_qps = nq / total_s
            # Round 0 runs cold (page cache not yet warm from the small warmup);
            # discard it when we have more than one round so QPS reflects the
            # steady-state hot path, not first-touch SSD misses.
            if rnd > 0 or n_rounds == 1:
                qps_samples.append(round_qps)
            all_res = round_res
        qps = float(np.mean(qps_samples))

        # Recall
        hit = 0
        for i in range(nq):
            got = {int(h["id"]) for h in all_res[i]}
            hit += len(got & set(gt[i].tolist()))
        recall = hit / (nq * topk)

        # Latency: single-query sequential (reopen client for gRPC)
        if is_grpc:
            lat_client, _ = make_client(uri)
        else:
            lat_client = client
        all_lat = []
        for _ in range(n_rounds):
            for i in range(nq_lat):
                t0 = time.perf_counter()
                lat_client.search(name, data=[queries[i].tolist()], limit=topk, search_params=sp)
                all_lat.append((time.perf_counter() - t0) * 1000.0)
        if is_grpc:
            lat_client.close()
            del lat_client

        row = dict(ef=ef, recall=recall, qps=qps,
                   p50=percentile(all_lat, 50), p99=percentile(all_lat, 99),
                   mean=float(np.mean(all_lat)))
        results.append(row)
        print(f"  ef={ef:<4} recall@{topk}={recall:.4f}  QPS={qps:>10,.0f}  "
              f"p50={row['p50']:.3f}ms  p99={row['p99']:.3f}ms")

    return results


# ---------------------------------------------------------------------------
# Multiprocessing throughput helper for filtered search (gRPC backends)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multiprocessing throughput helper for UNFILTERED search (gRPC backends)
# ---------------------------------------------------------------------------

def _mp_unfiltered_worker(uri, collection, config_path, topk, sp,
                          start, end, barrier, result_queue):
    """Worker process for unfiltered search: searches [start, end)."""
    import pickle
    from pymilvus import MilvusClient as _PMC

    with open(config_path, "rb") as f:
        queries = pickle.load(f)

    client = _PMC(uri=uri)
    # warmup
    for i in range(start, min(start + 2, end)):
        client.search(collection, data=[queries[i].tolist()], limit=topk,
                      search_params=sp)

    barrier.wait()

    local_res = {}
    t0 = time.perf_counter()
    for i in range(start, end):
        r = client.search(collection, data=[queries[i].tolist()], limit=topk,
                          search_params=sp)
        local_res[i] = r[0]
    elapsed = time.perf_counter() - t0
    result_queue.put((start, end, elapsed, local_res))


def _mp_unfiltered_throughput(uri, collection, queries, topk, sp, n_procs):
    """Run unfiltered throughput test using multiprocessing. Returns (qps, all_res)."""
    import tempfile, os, pickle

    nq = len(queries)
    tmp_dir = tempfile.mkdtemp(prefix="pipeann_bench_unfilt_")
    config_path = os.path.join(tmp_dir, "queries.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(queries, f)

    barrier = Barrier(n_procs)
    result_queue = Queue()
    chunk = nq // n_procs

    procs = []
    for p in range(n_procs):
        start = p * chunk
        end = start + chunk if p < n_procs - 1 else nq
        proc = Process(target=_mp_unfiltered_worker,
                       args=(uri, collection, config_path, topk, sp,
                             start, end, barrier, result_queue))
        procs.append(proc)

    t0 = time.perf_counter()
    for p in procs:
        p.start()

    all_res = [None] * nq
    # Each worker times only its post-barrier search loop, so its `elapsed`
    # excludes fork/import/connect/warmup startup. All workers release from the
    # barrier together and run concurrently, so the aggregate throughput window
    # is the slowest worker's search loop (max elapsed), NOT the outer wall
    # clock (t0..now) -- the latter folds ~seconds of process startup into the
    # denominator and underreports QPS several-fold.
    worker_elapsed = []
    for _ in procs:
        start, end, elapsed, local_res = result_queue.get()
        worker_elapsed.append(elapsed)
        for i, r in local_res.items():
            all_res[i] = r

    for p in procs:
        p.join()
    window_s = max(worker_elapsed)
    qps = nq / window_s

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return qps, all_res


# ---------------------------------------------------------------------------
# Multiprocessing throughput helper for filtered search (gRPC backends)
# ---------------------------------------------------------------------------

def _mp_search_worker(uri, collection, queries_path, nq, dim,
                      ql1_path, ql2_path, topk, sp, start, end,
                      barrier, result_queue):
    """Worker process: loads own data, searches [start, end), returns results."""
    from pymilvus import MilvusClient as _PMC

    queries = np.fromfile(queries_path, dtype=np.uint8, offset=8,
                          count=nq * dim).reshape(nq, dim).astype(np.float32)

    def _read_spmat(path, n):
        with open(path, "rb") as f:
            nrow, ncol, nnz = struct.unpack("<qqq", f.read(24))
            n = min(n, nrow)
            indptr = np.frombuffer(f.read((n + 1) * 8), dtype=np.int64)
            if n < nrow:
                f.seek((nrow - n) * 8, 1)
            max_idx = int(indptr[n])
            indices = np.fromfile(f, dtype=np.int32, count=max_idx)
            data = np.fromfile(f, dtype=np.float32, count=max_idx)
        labels = []
        for i in range(n):
            s, e = int(indptr[i]), int(indptr[i + 1])
            labels.append(indices[s:e][data[s:e] != 0.0].tolist())
        return labels

    ql1 = _read_spmat(ql1_path, nq)
    ql2 = _read_spmat(ql2_path, nq)

    def _make_filter(i):
        l1 = ql1[i][0] if ql1[i] else 0
        l2 = ql2[i][0] if ql2[i] else 0
        return f"array_contains_all(tags, [{l1}, {l2}])"

    client = _PMC(uri=uri)
    # warmup
    for i in range(start, min(start + 2, end)):
        client.search(collection, data=[queries[i].tolist()], limit=topk,
                      filter=_make_filter(i), search_params=sp)

    barrier.wait()

    local_res = {}
    t0 = time.perf_counter()
    for i in range(start, end):
        r = client.search(collection, data=[queries[i].tolist()], limit=topk,
                          filter=_make_filter(i), search_params=sp)
        local_res[i] = r[0]
    elapsed = time.perf_counter() - t0
    result_queue.put((start, end, elapsed, local_res))


def _mp_filtered_throughput(uri, collection, queries, query_labels_1, query_labels_2,
                            topk, sp, n_procs):
    """Run filtered throughput test using multiprocessing. Returns (qps, all_res)."""
    nq = len(queries)
    # We need file paths for workers to reload data independently.
    # Stash queries + labels to temp files if they came from arrays.
    import tempfile, os, pickle

    tmp_dir = tempfile.mkdtemp(prefix="pipeann_bench_")
    queries_path = os.path.join(tmp_dir, "queries.bin")
    ql1_path = os.path.join(tmp_dir, "ql1.pkl")
    ql2_path = os.path.join(tmp_dir, "ql2.pkl")

    # Write queries as raw .u8bin-like (header + uint8 data, but these are float32)
    # Simpler: just pickle everything since workers will load from original files anyway.
    # Actually, let's pass original file paths via a config pickle.
    config_path = os.path.join(tmp_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump({
            "queries": queries,
            "ql1": query_labels_1,
            "ql2": query_labels_2,
        }, f)

    barrier = Barrier(n_procs)
    result_queue = Queue()
    chunk = nq // n_procs

    procs = []
    for p in range(n_procs):
        start = p * chunk
        end = start + chunk if p < n_procs - 1 else nq
        proc = Process(target=_mp_search_worker_simple,
                       args=(uri, collection, config_path, topk, sp,
                             start, end, barrier, result_queue))
        procs.append(proc)

    t0 = time.perf_counter()
    for p in procs:
        p.start()

    all_res = [None] * nq
    # Use the slowest worker's barrier-synced search window, not the outer wall
    # clock; the latter would include per-process fork/import/connect/warmup.
    worker_elapsed = []
    for _ in procs:
        start, end, elapsed, local_res = result_queue.get()
        worker_elapsed.append(elapsed)
        for i, r in local_res.items():
            all_res[i] = r

    for p in procs:
        p.join()
    window_s = max(worker_elapsed)
    qps = nq / window_s

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return qps, all_res


def _mp_search_worker_simple(uri, collection, config_path, topk, sp,
                             start, end, barrier, result_queue):
    """Simpler worker: loads config from pickle."""
    import pickle
    from pymilvus import MilvusClient as _PMC

    with open(config_path, "rb") as f:
        config = pickle.load(f)
    queries = config["queries"]
    ql1 = config["ql1"]
    ql2 = config["ql2"]

    def _make_filter(i):
        l1 = ql1[i][0] if ql1[i] else 0
        l2 = ql2[i][0] if ql2[i] else 0
        return f"array_contains_all(tags, [{l1}, {l2}])"

    client = _PMC(uri=uri)
    # warmup
    for i in range(start, min(start + 2, end)):
        client.search(collection, data=[queries[i].tolist()], limit=topk,
                      filter=_make_filter(i), search_params=sp)

    barrier.wait()

    local_res = {}
    t0 = time.perf_counter()
    for i in range(start, end):
        r = client.search(collection, data=[queries[i].tolist()], limit=topk,
                          filter=_make_filter(i), search_params=sp)
        local_res[i] = r[0]
    elapsed = time.perf_counter() - t0
    result_queue.put((start, end, elapsed, local_res))


# ---------------------------------------------------------------------------
# Filtered benchmark (YFCC10M style: AND of label conditions)
# ---------------------------------------------------------------------------

def bench_filtered(uri: str, label: str, queries: np.ndarray,
                   gt: np.ndarray, query_labels_1: list[list[int]],
                   query_labels_2: list[list[int]],
                   ef_list: list[int], nq_lat: int, n_rounds: int,
                   topk: int, threads: int, nb: int,
                   load_build_data=None) -> list[dict]:
    """Benchmark filtered vector search (AND of two label conditions).

    Collection lifecycle is self-contained: if the collection already holds `nb`
    rows it is reused as-is (no rebuild); otherwise it is built. The base vectors
    and per-row labels needed for a build are pulled lazily from `load_build_data`
    (a callable returning `(base, base_labels)`) so the reuse path never pays the
    multi-GB load. A build with no `load_build_data` is an error.
    """
    dim = queries.shape[1]
    nq = len(queries)
    is_grpc = uri.startswith("http://") or uri.startswith("https://")

    print(f"\n===== {label} ({uri}) [FILTERED] =====")
    client, DataType = make_client(uri)
    name = "bench_filtered"

    if collection_ready(client, name, nb):
        print(f"  reuse existing collection: {nb} vecs")
    else:
        if load_build_data is None:
            raise RuntimeError(
                f"collection '{name}' ({nb} rows) not present at {uri} and no "
                f"load_build_data provided to build it")
        print("  build collection")
        base, base_labels = load_build_data()
        try:
            client.drop_collection(name)
        except Exception:
            pass

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("tags", DataType.ARRAY, element_type=DataType.INT64, max_capacity=4096)
        client.create_collection(name, schema=schema, metric_type="L2")

        # Insert with labels
        t0 = time.perf_counter()
        batch = 5000
        for s in range(0, nb, batch):
            e = min(s + batch, nb)
            rows = [{"id": str(i), "vector": base[i],
                     "tags": base_labels[i]}
                    for i in range(s, e)]
            client.insert(name, rows)
            if (s // batch) % 200 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    insert progress: {e}/{nb} ({elapsed:.1f}s)")
        insert_s = time.perf_counter() - t0
        print(f"  insert {nb} vecs: {insert_s:.1f}s ({nb/insert_s:,.0f} vec/s)")
        client.flush(name)

        # create_index + load
        try:
            ip = client.prepare_index_params()
            add_vector_index(ip, uri)
            client.create_index(name, ip)
        except Exception:
            pass
    try:
        client.load_collection(name)
    except Exception:
        pass

    # Set search threads for PipeANN standalone
    if hasattr(client, '_col'):
        try:
            col = client._col(name)
            col._index.omp_set_num_threads(threads)
        except Exception:
            pass

    # Build per-query filter expressions
    # Filter: tags contains ALL of [query_labels_1[i], query_labels_2[i]]
    def make_filter(i):
        l1 = query_labels_1[i][0] if query_labels_1[i] else 0
        l2 = query_labels_2[i][0] if query_labels_2[i] else 0
        return f"array_contains_all(tags, [{l1}, {l2}])"

    # Warmup
    for i in range(min(50, nq)):
        client.search(name, data=[queries[i].tolist()], limit=topk,
                      filter=make_filter(i), search_params={"params": {"L": 64}})

    # Close parent client before multiprocessing to avoid gRPC fork issues
    if is_grpc:
        client.close()
        del client

    # Search sweep
    results = []
    # Concurrent warmup: the single-threaded 50-query warmup above barely dents
    # a multi-GB on-disk index, so the first ef in the sweep would otherwise run
    # cold and report artificially low QPS (non-monotonic vs higher ef). Run one
    # full concurrent pass at the lowest ef and discard it to heat the cache.
    if len(ef_list) > 0:
        warm_sp = {"params": {"ef": ef_list[0], "L": ef_list[0]}}
        if is_grpc:
            _mp_filtered_throughput(uri, name, queries, query_labels_1,
                                    query_labels_2, topk, warm_sp, threads)
        else:
            def _warm_one(i):
                return client.search(name, data=[queries[i].tolist()], limit=topk,
                                     filter=make_filter(i), search_params=warm_sp)
            with ThreadPoolExecutor(max_workers=threads) as pool:
                list(pool.map(_warm_one, range(nq)))
    for ef in ef_list:
        sp = {"params": {"ef": ef, "L": ef}}

        # Throughput: concurrent queries via multiprocessing (bypasses GIL
        # limitations in grpcio/pymilvus).  For in-process PipeANN the local
        # client is not fork-safe, so we fall back to ThreadPoolExecutor.
        if is_grpc:
            qps, all_res = _mp_filtered_throughput(
                uri, name, queries, query_labels_1, query_labels_2,
                topk, sp, threads)
        else:
            def _search_one(i):
                return i, client.search(name, data=[queries[i].tolist()], limit=topk,
                                        filter=make_filter(i), search_params=sp)

            t0 = time.perf_counter()
            all_res = [None] * nq
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futs = [pool.submit(_search_one, i) for i in range(nq)]
                for f in as_completed(futs):
                    i, r = f.result()
                    all_res[i] = r[0]
            total_s = time.perf_counter() - t0
            qps = nq / total_s

        # Recall
        hit = 0
        for i in range(nq):
            got = {int(h["id"]) for h in all_res[i]}
            hit += len(got & set(gt[i].tolist()))
        recall = hit / (nq * topk)

        # Latency: single-query sequential (reopen client for gRPC)
        if is_grpc:
            lat_client, _ = make_client(uri)
        else:
            lat_client = client
        all_lat = []
        for _ in range(n_rounds):
            for i in range(nq_lat):
                t0 = time.perf_counter()
                lat_client.search(name, data=[queries[i].tolist()], limit=topk,
                              filter=make_filter(i), search_params=sp)
                all_lat.append((time.perf_counter() - t0) * 1000.0)
        if is_grpc:
            lat_client.close()
            del lat_client

        row = dict(ef=ef, recall=recall, qps=qps,
                   p50=percentile(all_lat, 50), p99=percentile(all_lat, 99),
                   mean=float(np.mean(all_lat)))
        results.append(row)
        print(f"  ef={ef:<4} recall@{topk}={recall:.4f}  QPS={qps:>8,.0f}  "
              f"p50={row['p50']:.3f}ms  p99={row['p99']:.3f}ms")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milvus", default="http://127.0.0.1:19530")
    ap.add_argument("--pipeann-grpc", default="http://127.0.0.1:19540")

    # SIFT1M / BigANN
    ap.add_argument("--base", default="/mnt/nvme/data/bigann/bigann_1M.bbin")
    ap.add_argument("--query", default="/mnt/nvme/data/bigann/bigann_query.bbin")
    ap.add_argument("--gt", default="/mnt/nvme/data/bigann/1M_gt.bin")

    # YFCC10M filtered
    ap.add_argument("--yfcc-base", default="/mnt/nvme/data/yfcc10M-filtered/base.10M.u8bin")
    ap.add_argument("--yfcc-query", default="/mnt/nvme/data/yfcc10M-filtered/query.public.100K.u8bin")
    ap.add_argument("--yfcc-gt", default="/mnt/nvme/data/yfcc10M-filtered/GT.public.ibin")
    ap.add_argument("--yfcc-base-meta", default="/mnt/nvme/data/yfcc10M-filtered/base.metadata.10M.spmat")
    ap.add_argument("--yfcc-query-split1", default="/mnt/nvme/data/yfcc10M-filtered/query.metadata.public.100K.split1.spmat")
    ap.add_argument("--yfcc-query-split2", default="/mnt/nvme/data/yfcc10M-filtered/query.metadata.public.100K.split2.spmat")

    ap.add_argument("--nb", type=int, default=1000000)
    ap.add_argument("--yfcc-nb", type=int, default=10000000)
    ap.add_argument("--nq", type=int, default=10000)
    ap.add_argument("--yfcc-nq", type=int, default=10000)
    ap.add_argument("--nq-lat", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--ef", default="32,64,128,256")
    ap.add_argument("--skip-milvus", action="store_true")
    ap.add_argument("--skip-grpc", action="store_true")
    ap.add_argument("--skip-sift", action="store_true")
    ap.add_argument("--skip-yfcc", action="store_true")
    ap.add_argument("--out", default="/tmp/bench_results.json")
    args = ap.parse_args()

    ef_list = [int(x) for x in args.ef.split(",")]
    out = {}

    # --- SIFT1M unfiltered ---
    if not args.skip_sift:
        print("\n" + "=" * 60)
        print("SIFT1M (BigANN 1M, 128-dim, unfiltered)")
        print("=" * 60)
        base = read_bbin(args.base, args.nb)
        queries = read_bbin(args.query, args.nq)
        gt = read_gt(args.gt, len(queries), args.topk)
        print(f"base={base.shape} queries={queries.shape} gt={gt.shape}")

        if not args.skip_grpc:
            out["sift1m_pipeann_grpc"] = bench_unfiltered(
                args.pipeann_grpc, "PipeANN gRPC",
                base, queries, gt, ef_list, args.nq_lat, args.rounds, args.topk, args.threads)
        if not args.skip_milvus:
            out["sift1m_milvus"] = bench_unfiltered(
                args.milvus, "Milvus gRPC",
                base, queries, gt, ef_list, args.nq_lat, args.rounds, args.topk, args.threads)
        del base, queries, gt

    # --- YFCC10M filtered ---
    if not args.skip_yfcc:
        print("\n" + "=" * 60)
        print("YFCC10M (10M, 192-dim, filtered AND)")
        print("=" * 60)
        queries = read_bbin(args.yfcc_query, args.yfcc_nq)
        gt = read_gt(args.yfcc_gt, args.yfcc_nq, args.topk)
        print(f"base=({args.yfcc_nb}, {queries.shape[1]}) queries={queries.shape} gt={gt.shape}")

        print("loading query metadata splits...")
        ql1 = read_spmat_labels(args.yfcc_query_split1, args.yfcc_nq)
        ql2 = read_spmat_labels(args.yfcc_query_split2, args.yfcc_nq)

        # Base vectors + per-row labels are only needed to BUILD a collection.
        # bench_filtered calls this lazily and only when the collection is
        # absent, so reusing an existing collection never pays the ~2 GB load.
        def load_yfcc_build_data():
            print("collection absent -> loading YFCC base vectors + labels for insert...")
            return (read_bbin(args.yfcc_base, args.yfcc_nb),
                    read_spmat_labels(args.yfcc_base_meta, args.yfcc_nb))

        if not args.skip_grpc:
            out["yfcc10m_pipeann_grpc"] = bench_filtered(
                args.pipeann_grpc, "PipeANN gRPC",
                queries, gt, ql1, ql2,
                ef_list, args.nq_lat, args.rounds, args.topk, args.threads,
                nb=args.yfcc_nb, load_build_data=load_yfcc_build_data)
        if not args.skip_milvus:
            out["yfcc10m_milvus"] = bench_filtered(
                args.milvus, "Milvus gRPC",
                queries, gt, ql1, ql2,
                ef_list, args.nq_lat, args.rounds, args.topk, args.threads,
                nb=args.yfcc_nb, load_build_data=load_yfcc_build_data)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
