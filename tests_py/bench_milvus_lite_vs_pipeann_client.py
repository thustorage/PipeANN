#!/usr/bin/env python3
"""Benchmark: Milvus Lite vs PipeANN in-process MilvusClient."""
from __future__ import annotations

import argparse
import json

import bench_milvus_vs_pipeann as common


def make_client(uri: str):
    if uri.endswith(".db"):
        from pymilvus import MilvusClient, DataType
        return MilvusClient(uri=uri), DataType
    from pipeann import MilvusClient, DataType
    return MilvusClient(uri=uri), DataType


common.make_client = make_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milvus-lite", default="/mnt/nvme4/milvus_lite_bench.db")
    ap.add_argument("--pipeann-local", default="/mnt/nvme4/bench")

    ap.add_argument("--base", default="/mnt/nvme/data/bigann/bigann_1M.bbin")
    ap.add_argument("--query", default="/mnt/nvme/data/bigann/bigann_query.bbin")
    ap.add_argument("--gt", default="/mnt/nvme/data/bigann/1M_gt.bin")

    ap.add_argument("--yfcc-query", default="/mnt/nvme/data/yfcc10M-filtered/query.public.100K.u8bin")
    ap.add_argument("--yfcc-gt", default="/mnt/nvme/data/yfcc10M-filtered/GT.public.ibin")
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
    ap.add_argument("--skip-lite", action="store_true")
    ap.add_argument("--skip-pipeann", action="store_true")
    ap.add_argument("--skip-sift", action="store_true")
    ap.add_argument("--skip-yfcc", action="store_true")
    ap.add_argument("--out", default="/tmp/bench_lite_vs_pipeann_client.json")
    args = ap.parse_args()

    ef_list = [int(x) for x in args.ef.split(",")]
    out = {}

    if not args.skip_sift:
        print("\n" + "=" * 60)
        print("SIFT1M (BigANN 1M, 128-dim, unfiltered)")
        print("=" * 60)
        base = common.read_bbin(args.base, args.nb)
        queries = common.read_bbin(args.query, args.nq)
        gt = common.read_gt(args.gt, len(queries), args.topk)
        print(f"base={base.shape} queries={queries.shape} gt={gt.shape}")

        if not args.skip_pipeann:
            out["sift1m_pipeann_client"] = common.bench_unfiltered(
                args.pipeann_local, "PipeANN MilvusClient",
                base, queries, gt, ef_list, args.nq_lat, args.rounds, args.topk, args.threads)
        if not args.skip_lite:
            out["sift1m_milvus_lite"] = common.bench_unfiltered(
                args.milvus_lite, "Milvus Lite",
                base, queries, gt, ef_list, args.nq_lat, args.rounds, args.topk, args.threads)

    if not args.skip_yfcc:
        print("\n" + "=" * 60)
        print("YFCC10M (10M, 192-dim, filtered AND)")
        print("=" * 60)
        queries = common.read_bbin(args.yfcc_query, args.yfcc_nq)
        gt = common.read_gt(args.yfcc_gt, args.yfcc_nq, args.topk)
        print(f"base=({args.yfcc_nb}, {queries.shape[1]}) queries={queries.shape} gt={gt.shape}")

        print("loading query metadata splits...")
        ql1 = common.read_spmat_labels(args.yfcc_query_split1, args.yfcc_nq)
        ql2 = common.read_spmat_labels(args.yfcc_query_split2, args.yfcc_nq)

        if not args.skip_pipeann:
            out["yfcc10m_pipeann_client"] = common.bench_filtered(
                args.pipeann_local, "PipeANN MilvusClient",
                None, queries, gt, [], ql1, ql2,
                ef_list, args.nq_lat, args.rounds, args.topk, args.threads,
                expected_count=args.yfcc_nb)
        if not args.skip_lite:
            out["yfcc10m_milvus_lite"] = common.bench_filtered(
                args.milvus_lite, "Milvus Lite",
                None, queries, gt, [], ql1, ql2,
                ef_list, args.nq_lat, args.rounds, args.topk, args.threads,
                expected_count=args.yfcc_nb)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
