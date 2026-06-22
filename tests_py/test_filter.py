"""Filtered-search test for the native IndexPipeANN API.

Builds one disk index with two attribute columns:

  * key 0 — ``label`` (categorical tag)
  * key 1 — ``range`` (numeric price)

and exercises the two filter paths PipeANN exposes, both against the same
numpy ground truth (tag == query_tag AND price in [lo, hi)):

  1. **Native selector composition** — And/Or/Not/Label/Range selectors built
     from C++ attr indexes. This is the production path.
  2. **Python custom selector** — a ``Selector`` subclass implementing the
     callback ABC. Slower (each callback crosses pybind), but lets callers
     express arbitrary predicates.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeann import (
    AndSelector,
    Attributes,
    AttrsVec,
    IndexPipeANN,
    LabelOrSelector,
    Metric,
    NotSelector,
    OrSelector,
    RangeSelector,
    Selector,
)
from utils import bin_write, compute_recall, write_spmat


# --------------------------------------------------------------------------
# Dataset + ground truth
# --------------------------------------------------------------------------
def build_dataset(
    rng: np.random.Generator, n_base: int, n_queries: int, dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    centers = rng.normal(0.0, 4.0, size=(6, dim)).astype(np.float32)
    assignments = rng.integers(0, len(centers), size=n_base)
    vectors = (centers[assignments] + rng.normal(0.0, 0.35, size=(n_base, dim))).astype(np.float32)
    tags = rng.integers(0, 8, size=n_base, dtype=np.uint32)
    prices = rng.integers(0, 1000, size=n_base, dtype=np.uint32)

    query_ids = rng.choice(n_base, size=n_queries, replace=False)
    queries = vectors[query_ids].copy()
    query_ranges: list[tuple[int, int, int]] = []
    for i, base_id in enumerate(query_ids):
        tag = int(tags[base_id])
        upper = min(1000, max(0, int(prices[base_id]) - 80) + 160)
        lower = upper - 160
        query_ranges.append((tag, lower, upper))
        queries[i] += rng.normal(0.0, 0.02, size=dim).astype(np.float32)
    return vectors, tags, prices, queries, query_ranges


def compute_ground_truth(
    vectors: np.ndarray,
    tags: np.ndarray,
    prices: np.ndarray,
    queries: np.ndarray,
    query_ranges: list[tuple[int, int, int]],
    topk: int,
) -> np.ndarray:
    gt = np.empty((len(queries), topk), dtype=np.int32)
    for i, (tag, lower, upper) in enumerate(query_ranges):
        mask = (tags == tag) & (prices >= lower) & (prices < upper)
        ids = np.flatnonzero(mask)
        dists = np.sum((vectors[ids] - queries[i]) ** 2, axis=1)
        gt[i] = ids[np.argsort(dists)[:topk]]
    return gt


# --------------------------------------------------------------------------
# Python custom selector — implements the callback ABC over numpy attrs.
# Predicate: target tag == query tag AND query_lo <= target price < query_hi.
# --------------------------------------------------------------------------
class HybridSelector(Selector):
    def __init__(self, tags: np.ndarray, prices: np.ndarray):
        super().__init__()
        self.tags = np.asarray(tags, dtype=np.uint32)
        self.prices = np.asarray(prices, dtype=np.uint32)
        self.tag_to_ids: dict[int, list[int]] = {}
        for idx, tag in enumerate(self.tags.tolist()):
            self.tag_to_ids.setdefault(int(tag), []).append(idx)

    @staticmethod
    def _parse(query_attrs: Attributes) -> tuple[int, int, int]:
        row = query_attrs.to_dict()
        return int(row[0][0]), int(row[1][0]), int(row[1][1])

    def estimate_selectivity(self, query_attrs: Attributes) -> float:
        tag, lower, upper = self._parse(query_attrs)
        ids = self.tag_to_ids.get(tag, [])
        if not ids:
            return 0.0
        matched = sum(lower <= int(self.prices[idx]) < upper for idx in ids)
        return matched / len(self.tags)

    def estimate_precision(self, query_attrs: Attributes) -> float:
        tag, lower, upper = self._parse(query_attrs)
        ids = self.tag_to_ids.get(tag, [])
        if not ids:
            return 1.0
        bucket_hits = sum(lower <= int(self.prices[idx]) < upper for idx in ids)
        return max(bucket_hits / len(ids), 0.2)

    def estimate_prefilter_reads(self, query_attrs: Attributes) -> int:
        return 1

    def pre_filter(self, query_attrs: Attributes) -> list[int]:
        tag, _, _ = self._parse(query_attrs)
        return self.tag_to_ids.get(tag, [])

    def is_member(self, target_id: int, query_attrs: Attributes, target_attrs: Attributes) -> bool:
        tag, lower, upper = self._parse(query_attrs)
        row = target_attrs.to_dict()
        return row[0][0] == tag and lower <= row[1][0] < upper

    def estimate_infilter_reads(self, query_attrs: Attributes) -> int:
        return 0

    def prepare_in_filter(self, query_attrs: Attributes) -> None:
        _, lower, upper = self._parse(query_attrs)
        self.prepared_ids = set(
            np.flatnonzero((self.prices >= lower) & (self.prices < upper)).astype(np.uint32).tolist()
        )

    def is_member_approx(self, target_id: int, query_attrs: Attributes) -> bool:
        tag, _, _ = self._parse(query_attrs)
        return int(self.tags[target_id]) == tag or target_id in self.prepared_ids


def main() -> None:
    rng = np.random.default_rng(42)
    n_base = int(os.environ.get("PIPEANN_FILTER_N_BASE", "10000"))
    n_queries = int(os.environ.get("PIPEANN_FILTER_N_QUERY", "16"))
    dim = int(os.environ.get("PIPEANN_FILTER_DIM", "32"))
    topk = int(os.environ.get("PIPEANN_FILTER_TOPK", "10"))
    search_L = int(os.environ.get("PIPEANN_FILTER_SEARCH_L", "2000"))
    recall_threshold = float(os.environ.get("PIPEANN_FILTER_RECALL", "0.85"))

    vectors, tags, prices, queries, query_ranges = build_dataset(rng, n_base, n_queries, dim)
    gt = compute_ground_truth(vectors, tags, prices, queries, query_ranges, topk)

    attrs = AttrsVec(attr_types={0: "label", 1: "range"})
    for tag, price in zip(tags, prices, strict=True):
        attrs.append({0: [int(tag)], 1: [int(price)]})

    with tempfile.TemporaryDirectory(prefix="pipeann_filter_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "base.bin"
        index_prefix = tmp / "filter"
        tag_index_path = tmp / "filter.label.0"
        range_index_path = tmp / "filter.range.1"
        tag_query_path = tmp / "tag_query.spmat"
        range_query_path = tmp / "range_query.spmat"

        bin_write(vectors, data_path)
        attrs.save(0, tag_index_path)
        attrs.save(1, range_index_path)

        # Per-query attr rows for the native path, in .spmat form: one tag per
        # row, and a single [lo, hi) interval per row.
        tag_rows = [[tag] for tag, _, _ in query_ranges]
        tag_values = [[1.0] for _ in query_ranges]
        range_rows = [[0, 0] for _ in query_ranges]
        range_values = [[float(lo), float(hi)] for _, lo, hi in query_ranges]
        write_spmat(tag_rows, tag_values, int(tags.max()) + 1, tag_query_path)
        write_spmat(range_rows, range_values, 1, range_query_path)

        idx = IndexPipeANN(data_dim=dim, data_type=np.dtype(np.float32), metric=Metric.L2)
        idx.omp_set_num_threads(4)
        idx.build(
            str(data_path),
            str(index_prefix),
            max_nbrs=32,
            build_L=100,
            PQ_bytes=16,
            memory_use_GB=1,
            attrs=attrs,
        )
        idx.load(str(index_prefix))

        # ---- Path 1: native selector composition ----
        tag_index = idx.load_attr_index_from_file(0, tag_index_path, "label")
        range_index = idx.load_attr_index_from_file(1, range_index_path, "range")
        npoints = idx.npoints()

        native_attrs = AttrsVec()
        native_attrs.load_from_file(0, "label", tag_query_path)
        native_attrs.load_from_file(1, "range", range_query_path)

        # Reduces to (tag == query_tag) AND (price in [lo, hi)); the redundant
        # Or/Not wrappers also exercise the composite constructors.
        native_selector = AndSelector(
            OrSelector(
                LabelOrSelector(key=0, base_key=0, attr_index=tag_index),
                LabelOrSelector(key=0, base_key=0, attr_index=tag_index),
            ),
            RangeSelector(key=1, base_key=1, attr_index=range_index),
            NotSelector(
                NotSelector(LabelOrSelector(key=0, base_key=0, attr_index=tag_index), npoints),
                npoints,
            ),
        )
        native_ids, _ = idx.search(
            queries, topk=topk, L=search_L, selector=native_selector, query_attrs=native_attrs
        )
        native_recall = compute_recall(native_ids, gt)

        # ---- Path 2: Python custom selector ----
        py_attrs = [{0: [tag], 1: [lo, hi]} for tag, lo, hi in query_ranges]
        py_selector = HybridSelector(tags, prices)
        py_ids, _ = idx.search(
            queries, topk=topk, L=search_L, selector=py_selector, query_attrs=py_attrs
        )
        py_recall = compute_recall(py_ids, gt)

    print(f"Dataset: base={n_base}, queries={n_queries}, dim={dim}, topk={topk}")
    print(f"Native selector recall@{topk}: {native_recall:.4f}")
    print(f"Python selector recall@{topk}: {py_recall:.4f}")

    assert native_recall >= recall_threshold, f"native selector recall too low: {native_recall:.4f}"
    assert py_recall >= recall_threshold, f"python selector recall too low: {py_recall:.4f}"
    print("PASS: native and python filter paths both stay above threshold.")


def test_filter_recall() -> None:
    main()


if __name__ == "__main__":
    main()
