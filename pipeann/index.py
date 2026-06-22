from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Tuple
import struct
import numpy as np
from .C import PyIndex as _NativeIndex
from .C import Metric as _Metric
from .filter import AttrsVec, Attributes, NativeAttrIndex, Selector


class Metric:
    """Distance metrics supported by PipeANN."""

    L2 = _Metric.L2
    INNER_PRODUCT = _Metric.INNER_PRODUCT
    COSINE = _Metric.COSINE

    _STR_MAP = {
        "l2": _Metric.L2,
        "inner_product": _Metric.INNER_PRODUCT,
        "cosine": _Metric.COSINE,
    }

    VALID_STRINGS = frozenset(_STR_MAP)

    @classmethod
    def from_str(cls, s: str):
        """Convert a string like ``"l2"`` to the corresponding enum value."""
        try:
            return cls._STR_MAP[s]
        except KeyError:
            raise ValueError(
                f"metric must be one of {set(cls._STR_MAP)}, got {s!r}"
            ) from None


VALID_DATA_TYPES = frozenset({"float32", "int8", "uint8"})


class IndexPipeANN:
    """A Python wrapper for the PipeANN index."""

    def __init__(self, data_dim: int, data_type: str = "float32", metric=Metric.L2, attr_size: int = 0):
        """Create an empty index. ``data_type`` is one of ``VALID_DATA_TYPES``.

        ``attr_size`` reserves this many bytes per on-disk node for serialized
        scalar attributes. ``0`` keeps the old behavior: infer it from build
        attrs when present.
        """
        self._impl = _NativeIndex(int(data_dim), str(np.dtype(data_type)), _Metric(metric), int(attr_size))

    def set_index_prefix(self, index_prefix: str) -> None:
        """Set the default on-disk prefix used by save/load helpers."""
        self._impl.set_index_prefix(index_prefix)

    def omp_set_num_threads(self, num_threads: int) -> None:
        """Set the OpenMP thread count used by search and update paths."""
        self._impl.omp_set_num_threads(num_threads)

    def npoints(self) -> int:
        """Return the current index point count, including inserted points."""
        return int(self._impl.npoints())

    # I/O
    def load(self, index_prefix: str) -> None:
        """Load an existing PipeANN index from disk."""
        self._impl.load(index_prefix)

    def save(self, index_prefix: str) -> bool:
        """Save the current index state to disk."""
        return self._impl.save(index_prefix)

    # Build
    def build(
        self,
        data_path: str,
        index_prefix: str,
        tag_file: str | None = None,
        build_mem_index: bool = False,
        max_nbrs: int = 0,
        build_L: int = 0,
        PQ_bytes: int = 0,
        memory_use_GB: int = 0,
        attrs: AttrsVec | None = None,
        range_dense: int = 0,
        train_query_path: str = "",
        R_ood: int = 0,
        L_ood: int = 1500,
    ) -> None:
        """Build a new disk index from a base vector file.

        Args:
            data_path: Base vector file (.bin: uint32 npts, uint32 dim, then data).
            index_prefix: Output prefix; writes ``{prefix}_disk.index``.
            tag_file: uint32 tag per vector. None → tags ``0..npts-1``.
            build_mem_index: Also build in-memory nav graph.
            max_nbrs: Out-degree ``R = R_base + R_ood`` (normal + OOD edges). 0 → auto.
            build_L: Build candidate list. 0 → ``max_nbrs + 32``.
            PQ_bytes: PQ code size. 0 → auto, clamped to [32, 128].
            memory_use_GB: Build RAM budget. 0 → 75% of system RAM.
            attrs: Per-vector attribute rows; length must equal ``npts``.
            range_dense: Total #densified neighbors (max_nbrs + dense_nbrs) for in-filtering. 0 → auto.
            train_query_path: Training query file for OOD refine. Empty disables refine.
            R_ood: #Neighbors for OOD. Requires ``train_query_path`` and ``R_ood < max_nbrs``. 0 → auto.
            L_ood: Beam width for OOD refine. Default: 1500.
        """
        if attrs is not None:
            with open(data_path, "rb") as reader:
                npts = struct.unpack("<I", reader.read(4))[0]
            if len(attrs) != npts:
                raise ValueError(
                    f"attrs has {len(attrs)} rows, but data file contains {npts} vectors"
                )

        self._impl.build(
            data_path,
            index_prefix,
            tag_file,
            build_mem_index,
            max_nbrs,
            build_L,
            PQ_bytes,
            memory_use_GB,
            None if attrs is None else attrs._impl,
            range_dense,
            train_query_path,
            R_ood,
            L_ood,
        )

    # Update & Query
    def add(
        self,
        vectors: np.ndarray,
        tags: np.ndarray,
        attrs: AttrsVec | Sequence[Attributes | Mapping[int, Sequence[int]]] | None = None,
    ) -> None:
        """Insert vectors with explicit tags and optional per-vector attributes.

        Args:
            vectors: 2D array ``(n, dim)`` of the index dtype.
            tags: 1D uint32 array of length ``n``; must be unique across the index.
            attrs: Per-vector attribute rows, length ``n``. None = no attributes.
        """
        assert vectors.ndim == 2, "vectors must be 2D"
        assert tags.ndim == 1 and len(tags) == len(
            vectors
        ), "tags must be 1D with same length as vectors"
        attrs_impl = None
        if attrs is not None:
            if not isinstance(attrs, AttrsVec):
                attrs = AttrsVec(attrs)
            assert len(attrs) == len(vectors), "attrs must have same length as vectors"
            attrs_impl = attrs._impl
        self._impl.add(
            np.ascontiguousarray(vectors),
            np.ascontiguousarray(tags.astype(np.uint32)),
            attrs_impl,
        )

    def remove(self, tags: np.ndarray) -> None:
        """Remove vectors by tag (1D uint32 array; unknown tags are ignored)."""
        assert tags.ndim == 1, "tags must be 1D"
        self._impl.remove(np.ascontiguousarray(tags.astype(np.uint32)))

    def search(
        self,
        queries: np.ndarray,
        topk: int,
        L: int,
        selector: Selector | None = None,
        query_attrs: (
            AttrsVec | Sequence[Attributes | Mapping[int, Sequence[int]]] | None
        ) = None,
        range: float = float("inf"),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-``topk`` neighbors for each query.

        Args:
            queries: 2D array ``(nq, dim)`` of the index dtype.
            topk: Number of neighbors to return per query.
            L: Search beam width (``L >= topk``; larger → higher recall).
            selector: Optional attribute selector to restrict candidates.
            query_attrs: Per-query attributes consumed by ``selector``.
            range: If finite, do a range search: return only neighbors whose
                distance (in user-facing metric) is ``<= range``. Unused
                slots up to ``topk`` are padded with ``UINT32_MAX`` / ``inf``.
                Defaults to ``+inf`` (plain top-k).

        Returns:
            ``(ids, dists)`` arrays of shape ``(nq, topk)``.
        """
        assert queries.ndim == 2, "queries must be 2D"
        if query_attrs is None:
            query_attrs = AttrsVec()
        elif not isinstance(query_attrs, AttrsVec):
            query_attrs = AttrsVec(query_attrs)
        query_attrs_impl = query_attrs._impl
        ids, dists = self._impl.search(
            np.ascontiguousarray(queries),
            int(topk),
            int(L),
            selector,
            query_attrs_impl,
            float(range),
        )
        return np.asarray(ids), np.asarray(dists)

    def load_attr_index_from_file(
        self, key: int, filename: str | Path, attr_type: str
    ) -> NativeAttrIndex:
        """Load one native attr index using the current index's vector count."""
        return self._impl.load_attr_index_from_file(int(key), str(filename), attr_type)

    def create_attr_index(
        self, key: int, filename: str | Path, attr_type: str
    ) -> NativeAttrIndex:
        """Create and register an empty scalar attr index for streaming inserts."""
        return self._impl.create_attr_index(int(key), str(filename), attr_type)

    def compile_filter(
        self, filter_json: str, schema: dict[str, tuple[int, str, NativeAttrIndex]]
    ) -> tuple[Selector, Attributes, dict[str, int], dict[str, str]]:
        """Compile a SQL-like filter string into a (Selector, Attributes,
        slot_map, var_field_type) tuple.

        ``schema`` maps field name → (key, attr_type, attr_index). Caller owns
        the returned Selector (managed by Python's GC via pybind).

        ``slot_map`` maps ``$$var`` placeholder names to slot indices in
        ``Attributes``; ``var_field_type`` maps each var to its field type
        (``"label"`` / ``"range"`` / ``"string"``). Both are empty when the
        filter uses no placeholders, in which case ``Attributes`` is fully
        populated and can be passed directly to search.
        """
        selector, attrs_impl, slot_map, var_field_type = self._impl.compile_filter(filter_json, schema)
        return selector, Attributes(attrs_impl), dict(slot_map), dict(var_field_type)

    def load_filter_from_json(
        self, config_path: str | Path
    ) -> tuple[Selector, AttrsVec]:
        """Load a unified filter config and return ``(selector, query_attrs)``.

        Config layout::

            {
              "attr_indexes": [{"name": ..., "key": ..., "type": ..., "file": ...}, ...],
              "filter": "<SQL-like filter string, may use $$var placeholders>",
              "bindings": {var: <path to .spmat with one row per query>}
            }

        Each declared AttrIndex is loaded into the underlying disk index (kept
        alive for as long as this index lives). ``query_attrs`` has one row per
        binding query; pass it directly to :meth:`search`.
        """
        selector, native_attrs_vec = self._impl.load_filter_from_json(str(config_path))
        return selector, AttrsVec(native_attrs_vec)

    def filter_only(
        self,
        selector: Selector,
        query_attrs: Attributes,
        limit: int = 0,
    ) -> np.ndarray:
        """Run pre_filter only; returns a 1-D ``np.uint32`` array of tags.

        ``limit == 0`` means no truncation.
        """
        impl = query_attrs._impl if hasattr(query_attrs, "_impl") else query_attrs
        return self._impl.filter_only(selector, impl, int(limit))

    def __repr__(self) -> str:  # pragma: no cover
        return repr(self._impl)
