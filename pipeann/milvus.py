"""Milvus-compatible Python API backed by the native PipeANN CollectionStore.

A local MilvusClient facade implementing collection CRUD, insert/upsert, vector
search with SQL scalar filters, query, get, delete, and flush. All data-model
logic (schema, scalar codec, SQL filter compile, output-field projection) lives
in the C++ CollectionStore engine — this module is a thin row<->column shim that
also serves the pymilvus schema/index builder API.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Iterable, Optional

import numpy as np

from .C import CollectionStore


class DataType(Enum):
    INT64 = "INT64"
    VARCHAR = "VARCHAR"
    BOOL = "BOOL"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    ARRAY = "ARRAY"


# Map a pymilvus/DataType-ish dtype string to a native CollectionStore field kind.
def _kind(dtype: Any) -> str:
    s = dtype.value if isinstance(dtype, Enum) else str(dtype)
    s = s.upper()
    if "VECTOR" in s:
        return "float_vector"
    if "ARRAY" in s:
        return "array_int"
    if "VARCHAR" in s or "STRING" in s:
        return "varchar"
    if "BOOL" in s:
        return "bool"
    if "FLOAT" in s or "DOUBLE" in s:
        return "float"
    if "INT" in s:
        return "int64"
    return "json"


def _metric(m: str) -> str:
    m = (m or "").upper()
    if m in ("IP", "INNER_PRODUCT"):
        return "inner_product"
    if m == "L2":
        return "l2"
    return "cosine"


class MilvusClient:
    def __init__(self, uri: Optional[str] = None, **_: Any) -> None:
        if uri is None or uri in ("", "./", "default"):
            path = os.path.abspath("./data")
        elif uri.startswith("file://"):
            path = os.path.abspath(uri[len("file://"):])
        elif uri.startswith(("http://", "https://")):
            path = os.path.abspath("./data")
        else:
            path = os.path.abspath(uri)
        os.makedirs(path, exist_ok=True)
        self._store = CollectionStore(path, 0)
        # Per-collection bookkeeping the engine doesn't track: auto_id + the
        # next auto id to hand out.
        self._auto_id: dict[str, bool] = {}
        self._next_auto: dict[str, int] = {}

    # ---- Collection CRUD ----

    def create_collection(
        self, collection_name: str, dimension: Optional[int] = None,
        primary_field_name: str = "id", vector_field_name: str = "vector",
        metric_type: str = "COSINE", schema: Any = None, **kw: Any,
    ) -> None:
        if self._store.has_collection(collection_name):
            return
        fields = _fields_from_schema(schema, primary_field_name, vector_field_name, dimension or kw.get("dim"))
        self._store.create_collection(collection_name, fields, _metric(metric_type))
        auto = bool(kw.get("auto_id") or (schema.get("auto_id") if isinstance(schema, dict)
                                          else getattr(schema, "auto_id", False)))
        self._auto_id[collection_name] = auto
        self._next_auto[collection_name] = 0

    def has_collection(self, collection_name: str, **_: Any) -> bool:
        return self._store.has_collection(collection_name)

    def list_collections(self, **_: Any) -> list[str]:
        return self._store.list_collections()

    def drop_collection(self, collection_name: str, **_: Any) -> None:
        self._store.drop_collection(collection_name)

    def describe_collection(self, collection_name: str, **_: Any) -> dict[str, Any]:
        meta = self._store.describe(collection_name)
        if meta is None:
            raise RuntimeError(f"Collection {collection_name!r} does not exist")
        return {
            "collection_name": collection_name,
            "num_entities": self._store.row_count(collection_name),
            "dimension": meta["dim"],
            "primary_field": meta["primary_field"],
            "vector_field": meta["vector_field"],
        }

    def count(self, collection_name: str, filter: str = "", **_: Any) -> int:
        self._ensure_built(collection_name)
        return int(self._store.count(collection_name, filter))

    def load_collection(self, collection_name: str, **_: Any) -> None:
        self._store.build_index(collection_name)

    def release_collection(self, collection_name: str, **_: Any) -> None:
        pass

    # ---- pymilvus-compatible schema / index / load stubs ----

    def create_schema(self, **kw: Any) -> "_Schema":
        return _Schema(auto_id=bool(kw.get("auto_id", False)))

    def prepare_index_params(self, **_: Any) -> "_IndexParams":
        return _IndexParams()

    def create_index(self, collection_name: str, index_params: Any = None, **_: Any) -> None:
        # PipeANN builds one native index regardless of index_type. We translate
        # the HNSW build knobs: M -> R (2*M, matching base-layer degree),
        # efConstruction -> L. Anything else is accepted and ignored.
        build_r, build_l = 0, 0
        indexes = getattr(index_params, "indexes", None) or []
        for idx in indexes:
            params = idx.get("params", idx) if isinstance(idx, dict) else {}
            m = params.get("M")
            ef = params.get("efConstruction")
            if m:
                build_r = 2 * int(m)
            if ef:
                build_l = int(ef)
        self._store.build_index(collection_name, build_R=build_r, build_L=build_l)

    # ---- Entity write/read ----

    def insert(self, collection_name: str, data: list | dict, **_: Any) -> dict[str, Any]:
        return self._write(collection_name, data, is_upsert=False)

    def upsert(self, collection_name: str, data: list | dict, **_: Any) -> dict[str, Any]:
        return self._write(collection_name, data, is_upsert=True)

    def _write(self, collection_name: str, data: list | dict, is_upsert: bool) -> dict[str, Any]:
        meta = self._meta(collection_name)
        rows = _to_rows(data)
        pkf, vf = meta["primary_field"], meta["vector_field"]
        if self._auto_id.get(collection_name):
            nxt = self._next_auto.get(collection_name, 0)
            for row in rows:
                row.setdefault(pkf, str(nxt))
                nxt += 1
            self._next_auto[collection_name] = nxt

        # Field kind per name (from the resolved schema).
        kinds = {name: kind for (name, kind, _is_pk, _dim) in meta["fields"]}
        scalar_names = [name for (name, kind, is_pk, _d) in meta["fields"]
                        if not is_pk and kind != "float_vector"]

        n = len(rows)
        pks = [str(r.get(pkf)) for r in rows]
        vectors = np.ascontiguousarray(
            np.array([_vec(r, vf) for r in rows], dtype=np.float32).reshape(n, meta["dim"]))
        scalars: dict[str, list] = {}
        for name in scalar_names:
            if any(name in r for r in rows):
                scalars[name] = [r.get(name) for r in rows]
        res = self._store.insert(collection_name, n, pks, vectors, scalars,
                                 {k: kinds[k] for k in scalars}, is_upsert)
        key = "upsert_count" if is_upsert else "insert_count"
        return {key: res["count"], "ids": res["ids"]}

    def search(
        self, collection_name: str, data: Iterable, filter: str = "",
        limit: int = 10, output_fields: Optional[list[str]] = None,
        search_params: Optional[dict] = None, **_: Any,
    ) -> list[list[dict[str, Any]]]:
        meta = self._meta(collection_name)
        self._ensure_built(collection_name)
        queries = np.ascontiguousarray(np.asarray(list(data), dtype=np.float32).reshape(-1, meta["dim"]))
        nq = queries.shape[0]
        L = _search_L(limit, search_params)
        ofs = _normalize_fields(output_fields, meta)
        res = self._store.search(collection_name, nq, limit, max(L, limit), queries, filter or "", ofs)
        return _split_search(res, nq, meta)

    def query(
        self, collection_name: str, filter: str = "",
        output_fields: Optional[list[str]] = None, limit: Optional[int] = None, **_: Any,
    ) -> list[dict[str, Any]]:
        meta = self._meta(collection_name)
        self._ensure_built(collection_name)
        ofs = _normalize_fields(output_fields, meta)
        res = self._store.query(collection_name, filter or "", int(limit or 0), ofs)
        return _rows_from_columns(res, meta)

    def get(self, collection_name: str, ids: list, output_fields: Optional[list[str]] = None, **_: Any) -> list[dict]:
        meta = self._meta(collection_name)
        pkf = meta["primary_field"]
        id_list = ", ".join(f'"{x}"' for x in ids)
        return self.query(collection_name, filter=f"{pkf} in [{id_list}]", output_fields=output_fields)

    def delete(self, collection_name: str, ids: Optional[list] = None, filter: str = "", **_: Any) -> dict:
        pks = [str(x) for x in (ids or [])]
        deleted = self._store.remove(collection_name, pks, filter or "")
        return {"delete_count": int(deleted)}

    def flush(self, collection_name: Optional[str] = None, **_: Any) -> None:
        self._store.flush(collection_name or "")

    # ---- internals ----

    def _meta(self, name: str) -> dict:
        meta = self._store.describe(name)
        if meta is None:
            raise RuntimeError(f"Collection {name!r} does not exist")
        return meta

    def _ensure_built(self, name: str) -> None:
        # The in-process facade builds the index lazily on first read, preserving
        # the old build-on-insert ergonomics (the gRPC path builds explicitly via
        # load_collection/create_index). Idempotent in the engine.
        if not self._store.index_built(name):
            self._store.build_index(name)

    # Alias used by older callers.
    _collection = _meta


# ---- Row / column shaping helpers ----

def _to_rows(data: list | dict) -> list[dict]:
    if isinstance(data, dict):
        keys = list(data)
        n = len(data[keys[0]]) if keys else 0
        return [{k: data[k][i] for k in keys} for i in range(n)]
    return list(data or [])


def _vec(row: dict, vf: str):
    v = row.get(vf)
    if v is None:
        v = row.get("vector")
    if v is None:
        v = row.get("embedding")
    return np.asarray(v, dtype=np.float32)


def _search_L(limit: int, params: Optional[dict]) -> int:
    p = (params or {}).get("params", params or {})
    return int(p.get("L") or p.get("ef") or max(50, limit))


def _normalize_fields(output_fields: Optional[list[str]], meta: dict) -> list[str]:
    """Resolve output_fields, expanding ["*"]/None to all declared fields."""
    if not output_fields or output_fields == ["*"]:
        names = [name for (name, kind, _pk, _d) in meta["fields"] if kind != "float_vector"]
        if meta["primary_field"] not in names:
            names = [meta["primary_field"]] + names
        return names
    return list(output_fields)


def _rows_from_columns(res: dict, meta: dict) -> list[dict]:
    """Turn the engine's columnar QueryResult into pymilvus row dicts."""
    cols = res["columns"]
    ids = res["ids"]
    n = len(ids)
    rows = []
    for i in range(n):
        ent = {}
        for name, values in cols.items():
            ent[name] = values[i]
        ent.setdefault("id", ids[i])
        rows.append(ent)
    return rows


def _split_search(res: dict, nq: int, meta: dict) -> list[list[dict]]:
    """Slice the flat search result back into per-query hit lists."""
    cols = res["columns"]
    ids = res["ids"]
    scores = res["scores"]
    topks = res["topks"] or [len(ids)]
    out = []
    pos = 0
    for q in range(nq):
        k = topks[q] if q < len(topks) else 0
        hits = []
        for j in range(pos, pos + k):
            ent = {name: values[j] for name, values in cols.items()}
            hits.append({"id": ids[j], "distance": float(scores[j]), "entity": ent})
        out.append(hits)
        pos += k
    return out


def _fields_from_schema(schema: Any, primary: str, vector: str, dim: Optional[int]) -> list:
    """Build the native field list [(name, kind, is_primary, dim), ...]."""
    fields = []
    seen_primary = False
    seen_vector = False
    raw = (schema.get("fields") if isinstance(schema, dict) else getattr(schema, "fields", None)) or [] if schema else []
    for f in raw:
        name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
        if not name:
            continue
        dtype = (f.get("dtype") or f.get("data_type") or f.get("datatype")) if isinstance(f, dict) \
            else getattr(f, "dtype", getattr(f, "data_type", ""))
        kind = _kind(dtype)
        is_pk = bool(f.get("is_primary") if isinstance(f, dict) else getattr(f, "is_primary", False))
        params = (f.get("params") if isinstance(f, dict) else getattr(f, "params", None)) or {}
        fdim = int(params.get("dim", 0)) if kind == "float_vector" else 0
        if is_pk:
            seen_primary = True
        if kind == "float_vector":
            seen_vector = True
        fields.append((name, kind, is_pk, fdim))

    # Synthesize the primary/vector fields if the schema (or a bare dimension=)
    # didn't declare them, matching the old default schema.
    if not seen_primary:
        fields.insert(0, (primary, "varchar", True, 0))
    if not seen_vector:
        fields.append((vector, "float_vector", False, int(dim or 0)))
    return fields


# ---- pymilvus-compatible schema / index param builders ----

class _Field:
    def __init__(self, name, dtype, is_primary=False, max_length=0, dim=0, **kw):
        self.name = name
        self.dtype = dtype
        self.is_primary = is_primary
        self.params = {}
        if dim:
            self.params["dim"] = dim
        if max_length:
            self.params["max_length"] = max_length


class _Schema:
    def __init__(self, auto_id: bool = False, **_):
        self.fields: list[_Field] = []
        self.auto_id = auto_id

    def add_field(self, name: str, dtype, is_primary: bool = False, **kw) -> "_Schema":
        self.fields.append(_Field(name, dtype, is_primary=is_primary, **kw))
        return self

    def get(self, key, default=None):
        return getattr(self, key, default)


class _IndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kw):
        self.indexes.append(kw)


__all__ = ["MilvusClient", "DataType"]
