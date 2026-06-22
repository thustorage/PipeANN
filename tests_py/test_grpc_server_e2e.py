#!/usr/bin/env python3
"""End-to-end test for the PipeANN Milvus-compatible gRPC server.

Launches pipeann_milvus_server as a subprocess and drives it through pymilvus
MilvusClient, exercising the CRUD + query semantics added for the 1-5 gaps:
  - Insert with filterable scalars (string/int/float) + display fields
  - CreateIndex / load
  - Search with output_fields (scalar passthrough + decode)
  - Search with filter
  - Query with filter + output_fields
  - Get by primary key
  - Delete by ids and by filter expr
  - Upsert (primary-key overwrite)
  - count(*)

Run: python3 tests_py/test_grpc_server_e2e.py
"""
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _server_binary() -> Path:
    """Locate the server binary the same way the package launcher does."""
    try:
        from pipeann.server import find_server_binary

        found = find_server_binary()
        if found is not None:
            return found
    except Exception:
        pass
    # Fallback for a bare CMake build tree without an installed package.
    return ROOT / "build" / "src" / "server" / "pipeann_milvus_server"


SERVER = _server_binary()

DIM = 64
N = 500


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("server did not start listening in time")


def main() -> int:
    if not SERVER.exists():
        print(f"SKIP: server binary not built at {SERVER}")
        return 0

    from pymilvus import MilvusClient, DataType
    return _run_server(MilvusClient, DataType)


def _run_server(MilvusClient, DataType) -> int:

    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="pipeann_grpc_e2e_") as data_dir:
        proc = subprocess.Popen(
            [str(SERVER), "--data_dir", data_dir, "--port", str(port), "--threads", "4"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_port(port, proc)
            client = MilvusClient(uri=f"http://127.0.0.1:{port}")
            failures = run_checks(client, MilvusClient, DataType)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if failures:
        print(f"\nFAILED: {failures} check(s) failed")
        return 1
    print("\nPASSED: gRPC server e2e")
    return 0


def run_checks(client, MilvusClient, DataType) -> int:
    failures = 0

    def check(cond, msg):
        nonlocal failures
        if cond:
            print(f"  ok: {msg}")
        else:
            print(f"  FAIL: {msg}")
            failures += 1

    name = "e2e_collection"
    rng = np.random.default_rng(7)

    # Explicit schema: id (str pk), vector, color (str filter), tag (int filter),
    # price (float filter), note (display-only varchar... use JSON-free string).
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("color", DataType.VARCHAR, max_length=32)
    schema.add_field("tag", DataType.INT64)
    schema.add_field("price", DataType.FLOAT)
    client.create_collection(collection_name=name, schema=schema)
    check(client.has_collection(name), "collection created")

    colors = ["red", "green", "blue", "yellow"]
    rows = []
    for i in range(N):
        rows.append({
            "id": str(i),
            "vector": rng.normal(size=DIM).astype(np.float32).tolist(),
            "color": colors[i % 4],
            "tag": int(i % 10),
            "price": float(round(rng.uniform(1.0, 100.0), 2)),
        })
    res = client.insert(collection_name=name, data=rows)
    check(res["insert_count"] == N, f"inserted {N} rows")

    # AUTOINDEX path: pymilvus create_index needs index params; PipeANN builds on load.
    try:
        ip = client.prepare_index_params()
        ip.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
        client.create_index(name, ip)
    except Exception as e:
        print(f"  (create_index note: {e})")
    client.load_collection(name)

    # count(*)
    cnt = client.query(name, filter="", output_fields=["count(*)"])
    got = cnt[0]["count(*)"] if cnt else None
    check(got == N, f"count(*) == {N} (got {got})")

    # Search with output_fields: scalar passthrough + decode.
    q = rng.normal(size=DIM).astype(np.float32).tolist()
    sres = client.search(name, data=[q], limit=5, output_fields=["id", "color", "tag", "price"])
    check(len(sres) == 1 and len(sres[0]) > 0, "search returned hits")
    hit = sres[0][0]
    ent = hit.get("entity", {})
    check("color" in ent and ent["color"] in colors, f"search output color decoded ({ent.get('color')})")
    check("tag" in ent and 0 <= int(ent["tag"]) <= 9, f"search output tag decoded ({ent.get('tag')})")
    check("price" in ent and isinstance(ent["price"], (int, float)), f"search output price decoded ({ent.get('price')})")

    # Verify price decodes to the stored value for a known id.
    one = client.get(name, ids=[hit["id"]], output_fields=["price", "color"])
    if one:
        orig = rows[int(hit["id"])]
        check(abs(float(one[0]["price"]) - orig["price"]) < 0.01,
              f"get price matches insert ({one[0].get('price')} vs {orig['price']})")
        check(one[0].get("color") == orig["color"], "get color matches insert")

    # Search with filter.
    fres = client.search(name, data=[q], limit=10, filter='color == "red"', output_fields=["color"])
    reds_ok = all(h["entity"].get("color") == "red" for h in fres[0]) if fres and fres[0] else False
    check(reds_ok, "filtered search returns only red")

    # Query with filter + output_fields.
    qres = client.query(name, filter="tag == 3", output_fields=["id", "tag", "color"], limit=50)
    check(len(qres) > 0 and all(int(r["tag"]) == 3 for r in qres), f"query tag==3 ({len(qres)} rows)")

    # Query with float range filter.
    qrange = client.query(name, filter="price < 50.0", output_fields=["id", "price"], limit=1000)
    range_ok = len(qrange) > 0 and all(float(r["price"]) < 50.0 for r in qrange)
    check(range_ok, f"query price<50 all below ({len(qrange)} rows)")

    # Delete by ids.
    client.delete(name, ids=["0", "1", "2"])
    after = client.query(name, filter="", output_fields=["count(*)"])
    check(after[0]["count(*)"] == N - 3, f"count after delete-by-ids == {N-3} (got {after[0]['count(*)']})")

    # Delete by filter expr.
    del_res = client.delete(name, filter="tag == 7")
    remaining = client.query(name, filter="tag == 7", output_fields=["id"], limit=50)
    check(len(remaining) == 0, "delete by filter removed all tag==7")

    # Upsert overwrites by primary key.
    new_vec = rng.normal(size=DIM).astype(np.float32).tolist()
    client.upsert(name, data=[{"id": "100", "vector": new_vec, "color": "purple", "tag": 99, "price": 7.5}])
    up = client.get(name, ids=["100"], output_fields=["color", "tag", "price"])
    if up:
        check(up[0].get("color") == "purple" and int(up[0]["tag"]) == 99,
              f"upsert overwrote row 100 ({up[0].get('color')}, {up[0].get('tag')})")

    # Partition compatibility stubs: pymilvus partition calls must succeed even
    # though PipeANN keeps all rows in one implicit "_default" partition.
    try:
        client.create_partition(name, "p1")
        check(client.has_partition(name, "p1"), "has_partition true after create")
        parts = client.list_partitions(name)
        check("_default" in parts, f"list_partitions reports _default ({parts})")
        client.load_partitions(name, ["p1"])
        client.release_partitions(name, ["p1"])
        client.drop_partition(name, "p1")
        check(True, "partition load/release/drop stubs return ok")
    except Exception as e:
        check(False, f"partition stubs raised: {e}")

    client.drop_collection(name)
    check(not client.has_collection(name), "collection dropped")
    return failures


def test_grpc_server_e2e() -> None:
    import pytest

    if not SERVER.exists():
        pytest.skip(f"server binary not built at {SERVER}")
    try:
        from pymilvus import MilvusClient, DataType
    except ImportError:
        pytest.skip("pymilvus not installed")

    assert _run_server(MilvusClient, DataType) == 0


if __name__ == "__main__":
    sys.exit(main())
