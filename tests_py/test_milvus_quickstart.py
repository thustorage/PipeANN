"""Test PipeANN's MilvusClient against the official Milvus Quick Start workflow.

Covers: create_collection, insert, search, query, delete, drop_collection.
Uses the local PipeANN MilvusClient (no gRPC server needed).

Reference: https://milvus.io/docs/quickstart.md
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeann import MilvusClient, DataType


def main():
    with tempfile.TemporaryDirectory(prefix="pipeann_quickstart_") as tmp_dir:
        # 1. Connect
        client = MilvusClient(uri=tmp_dir)
        print("Connected to PipeANN MilvusClient")

        # 2. Create collection. Scalar fields used in filters must be declared
        # up front (the schema is fixed; the engine indexes them for filtering).
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=128)
        schema.add_field("color", DataType.VARCHAR, max_length=32)
        schema.add_field("price", DataType.FLOAT)
        client.create_collection(collection_name="demo_collection", schema=schema)
        assert client.has_collection("demo_collection")
        print("Created collection: demo_collection")

        # 3. Insert data
        rng = np.random.default_rng(42)
        n = 1000
        data = [
            {
                "id": str(i),
                "vector": rng.normal(size=128).astype(np.float32).tolist(),
                "color": ["red", "green", "blue", "yellow"][i % 4],
                "price": float(rng.integers(10, 500)),
            }
            for i in range(n)
        ]
        res = client.insert(collection_name="demo_collection", data=data)
        assert res["insert_count"] == n
        print(f"Inserted {n} entities")

        # 4. Search (single vector)
        query_vector = rng.normal(size=128).astype(np.float32).tolist()
        results = client.search(
            collection_name="demo_collection",
            data=[query_vector],
            limit=5,
            output_fields=["id", "color", "price"],
        )
        assert len(results) == 1  # one query
        assert len(results[0]) <= 5
        for hit in results[0]:
            assert "id" in hit
            assert "distance" in hit
            assert "entity" in hit
            assert "color" in hit["entity"]
            assert "price" in hit["entity"]
        print(f"Search returned {len(results[0])} hits")
        print(f"  Top hit: id={results[0][0]['id']} dist={results[0][0]['distance']:.4f}")

        # 5. Search with filter
        results_filtered = client.search(
            collection_name="demo_collection",
            data=[query_vector],
            limit=5,
            filter='color == "red"',
            output_fields=["id", "color"],
        )
        assert len(results_filtered) == 1
        for hit in results_filtered[0]:
            assert hit["entity"]["color"] == "red"
        print(f"Filtered search (color=red): {len(results_filtered[0])} hits")

        # 6. Query (get by filter)
        query_res = client.query(
            collection_name="demo_collection",
            filter='id in ["0", "1", "2"]',
            output_fields=["id", "color", "price"],
        )
        assert len(query_res) == 3
        returned_ids = {r["id"] for r in query_res}
        assert returned_ids == {"0", "1", "2"}
        print(f"Query by id: got {len(query_res)} entities")

        # 7. Delete
        del_res = client.delete(
            collection_name="demo_collection",
            filter='id in ["0", "1"]',
        )
        assert del_res["delete_count"] == 2
        print(f"Deleted 2 entities")

        # Verify deletion
        query_after = client.query(
            collection_name="demo_collection",
            filter='id in ["0", "1", "2"]',
            output_fields=["id"],
        )
        remaining_ids = {r["id"] for r in query_after}
        assert "0" not in remaining_ids
        assert "1" not in remaining_ids
        assert "2" in remaining_ids
        print("Verified deletion")

        # 8. Drop collection
        client.drop_collection("demo_collection")
        assert not client.has_collection("demo_collection")
        print("Dropped collection")

    print("\nALL QUICK START TESTS PASSED")


def test_milvus_quickstart() -> None:
    main()


if __name__ == "__main__":
    main()
