"""PostgreSQL + pgvector live integration tests (M0-017A).

These tests run against a live PostgreSQL database with pgvector extension enabled.
If pgvector extension is unavailable in the environment, tests gracefully skip with clear reason,
satisfying CI requirements while validating full vector search semantics when available.
"""

from __future__ import annotations

import math
import os
import uuid

import psycopg
import pytest

from flowforge.ai.retrieval.pgvector import (
    DistanceMetric,
    PgVectorConfig,
    PgVectorIndex,
)
from flowforge.ai.retrieval.protocols import QueryBundle, VectorDocument


def get_pgvector_conn_string() -> str | None:
    """Return pgvector test DSN from environment or default local endpoints."""
    # Explicit env variable
    if "PGVECTOR_TEST_DSN" in os.environ:
        return os.environ["PGVECTOR_TEST_DSN"]

    # Try standard flowforge container on 5433 (pgvector container started locally)
    candidates = [
        "postgresql://flowforge:flowforge_password@localhost:5433/flowforge_test",
        "postgresql://flowforge:flowforge_password@localhost:5432/flowforge_test",
    ]
    for dsn in candidates:
        try:
            with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
                if cur.fetchone() is not None:
                    return dsn
        except Exception:
            continue
    return None


PGVECTOR_DSN = get_pgvector_conn_string()
requires_pgvector = pytest.mark.skipif(
    PGVECTOR_DSN is None,
    reason="PostgreSQL with pgvector extension not available on localhost:5432 or localhost:5433",
)


@pytest.fixture(scope="module")
def pgvector_index():
    """Fixture providing an isolated test table in pgvector test database."""
    if PGVECTOR_DSN is None:
        pytest.skip("pgvector database not available")

    table_name = f"test_vec_{uuid.uuid4().hex[:12]}"
    config = PgVectorConfig(
        table_name=table_name,
        dimension=3,
        metric=DistanceMetric.COSINE,
        create_extension=True,
    )
    index = PgVectorIndex(connection_factory=PGVECTOR_DSN, config=config)
    index.bootstrap()

    yield index

    # Cleanup table after tests
    try:
        with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    except Exception:
        pass


@requires_pgvector
def test_live_upsert_and_strict_tenant_isolation(pgvector_index: PgVectorIndex):
    """Prove that tenant isolation is strictly enforced at the SQL database layer.

    Tenant A and Tenant B both insert documents with the SAME document ID 'shared-doc-id'.
    Neither tenant can see or overwrite the other's document.
    """
    tenant_a = "tenant-alpha"
    tenant_b = "tenant-beta"

    # Tenant A documents
    doc_a = VectorDocument(
        id="shared-doc-id",
        vector=(1.0, 0.0, 0.0),
        tenant_id=tenant_a,
        content="Alpha confidential project specs",
        metadata={"project": "Alpha"},
    )
    # Tenant B documents
    doc_b = VectorDocument(
        id="shared-doc-id",
        vector=(0.0, 1.0, 0.0),
        tenant_id=tenant_b,
        content="Beta public roadmap",
        metadata={"project": "Beta"},
    )

    # Upsert both
    count_a = pgvector_index.upsert(tenant_a, [doc_a])
    count_b = pgvector_index.upsert(tenant_b, [doc_b])
    assert count_a == 1
    assert count_b == 1

    # Search Tenant A - MUST ONLY see Tenant A's document
    query_a = QueryBundle(
        tenant_id=tenant_a,
        query_vector=(1.0, 0.0, 0.0),
        top_k=10,
    )
    results_a = pgvector_index.search(query_a)
    assert len(results_a) == 1
    assert results_a[0].id == "shared-doc-id"
    assert results_a[0].tenant_id == tenant_a
    assert results_a[0].content == "Alpha confidential project specs"
    assert results_a[0].metadata == {"project": "Alpha"}
    assert math.isclose(results_a[0].score, 1.0, rel_tol=1e-5)

    # Search Tenant B - MUST ONLY see Tenant B's document
    query_b = QueryBundle(
        tenant_id=tenant_b,
        query_vector=(0.0, 1.0, 0.0),
        top_k=10,
    )
    results_b = pgvector_index.search(query_b)
    assert len(results_b) == 1
    assert results_b[0].id == "shared-doc-id"
    assert results_b[0].tenant_id == tenant_b
    assert results_b[0].content == "Beta public roadmap"
    assert results_b[0].metadata == {"project": "Beta"}
    assert math.isclose(results_b[0].score, 1.0, rel_tol=1e-5)

    # Delete Tenant A doc - Tenant B must remain completely intact
    del_count = pgvector_index.delete(tenant_a, ["shared-doc-id"])
    assert del_count == 1

    results_a_after = pgvector_index.search(query_a)
    assert len(results_a_after) == 0

    results_b_after = pgvector_index.search(query_b)
    assert len(results_b_after) == 1
    assert results_b_after[0].id == "shared-doc-id"


@requires_pgvector
def test_live_upsert_idempotency_and_update(pgvector_index: PgVectorIndex):
    """Prove that re-upserting a document with updated content/vector updates in-place."""
    tenant = "tenant-update-test"
    doc_initial = VectorDocument(
        id="doc-mutable",
        vector=(1.0, 0.0, 0.0),
        tenant_id=tenant,
        content="Initial content version 1",
        metadata={"version": 1},
    )
    pgvector_index.upsert(tenant, [doc_initial])

    # Re-upsert with updated content
    doc_updated = VectorDocument(
        id="doc-mutable",
        vector=(0.0, 1.0, 0.0),
        tenant_id=tenant,
        content="Updated content version 2",
        metadata={"version": 2},
    )
    pgvector_index.upsert(tenant, [doc_updated])

    # Search for updated vector
    results = pgvector_index.search(
        QueryBundle(tenant_id=tenant, query_vector=(0.0, 1.0, 0.0), top_k=5)
    )
    assert len(results) == 1
    assert results[0].content == "Updated content version 2"
    assert results[0].metadata == {"version": 2}
    assert math.isclose(results[0].score, 1.0, rel_tol=1e-5)


@requires_pgvector
def test_live_metadata_filtering_and_top_k(pgvector_index: PgVectorIndex):
    """Test safe metadata filters and top_k bounds."""
    tenant = "tenant-filter-test"
    docs = [
        VectorDocument(
            id=f"doc-{i}",
            vector=(float(i) / 10.0, 1.0 - float(i) / 10.0, 0.5),
            tenant_id=tenant,
            content=f"Doc {i} content",
            metadata={"category": "finance" if i % 2 == 0 else "tech", "priority": i},
        )
        for i in range(10)
    ]
    pgvector_index.upsert(tenant, docs)

    # Filter by category = finance
    q_filtered = QueryBundle(
        tenant_id=tenant,
        query_vector=(0.0, 1.0, 0.5),
        top_k=10,
        filter_metadata={"category": "finance"},
    )
    results = pgvector_index.search(q_filtered)
    assert len(results) == 5
    for r in results:
        assert r.metadata["category"] == "finance"

    # Top-k boundary limit = 2
    q_top2 = QueryBundle(
        tenant_id=tenant,
        query_vector=(0.0, 1.0, 0.5),
        top_k=2,
        filter_metadata={"category": "finance"},
    )
    results_top2 = pgvector_index.search(q_top2)
    assert len(results_top2) == 2


@requires_pgvector
def test_live_wrong_tenant_overwrite_prevention(pgvector_index: PgVectorIndex):
    """Prove that Tenant B cannot overwrite Tenant A's document via SQL."""
    tenant_a = "tenant-owner"
    tenant_attacker = "tenant-attacker"

    doc_a = VectorDocument(
        id="critical-doc",
        vector=(1.0, 0.0, 0.0),
        tenant_id=tenant_a,
        content="Original authentic text",
    )
    pgvector_index.upsert(tenant_a, [doc_a])

    # Attacker tries to upsert with same id under tenant_attacker scope
    doc_attacker = VectorDocument(
        id="critical-doc",
        vector=(0.0, 0.0, 1.0),
        tenant_id=tenant_attacker,
        content="Tampered text",
    )
    pgvector_index.upsert(tenant_attacker, [doc_attacker])

    # Verify Owner's document was NOT overwritten
    res_a = pgvector_index.search(
        QueryBundle(tenant_id=tenant_a, query_vector=(1.0, 0.0, 0.0), top_k=1)
    )
    assert len(res_a) == 1
    assert res_a[0].content == "Original authentic text"

    # Verify Attacker also cannot delete Owner's document
    del_res = pgvector_index.delete(tenant_attacker, ["critical-doc"])
    assert del_res == 1  # Attacker's own doc deleted

    res_a_still_there = pgvector_index.search(
        QueryBundle(tenant_id=tenant_a, query_vector=(1.0, 0.0, 0.0), top_k=1)
    )
    assert len(res_a_still_there) == 1
    assert res_a_still_there[0].content == "Original authentic text"


@requires_pgvector
def test_live_score_threshold_and_tie_breaking(pgvector_index: PgVectorIndex):
    """Test score threshold filtering and deterministic tie-breaking by id."""
    tenant = "tenant-tie-test"
    # Three identical vectors with different IDs
    docs = [
        VectorDocument(id="doc-c", vector=(1.0, 0.0, 0.0), tenant_id=tenant, content="C"),
        VectorDocument(id="doc-a", vector=(1.0, 0.0, 0.0), tenant_id=tenant, content="A"),
        VectorDocument(id="doc-b", vector=(1.0, 0.0, 0.0), tenant_id=tenant, content="B"),
    ]
    pgvector_index.upsert(tenant, docs)

    results = pgvector_index.search(
        QueryBundle(tenant_id=tenant, query_vector=(1.0, 0.0, 0.0), top_k=5)
    )
    assert len(results) == 3
    # Same score (1.0), order must be deterministic id asc: doc-a, doc-b, doc-c
    assert [r.id for r in results] == ["doc-a", "doc-b", "doc-c"]

    # Threshold test: query with orthogonal vector (0, 1, 0) gives score 0.0 (cosine sim)
    results_threshold = pgvector_index.search(
        QueryBundle(
            tenant_id=tenant,
            query_vector=(0.0, 1.0, 0.0),
            score_threshold=0.5,
            top_k=5,
        )
    )
    assert len(results_threshold) == 0


@requires_pgvector
def test_live_delete_tenant(pgvector_index: PgVectorIndex):
    """Test delete_tenant completely purges tenant rows without affecting others."""
    t1 = "tenant-purge-1"
    t2 = "tenant-purge-2"

    pgvector_index.upsert(
        t1,
        [
            VectorDocument(id="doc1", vector=(1.0, 0.0, 0.0), tenant_id=t1, content="1"),
            VectorDocument(id="doc2", vector=(1.0, 0.0, 0.0), tenant_id=t1, content="2"),
        ],
    )
    pgvector_index.upsert(
        t2,
        [VectorDocument(id="doc1", vector=(1.0, 0.0, 0.0), tenant_id=t2, content="t2 doc1")],
    )

    deleted = pgvector_index.delete_tenant(t1)
    assert deleted == 2

    # Verify t1 has 0 docs
    res_t1 = pgvector_index.search(QueryBundle(tenant_id=t1, query_vector=(1.0, 0.0, 0.0), top_k=5))
    assert len(res_t1) == 0

    # Verify t2 still has 1 doc
    res_t2 = pgvector_index.search(QueryBundle(tenant_id=t2, query_vector=(1.0, 0.0, 0.0), top_k=5))
    assert len(res_t2) == 1
