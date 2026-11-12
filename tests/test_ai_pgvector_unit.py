"""Unit and contract tests for tenant-isolated pgvector vector index adapter (M0-017A)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from flowforge.ai.retrieval.exceptions import RetrievalContractError
from flowforge.ai.retrieval.pgvector import (
    DistanceMetric,
    PgVectorConfig,
    PgVectorExtensionUnavailableError,
    PgVectorFilterError,
    PgVectorIndex,
    compile_metadata_filters,
)
from flowforge.ai.retrieval.protocols import QueryBundle, VectorDocument, VectorIndex

# ---------------------------------------------------------------------------
# Protocol and Configuration Unit Tests
# ---------------------------------------------------------------------------


def test_protocol_conformance():
    """Verify that PgVectorIndex conforms to VectorIndex protocol."""
    assert issubclass(PgVectorIndex, VectorIndex)

    # Instance check with mock connection factory
    index = PgVectorIndex(connection_factory=lambda: MagicMock())
    assert isinstance(index, VectorIndex)


def test_config_defaults_and_validation():
    """Verify default config values and validation constraints."""
    config = PgVectorConfig()
    assert config.table_name == "flowforge_ai_vector_documents"
    assert config.dimension == 1536
    assert config.metric == DistanceMetric.COSINE
    assert config.create_extension is True
    assert config.auto_bootstrap is True

    # Custom valid config
    custom = PgVectorConfig(
        table_name="tenant_vectors_v2",
        dimension=384,
        metric=DistanceMetric.L2,
    )
    assert custom.table_name == "tenant_vectors_v2"
    assert custom.dimension == 384
    assert custom.metric == DistanceMetric.L2

    # String metric auto-coercion
    str_metric = PgVectorConfig(metric="inner_product")
    assert str_metric.metric == DistanceMetric.INNER_PRODUCT

    # Invalid table names
    with pytest.raises(RetrievalContractError, match="Invalid table name"):
        PgVectorConfig(table_name="invalid table; DROP TABLE users;--")
    with pytest.raises(RetrievalContractError, match="Invalid table name"):
        PgVectorConfig(table_name="123_invalid_lead_digit")
    with pytest.raises(RetrievalContractError, match="Invalid table name"):
        PgVectorConfig(table_name="a" * 64)

    # Invalid dimensions
    with pytest.raises(RetrievalContractError, match="Invalid dimension"):
        PgVectorConfig(dimension=0)
    with pytest.raises(RetrievalContractError, match="Invalid dimension"):
        PgVectorConfig(dimension=-10)
    with pytest.raises(RetrievalContractError, match="Invalid dimension"):
        PgVectorConfig(dimension=20_000)

    # Invalid metric
    with pytest.raises(RetrievalContractError, match="Unsupported distance metric"):
        PgVectorConfig(metric="manhattan")


def test_distance_metric_scores():
    """Test distance metric conversions to similarity score."""
    # Cosine: 1.0 - distance
    assert DistanceMetric.COSINE.operator == "<=>"
    assert math.isclose(DistanceMetric.COSINE.score_from_distance(0.0), 1.0)
    assert math.isclose(DistanceMetric.COSINE.score_from_distance(1.0), 0.0)

    # L2: 1.0 / (1.0 + distance)
    assert DistanceMetric.L2.operator == "<->"
    assert math.isclose(DistanceMetric.L2.score_from_distance(0.0), 1.0)
    assert math.isclose(DistanceMetric.L2.score_from_distance(1.0), 0.5)

    # Inner product: -distance (pgvector <#> returns negative inner product)
    assert DistanceMetric.INNER_PRODUCT.operator == "<#>"
    assert math.isclose(DistanceMetric.INNER_PRODUCT.score_from_distance(-0.85), 0.85)


# ---------------------------------------------------------------------------
# Metadata Filter Compiler Unit Tests
# ---------------------------------------------------------------------------


def test_filter_compiler_empty():
    """Verify empty or None filters produce empty clause."""
    sql_clause, params = compile_metadata_filters(None)
    assert sql_clause == ""
    assert params == []

    sql_clause, params = compile_metadata_filters({})
    assert sql_clause == ""
    assert params == []


def test_filter_compiler_basic_equality():
    """Verify direct key-value pairs compile to parameterized JSONB queries."""
    sql_clause, params = compile_metadata_filters({"author": "alice", "category": "finance"})
    assert "metadata->>%s = %s" in sql_clause
    assert "author" in params
    assert "alice" in params
    assert "category" in params
    assert "finance" in params


def test_filter_compiler_operators():
    """Verify supported operators ($eq, $ne, $in) compile correctly."""
    filters = {
        "status": {"$eq": "published"},
        "archived": {"$ne": "true"},
        "tag": {"$in": ["ai", "nlp"]},
    }
    sql_clause, params = compile_metadata_filters(filters)
    assert "metadata->>%s = %s" in sql_clause
    assert "(metadata->>%s IS NULL OR metadata->>%s != %s)" in sql_clause
    assert "metadata->>%s = ANY(%s)" in sql_clause


def test_filter_compiler_invalid_inputs():
    """Verify invalid filter inputs raise PgVectorFilterError."""
    with pytest.raises(PgVectorFilterError, match="must be a dictionary"):
        compile_metadata_filters(["not", "a", "dict"])

    # Malicious key
    with pytest.raises(PgVectorFilterError, match="Invalid filter metadata key"):
        compile_metadata_filters({"author'; DROP TABLE students;--": "evil"})

    # Unsupported operator
    with pytest.raises(PgVectorFilterError, match="Unsupported filter operator"):
        compile_metadata_filters({"author": {"$regex": ".*"}})

    # Non-list for $in
    with pytest.raises(PgVectorFilterError, match=r"Operator \$in expects a list"):
        compile_metadata_filters({"tag": {"$in": "single_string"}})


# ---------------------------------------------------------------------------
# Tenant Isolation Contract Unit Tests
# ---------------------------------------------------------------------------


def test_upsert_rejects_foreign_tenant_documents():
    """Upsert must reject documents with tenant_id mismatched from the target tenant scope."""
    index = PgVectorIndex(connection_factory=lambda: MagicMock())
    doc = VectorDocument(
        id="doc-1",
        vector=(0.1, 0.2, 0.3),
        tenant_id="foreign-tenant",
        content="some text",
    )

    with pytest.raises(RetrievalContractError, match="Tenant isolation violation"):
        index.upsert("primary-tenant", [doc])


def test_upsert_rejects_empty_tenant():
    """Upsert must reject empty or whitespace tenant identifier."""
    index = PgVectorIndex(connection_factory=lambda: MagicMock())
    doc = VectorDocument(
        id="doc-1",
        vector=(0.1, 0.2, 0.3),
        tenant_id="",
        content="text",
    )
    with pytest.raises(RetrievalContractError, match="tenant_id cannot be empty"):
        index.upsert("", [doc])


def test_search_rejects_missing_query_vector():
    """Search requires query_vector when using pgvector adapter."""
    index = PgVectorIndex(connection_factory=lambda: MagicMock())
    query = QueryBundle(tenant_id="tenant-1", query_text="hello world", query_vector=None)
    with pytest.raises(RetrievalContractError, match="requires query_vector"):
        index.search(query)


def test_search_rejects_dimension_mismatch():
    """Search must reject query_vector with dimension not matching index configuration."""
    config = PgVectorConfig(dimension=3)
    index = PgVectorIndex(connection_factory=lambda: MagicMock(), config=config)
    query = QueryBundle(tenant_id="tenant-1", query_vector=(0.1, 0.2))  # dim 2 vs 3
    with pytest.raises(RetrievalContractError, match="Query vector dimension"):
        index.search(query)


# ---------------------------------------------------------------------------
# Extension Graceful Handling Unit Tests (Mock DB)
# ---------------------------------------------------------------------------


def test_bootstrap_extension_unavailable_raises_clean_error():
    """Bootstrap must raise PgVectorExtensionUnavailableError if extension cannot be created."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Mock extension not existing
    mock_cur.fetchone.return_value = None
    # Mock extension creation failure
    mock_cur.execute.side_effect = [
        None,  # CHECK_EXTENSION_SQL
        Exception("extension 'vector' is not available"),  # CREATE_EXTENSION_SQL
    ]

    index = PgVectorIndex(connection_factory=lambda: mock_conn)
    with pytest.raises(
        PgVectorExtensionUnavailableError, match="Failed to create pgvector extension"
    ):
        index.bootstrap(conn=mock_conn)
