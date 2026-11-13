"""Unit and contract tests for tenant-isolated pgvector vector index adapter (M0-017A / M0-017A-1)."""

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
from flowforge.ai.retrieval.pgvector.config import (
    MAX_BATCH_SIZE,
    MAX_FILTER_IN_ITEMS,
    MAX_METADATA_DEPTH,
    MAX_PGVECTOR_DIMENSION,
    MAX_TEXT_CONTENT_LENGTH,
    MAX_TOP_K,
)
from flowforge.ai.retrieval.protocols import QueryBundle, VectorDocument, VectorIndex

# ---------------------------------------------------------------------------
# Protocol and Configuration Unit Tests
# ---------------------------------------------------------------------------


def test_protocol_conformance():
    """Verify that PgVectorIndex conforms to VectorIndex protocol."""
    assert issubclass(PgVectorIndex, VectorIndex)

    index = PgVectorIndex(connection_factory=lambda: MagicMock())
    assert isinstance(index, VectorIndex)


def test_config_defaults_and_validation():
    """Verify default config values and validation constraints."""
    config = PgVectorConfig()
    assert config.table_name == "flowforge_ai_vector_documents"
    assert config.dimension == 1536
    assert config.metric == DistanceMetric.COSINE
    # Least privilege defaults
    assert config.create_extension is False
    assert config.auto_bootstrap is False

    # Custom valid config
    custom = PgVectorConfig(
        table_name="tenant_vectors_v2",
        dimension=384,
        metric=DistanceMetric.L2,
        create_extension=True,
        auto_bootstrap=True,
    )
    assert custom.table_name == "tenant_vectors_v2"
    assert custom.dimension == 384
    assert custom.metric == DistanceMetric.L2
    assert custom.create_extension is True
    assert custom.auto_bootstrap is True

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

    # Invalid dimensions (0, negative, > 2000, bool)
    with pytest.raises(RetrievalContractError, match="Invalid dimension"):
        PgVectorConfig(dimension=0)
    with pytest.raises(RetrievalContractError, match="Invalid dimension"):
        PgVectorConfig(dimension=-10)
    with pytest.raises(RetrievalContractError, match="Invalid dimension"):
        PgVectorConfig(dimension=MAX_PGVECTOR_DIMENSION + 1)
    with pytest.raises(RetrievalContractError, match="dimension must be an integer"):
        PgVectorConfig(dimension=True)  # type: ignore
    with pytest.raises(RetrievalContractError, match="dimension must be an integer"):
        PgVectorConfig(dimension=False)  # type: ignore

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


def test_filter_compiler_bounds_and_invalid_inputs():
    """Verify invalid filter inputs and $in bound checks raise PgVectorFilterError."""
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

    # $in items exceeding MAX_FILTER_IN_ITEMS
    with pytest.raises(PgVectorFilterError, match=r"\$in operator exceeds maximum"):
        compile_metadata_filters(
            {"tag": {"$in": [f"val-{i}" for i in range(MAX_FILTER_IN_ITEMS + 1)]}}
        )


# ---------------------------------------------------------------------------
# Tenant Isolation and Parameter Bounds Unit Tests
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


def test_tenant_id_strict_validation():
    """Validate tenant_id strictly rejects non-string and empty/whitespace inputs."""
    index = PgVectorIndex(connection_factory=lambda: MagicMock())
    doc = VectorDocument(
        id="doc-1",
        vector=(0.1, 0.2, 0.3),
        tenant_id="tenant-1",
        content="text",
    )

    with pytest.raises(RetrievalContractError, match="tenant_id cannot be empty"):
        index.upsert("", [doc])

    with pytest.raises(RetrievalContractError, match="tenant_id must be a string"):
        index.upsert(12345, [doc])  # type: ignore

    with pytest.raises(RetrievalContractError, match="tenant_id must be a string"):
        index.delete(None, ["doc-1"])  # type: ignore

    with pytest.raises(RetrievalContractError, match="document_ids must be a list"):
        index.delete("tenant-1", "doc-1")  # type: ignore

    with pytest.raises(
        RetrievalContractError, match="Each document ID in document_ids must be a string"
    ):
        index.delete("tenant-1", [123])  # type: ignore


def test_bounds_top_k_and_batch_size():
    """Verify top_k, batch size, text length, and metadata depth limits."""
    config = PgVectorConfig(dimension=3)
    index = PgVectorIndex(connection_factory=lambda: MagicMock(), config=config)

    # top_k limit
    q_huge = QueryBundle(
        tenant_id="tenant-1",
        query_vector=(0.1, 0.2, 0.3),
        top_k=MAX_TOP_K + 1,
    )
    with pytest.raises(RetrievalContractError, match="top_k exceeds maximum"):
        index.search(q_huge)

    # Batch size limit
    docs_huge = [
        VectorDocument(id=f"doc-{i}", vector=(0.1, 0.2, 0.3), tenant_id="tenant-1", content="t")
        for i in range(MAX_BATCH_SIZE + 1)
    ]
    with pytest.raises(RetrievalContractError, match="Batch size exceeds maximum"):
        index.upsert("tenant-1", docs_huge)

    # Content length limit
    doc_oversized = VectorDocument(
        id="doc-over",
        vector=(0.1, 0.2, 0.3),
        tenant_id="tenant-1",
        content="x" * (MAX_TEXT_CONTENT_LENGTH + 1),
    )
    with pytest.raises(RetrievalContractError, match="content length exceeds maximum"):
        index.upsert("tenant-1", [doc_oversized])

    # Metadata nesting depth limit
    deep_meta: dict = {"level": 1}
    curr = deep_meta
    for i in range(MAX_METADATA_DEPTH + 1):
        curr["child"] = {"level": i + 2}
        curr = curr["child"]
    doc_deep_meta = VectorDocument(
        id="doc-deep",
        vector=(0.1, 0.2, 0.3),
        tenant_id="tenant-1",
        content="t",
        metadata=deep_meta,
    )
    with pytest.raises(RetrievalContractError, match="Metadata nesting depth exceeds maximum"):
        index.upsert("tenant-1", [doc_deep_meta])


def test_zero_vector_and_precision_validation():
    """Verify zero-magnitude vectors are rejected for cosine similarity and precision is preserved."""
    config = PgVectorConfig(dimension=3, metric=DistanceMetric.COSINE)
    index = PgVectorIndex(connection_factory=lambda: MagicMock(), config=config)

    # Upsert zero vector
    doc_zero = VectorDocument(
        id="doc-0",
        vector=(0.0, 0.0, 0.0),
        tenant_id="tenant-1",
        content="zero",
    )
    with pytest.raises(RetrievalContractError, match="Zero-magnitude vector is invalid for cosine"):
        index.upsert("tenant-1", [doc_zero])

    # Search zero vector
    q_zero = QueryBundle(tenant_id="tenant-1", query_vector=(0.0, 0.0, 0.0))
    with pytest.raises(RetrievalContractError, match="Zero-magnitude vector is invalid for cosine"):
        index.search(q_zero)

    # Precision formatting test (no 8-digit rounding)
    high_precision_vec = (0.123456789012345, -0.987654321098765, 0.0)
    formatted = PgVectorIndex._format_vector(high_precision_vec)
    assert "0.123456789012345" in formatted
    assert "-0.987654321098765" in formatted


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

    mock_cur.fetchone.return_value = None
    mock_cur.execute.side_effect = [
        None,  # CHECK_EXTENSION_SQL
        Exception("extension 'vector' is not available"),  # CREATE_EXTENSION_SQL
    ]

    # Explicit create_extension=True
    config = PgVectorConfig(create_extension=True)
    index = PgVectorIndex(connection_factory=lambda: mock_conn, config=config)
    with pytest.raises(
        PgVectorExtensionUnavailableError, match="Failed to create pgvector extension"
    ):
        index.bootstrap(conn=mock_conn)

    # create_extension=False defaults
    config_no_create = PgVectorConfig(create_extension=False)
    index_no_create = PgVectorIndex(connection_factory=lambda: mock_conn, config=config_no_create)
    mock_cur.execute.side_effect = [None]
    with pytest.raises(
        PgVectorExtensionUnavailableError, match="not installed in the target database"
    ):
        index_no_create.bootstrap(conn=mock_conn)
