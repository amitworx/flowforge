"""Red-phase tests for QA-2: Safe Exception Boundaries & Strict Retrieval Validation."""

from __future__ import annotations

import pytest

from flowforge.ai.retrieval.exceptions import RetrievalContractError, RetrievalError
from flowforge.ai.retrieval.ingestion import (
    IngestionConfig,
    IngestionDocument,
    IngestionService,
    validate_query_filter_bounds,
)
from flowforge.ai.retrieval.protocols import QueryBundle, SearchResult, VectorDocument
from tests.test_ai_ingestion_unit import MockEmbeddingProvider, MockVectorIndex


class FailingVectorIndex:
    """Mock VectorIndex that simulates DB/adapter failure containing sensitive SQL."""

    def upsert(self, tenant_id: str, documents: list[VectorDocument]) -> int:
        del tenant_id, documents
        raise RuntimeError(
            "psycopg.OperationalError: INSERT INTO pgvector_raw VALUES ('secret_data')"
        )

    def search(self, query: QueryBundle) -> list[SearchResult]:
        del query
        raise RuntimeError(
            "psycopg.OperationalError: SELECT * FROM secrets WHERE key='sk-db-secret'"
        )


def test_qa2_exception_context_completely_severed():
    """Ensure __context__ and __cause__ are completely None and no secret is reachable."""
    provider = MockEmbeddingProvider(raise_exception=True)
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    docs = [IngestionDocument(document_id="doc1", text="some text")]

    with pytest.raises(RetrievalError) as exc_info:
        svc.ingest_documents("tenant-1", docs)

    err = exc_info.value
    # __cause__ must be None
    assert err.__cause__ is None
    # __context__ must be explicitly severed (None) to avoid inspect/traceback leaking secrets
    assert err.__context__ is None
    assert "sk-12345" not in repr(err)
    assert "sk-12345" not in str(err)

    # Test query path as well
    with pytest.raises(RetrievalError) as query_exc_info:
        svc.query("tenant-1", "my query")

    q_err = query_exc_info.value
    assert q_err.__cause__ is None
    assert q_err.__context__ is None
    assert "sk-12345" not in repr(q_err)


def test_qa2_vector_index_adapter_failure_wrapped_safely():
    """Ensure adapter/database failures in vector_index.upsert and vector_index.search are wrapped safely."""
    provider = MockEmbeddingProvider(dimension=4)
    failing_index = FailingVectorIndex()
    svc = IngestionService(provider, failing_index, config=IngestionConfig(expected_dimension=4))

    docs = [IngestionDocument(document_id="doc1", text="some text")]

    # Upsert failure
    with pytest.raises(RetrievalError) as exc_info:
        svc.ingest_documents("tenant-1", docs)

    err = exc_info.value
    assert err.__cause__ is None
    assert err.__context__ is None
    assert "secret_data" not in str(err)
    assert "secret_data" not in repr(err)
    assert "psycopg" not in str(err)
    assert "Vector index operation failed" in str(err)

    # Search failure
    with pytest.raises(RetrievalError) as query_exc_info:
        svc.query("tenant-1", "search query")

    q_err = query_exc_info.value
    assert q_err.__cause__ is None
    assert q_err.__context__ is None
    assert "sk-db-secret" not in str(q_err)
    assert "sk-db-secret" not in repr(q_err)
    assert "Vector index search operation failed" in str(q_err)


def test_qa2_strict_query_filter_bounds_and_scalar_types():
    """Ensure validate_query_filter_bounds validates scalar types for $eq, $ne, and elements in $in."""
    # Complex non-scalar in $eq
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"status": {"$eq": {"nested": "obj"}}})

    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"status": {"$eq": [1, 2]}})

    # Complex non-scalar in $ne
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"status": {"$ne": {"nested": "obj"}}})

    # Non-scalar in $in list elements
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"status": {"$in": [{"dict": 1}]}})

    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"status": {"$in": [[1, 2]]}})

    # Valid scalar values allowed
    validate_query_filter_bounds({"status": {"$eq": "active"}, "count": {"$ne": 0}})
    validate_query_filter_bounds({"tags": {"$in": ["a", "b", 123, True, 4.5]}})
