"""Red-phase tests for QA-3: Cross-tenant result privacy, adapter return contract, filter values, top_k."""

from __future__ import annotations

import pytest

from flowforge.ai.retrieval.exceptions import RetrievalContractError
from flowforge.ai.retrieval.ingestion import (
    IngestionConfig,
    IngestionDocument,
    IngestionService,
    validate_query_filter_bounds,
)
from flowforge.ai.retrieval.protocols import QueryBundle, SearchResult, VectorDocument
from tests.test_ai_ingestion_unit import MockEmbeddingProvider, MockVectorIndex


class ForeignLeakingVectorIndex:
    """Mock VectorIndex returning foreign tenant results with sensitive IDs and contents."""

    def upsert(self, tenant_id: str, documents: list[VectorDocument]) -> int:
        del tenant_id
        return len(documents)

    def search(self, query: QueryBundle) -> list[SearchResult]:
        del query
        return [
            SearchResult(
                id="secret-foreign-doc-id-9999",
                score=0.9999,
                content="sensitive confidential foreign tenant text",
                tenant_id="foreign-secret-tenant-42",
                metadata={"secret_key": "confidential_val"},
            )
        ]


class BadUpsertReturnVectorIndex:
    """Mock VectorIndex returning invalid upsert return types/values."""

    def __init__(self, return_val):
        self.return_val = return_val

    def upsert(self, tenant_id: str, documents: list[VectorDocument]) -> int:
        del tenant_id, documents
        return self.return_val

    def search(self, query: QueryBundle) -> list[SearchResult]:
        del query
        return []


def test_qa3_cross_tenant_search_result_privacy():
    """Verify foreign tenant details, result IDs, scores, and contents do not leak in exception string, repr, args, cause or context."""
    provider = MockEmbeddingProvider(dimension=4)
    index = ForeignLeakingVectorIndex()
    svc = IngestionService(provider, index)

    with pytest.raises(RetrievalContractError) as exc_info:
        svc.query("tenant-1", "test query")

    err = exc_info.value
    err_str = str(err)
    err_repr = repr(err)
    err_args = str(err.args)

    assert err.__cause__ is None
    assert err.__context__ is None

    # Foreign tenant details must NOT leak
    assert "foreign-secret-tenant-42" not in err_str
    assert "foreign-secret-tenant-42" not in err_repr
    assert "foreign-secret-tenant-42" not in err_args
    assert "secret-foreign-doc-id-9999" not in err_str
    assert "0.9999" not in err_str
    assert "confidential" not in err_str
    assert "Vector search results violated tenant safety contracts" in err_str


@pytest.mark.parametrize(
    "bad_return",
    [-1, True, False, "5", 3.14, 10],  # 10 is impossible when only 1 doc was submitted
)
def test_qa3_adapter_upsert_return_contract(bad_return):
    """Verify upsert return value is strictly validated (int >= 0, <= submitted count, not bool)."""
    provider = MockEmbeddingProvider(dimension=4)
    index = BadUpsertReturnVectorIndex(bad_return)
    svc = IngestionService(provider, index, config=IngestionConfig(expected_dimension=4))

    docs = [IngestionDocument(document_id="doc1", text="content")]

    with pytest.raises(
        RetrievalContractError, match="Vector index upsert returned invalid document count"
    ):
        svc.ingest_documents("tenant-1", docs)


def test_qa3_strict_filter_scalar_types_and_bounds():
    """Enforce bounded primitive scalar types, reject bytes, objects, non-finite floats, oversized strings, and multiple operators."""
    # Reject non-finite floats in $eq
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"score": {"$eq": float("nan")}})

    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"score": {"$eq": float("inf")}})

    # Reject bytes
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"raw": {"$eq": b"bytes"}})

    # Reject arbitrary Python objects
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"obj": {"$eq": object()}})

    # Reject oversized scalar string (e.g. > 1024 chars)
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"long_str": {"$eq": "a" * 2000}})

    # Reject non-finite floats in $in
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"values": {"$in": [1, float("nan")]}})

    # Reject multiple operators in single operator dict
    with pytest.raises(RetrievalContractError):
        validate_query_filter_bounds({"field": {"$eq": 1, "$ne": 2}})

    # Valid scalar types
    validate_query_filter_bounds(
        {
            "str_field": "valid",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "none_field": None,
            "op_eq": {"$eq": "val"},
            "op_ne": {"$ne": 10},
            "op_in": {"$in": ["a", 1, 2.5, False, None]},
        }
    )


def test_qa3_top_k_fail_closed_on_exceeding_max():
    """top_k > config.max_top_k must fail closed with RetrievalContractError rather than silent clamping."""
    provider = MockEmbeddingProvider(dimension=4)
    index = MockVectorIndex()
    cfg = IngestionConfig(max_top_k=5)
    svc = IngestionService(provider, index, config=cfg)

    with pytest.raises(RetrievalContractError, match="top_k exceeds maximum allowed limit"):
        svc.query("tenant-1", "query text", top_k=6)
