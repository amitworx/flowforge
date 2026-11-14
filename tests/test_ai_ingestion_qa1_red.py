"""Red-phase tests for QA-1: Tenant-Safe Ingestion Privacy, Bounds, and Error Masking."""

from __future__ import annotations

import pytest

from flowforge.ai.retrieval.exceptions import RetrievalContractError, RetrievalError
from flowforge.ai.retrieval.ingestion import (
    IngestionConfig,
    IngestionDocument,
    IngestionService,
    _is_zero_vector,
)
from flowforge.ai.retrieval.pgvector.config import MAX_PGVECTOR_DIMENSION
from tests.test_ai_ingestion_unit import MockEmbeddingProvider, MockVectorIndex


def test_qa1_error_chaining_privacy_and_unmasking():
    """Verify provider exception chaining does not leak secret in __cause__ or __context__."""
    provider = MockEmbeddingProvider(raise_exception=True)
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    docs = [IngestionDocument(document_id="secret-doc", text="Confidential text")]

    with pytest.raises(RetrievalError) as exc_info:
        svc.ingest_documents("tenant-1", docs)

    # In Python, raise ... from None clears __cause__ and __suppress_context__ is True
    assert exc_info.value.__cause__ is None
    assert "sk-12345" not in str(exc_info.value)
    # Check that doc id is not leaked in error message
    assert "secret-doc" not in str(exc_info.value)


def test_qa1_query_error_chaining_privacy():
    provider = MockEmbeddingProvider(raise_exception=True)
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    with pytest.raises(RetrievalError) as exc_info:
        svc.query("tenant-1", "my secret query text with token sk-99999")

    assert exc_info.value.__cause__ is None
    assert "sk-12345" not in str(exc_info.value)
    assert "sk-99999" not in str(exc_info.value)


def test_qa1_ingestion_config_strict_validation():
    # bool rejected for batch_size, max_top_k, expected_dimension, etc.
    with pytest.raises(RetrievalContractError):
        IngestionConfig(batch_size=True)  # type: ignore

    with pytest.raises(RetrievalContractError):
        IngestionConfig(max_top_k=False)  # type: ignore

    with pytest.raises(RetrievalContractError):
        IngestionConfig(expected_dimension=True)  # type: ignore

    # float or non-int rejected
    with pytest.raises(RetrievalContractError):
        IngestionConfig(batch_size=12.5)  # type: ignore

    # dimension exceeding MAX_PGVECTOR_DIMENSION rejected
    with pytest.raises(RetrievalContractError):
        IngestionConfig(expected_dimension=MAX_PGVECTOR_DIMENSION + 1)

    # non-int caps
    with pytest.raises(RetrievalContractError):
        IngestionConfig(max_documents_per_batch=100.5)  # type: ignore

    with pytest.raises(RetrievalContractError):
        IngestionConfig(max_chunks_per_ingest="1000")  # type: ignore

    # malformed chunking_config
    with pytest.raises(RetrievalContractError):
        IngestionConfig(chunking_config=None)  # type: ignore

    with pytest.raises(RetrievalContractError):
        IngestionConfig(chunking_config="not-a-config")  # type: ignore


def test_qa1_query_input_validation_before_provider_call():
    provider = MockEmbeddingProvider()
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    # top_k bool, float, non-int, negative
    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", top_k=True)  # type: ignore

    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", top_k=3.5)  # type: ignore

    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", top_k=0)

    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", top_k=-5)

    # score_threshold non-numeric or non-finite
    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", score_threshold=float("nan"))

    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", score_threshold=float("inf"))

    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", score_threshold=True)  # type: ignore

    # provider should not have been called for invalid input
    assert len(provider.batches) == 0


def test_qa1_filter_metadata_pgvector_semantics():
    provider = MockEmbeddingProvider()
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    # Bad key pattern
    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", filter_metadata={"bad-key": "val"})

    # Unsupported operator
    with pytest.raises(RetrievalContractError):
        svc.query("tenant-1", "query text", filter_metadata={"tag": {"$gt": 5}})

    # Valid supported operators ($eq, $ne, $in)
    res = svc.query(
        "tenant-1",
        "query text",
        filter_metadata={"tag": {"$eq": "val"}, "category": {"$in": ["a", "b"]}},
    )
    assert isinstance(res, list)


def test_qa1_exact_zero_vector_and_tiny_valid_vectors():
    # Exact zeros
    assert _is_zero_vector((0.0, 0.0, 0.0)) is True
    assert _is_zero_vector([0, 0]) is True

    # Tiny valid vector should NOT be rejected as zero vector
    tiny_vec = (1e-7, 1e-7, 1e-7)
    assert _is_zero_vector(tiny_vec) is False
