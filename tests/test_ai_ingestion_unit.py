"""Comprehensive unit tests for M0-017B Tenant-Safe Ingestion and Retrieval Core."""

from __future__ import annotations

import pytest

from flowforge.ai.retrieval.chunking import ChunkingConfig
from flowforge.ai.retrieval.exceptions import RetrievalContractError, RetrievalError
from flowforge.ai.retrieval.ingestion import (
    IngestionConfig,
    IngestionDocument,
    IngestionService,
    build_tenant_document_id,
    validate_query_filter_bounds,
)
from flowforge.ai.retrieval.protocols import (
    QueryBundle,
    SearchResult,
    VectorDocument,
)


class MockEmbeddingProvider:
    def __init__(
        self,
        dimension: int = 4,
        fail_nonfinite: bool = False,
        count_delta: int = 0,
        zero_vector: bool = False,
        raise_exception: bool = False,
    ):
        self.dimension = dimension
        self.fail_nonfinite = fail_nonfinite
        self.count_delta = count_delta
        self.zero_vector = zero_vector
        self.raise_exception = raise_exception
        self.batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.raise_exception:
            raise RuntimeError("Remote provider connection timed out with secret=sk-12345")
        self.batches.append(texts)
        if self.count_delta != 0:
            count = max(0, len(texts) + self.count_delta)
            return [[0.1] * self.dimension for _ in range(count)]
        if self.fail_nonfinite:
            return [[float("nan")] * self.dimension for _ in texts]
        if self.zero_vector:
            return [[0.0] * self.dimension for _ in texts]

        results = []
        for text in texts:
            # Deterministic non-zero values
            val = (len(text) % 7 + 1) / 10.0
            results.append([val] * self.dimension)
        return results


class MockVectorIndex:
    def __init__(self, foreign_tenant_result: bool = False, nonfinite_score: bool = False):
        self.docs: dict[str, VectorDocument] = {}
        self.upsert_calls: list[tuple[str, list[VectorDocument]]] = []
        self.search_calls: list[QueryBundle] = []
        self.foreign_tenant_result = foreign_tenant_result
        self.nonfinite_score = nonfinite_score

    def upsert(self, tenant_id: str, documents: list[VectorDocument]) -> int:
        self.upsert_calls.append((tenant_id, documents))
        for doc in documents:
            self.docs[f"{tenant_id}:{doc.id}"] = doc
        return len(documents)

    def search(self, query: QueryBundle) -> list[SearchResult]:
        self.search_calls.append(query)
        res = []
        for _, doc in self.docs.items():
            t_id = "foreign_tenant" if self.foreign_tenant_result else doc.tenant_id
            score = float("nan") if self.nonfinite_score else 0.95
            res.append(
                SearchResult(
                    id=doc.id,
                    score=score,
                    content=doc.content,
                    tenant_id=t_id,
                    metadata=doc.metadata,
                )
            )
        return res[: query.top_k]


# --- IngestionConfig & ID construction tests ---


def test_ingestion_config_validation():
    cfg = IngestionConfig(batch_size=10, max_top_k=20, expected_dimension=128)
    assert cfg.batch_size == 10
    assert cfg.max_top_k == 20
    assert cfg.expected_dimension == 128

    with pytest.raises(RetrievalContractError, match="batch_size"):
        IngestionConfig(batch_size=0)

    with pytest.raises(RetrievalContractError, match="max_top_k"):
        IngestionConfig(max_top_k=0)

    with pytest.raises(RetrievalContractError, match="expected_dimension"):
        IngestionConfig(expected_dimension=0)


def test_build_tenant_document_id():
    # Valid deterministic ID
    doc_id1 = build_tenant_document_id(
        "tenant-alpha", "doc-1", chunk_index=0, chunk_hash="abcdef123456"
    )
    doc_id2 = build_tenant_document_id(
        "tenant-alpha", "doc-1", chunk_index=0, chunk_hash="abcdef123456"
    )
    assert doc_id1 == doc_id2
    assert "doc-1" in doc_id1
    assert "c0" in doc_id1
    assert "abcdef12" in doc_id1

    # Cross tenant creates different IDs
    doc_id_beta = build_tenant_document_id(
        "tenant-beta", "doc-1", chunk_index=0, chunk_hash="abcdef123456"
    )
    assert doc_id1 != doc_id_beta

    # Rejection of invalid characters or empty IDs
    with pytest.raises(RetrievalContractError):
        build_tenant_document_id("", "doc-1")

    with pytest.raises(RetrievalContractError):
        build_tenant_document_id("tenant-alpha", "")

    with pytest.raises(RetrievalContractError):
        build_tenant_document_id("tenant-alpha", "doc/invalid/slash")

    with pytest.raises(RetrievalContractError):
        build_tenant_document_id("tenant-alpha", "a" * 200)


# --- DI and Service Ingestion Tests ---


def test_service_requires_di():
    provider = MockEmbeddingProvider()
    index = MockVectorIndex()
    with pytest.raises(RetrievalContractError, match="embedding_provider"):
        IngestionService(embedding_provider=None, vector_index=index)

    with pytest.raises(RetrievalContractError, match="vector_index"):
        IngestionService(embedding_provider=provider, vector_index=None)


def test_ingest_documents_empty_list():
    provider = MockEmbeddingProvider()
    index = MockVectorIndex()
    svc = IngestionService(provider, index)
    res = svc.ingest_documents("tenant-1", [])
    assert res.documents_processed == 0
    assert res.chunks_created == 0
    assert res.upserted_count == 0
    assert len(index.upsert_calls) == 0


def test_ingest_documents_success_and_idempotency():
    provider = MockEmbeddingProvider(dimension=4)
    index = MockVectorIndex()
    cfg = IngestionConfig(
        batch_size=2,
        expected_dimension=4,
        chunking_config=ChunkingConfig(max_chars=50, overlap=10),
    )
    svc = IngestionService(provider, index, config=cfg)

    docs = [
        IngestionDocument(
            document_id="doc-A",
            text="First sentence. Second sentence is slightly longer. Third sentence wraps up.",
            metadata={"category": "finance", "tags": ["q3", "report"]},
        ),
        IngestionDocument(
            document_id="doc-B",
            text="Short note.",
            metadata={"priority": 1},
        ),
    ]

    res1 = svc.ingest_documents("tenant-1", docs)
    assert res1.documents_processed == 2
    assert res1.chunks_created > 0
    assert res1.upserted_count == res1.chunks_created
    assert len(index.docs) == res1.chunks_created

    # Verify provenance metadata preserved
    first_chunk_record = res1.chunk_records[0]
    upserted_doc = list(index.docs.values())[0]
    assert "_provenance" in upserted_doc.metadata
    assert upserted_doc.metadata["_provenance"]["source_doc_id"] in ["doc-A", "doc-B"]
    assert upserted_doc.metadata["_provenance"]["chunk_id"] == first_chunk_record.chunk_id

    # Verify idempotency: ingesting again updates same doc IDs and count
    res2 = svc.ingest_documents("tenant-1", docs)
    assert res2.chunks_created == res1.chunks_created
    assert len(index.docs) == res1.chunks_created  # Total unique keys unchanged


def test_ingest_tenant_isolation_and_rejection():
    provider = MockEmbeddingProvider()
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    # Missing or empty tenant_id
    with pytest.raises(RetrievalContractError):
        svc.ingest_documents("", [IngestionDocument(document_id="d1", text="abc")])

    with pytest.raises(RetrievalContractError):
        svc.ingest_documents("   ", [IngestionDocument(document_id="d1", text="abc")])

    # Duplicate document IDs in single batch
    with pytest.raises(RetrievalContractError, match="Duplicate document_id"):
        svc.ingest_documents(
            "tenant-1",
            [
                IngestionDocument(document_id="d1", text="hello"),
                IngestionDocument(document_id="d1", text="world"),
            ],
        )


def test_embedding_count_mismatch_fails_closed():
    provider = MockEmbeddingProvider(count_delta=-1)
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    docs = [IngestionDocument(document_id="d1", text="Hello world!")]
    with pytest.raises(RetrievalContractError, match="Embedding provider returned"):
        svc.ingest_documents("tenant-1", docs)
    # Index should not have received any upsert
    assert len(index.upsert_calls) == 0


def test_embedding_nonfinite_values_fails_closed():
    provider = MockEmbeddingProvider(fail_nonfinite=True)
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    docs = [IngestionDocument(document_id="d1", text="Hello world!")]
    with pytest.raises(RetrievalContractError, match="non-finite value"):
        svc.ingest_documents("tenant-1", docs)
    assert len(index.upsert_calls) == 0


def test_embedding_zero_vector_fails_closed():
    provider = MockEmbeddingProvider(zero_vector=True)
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    docs = [IngestionDocument(document_id="d1", text="Hello world!")]
    with pytest.raises(RetrievalContractError, match="zero norm"):
        svc.ingest_documents("tenant-1", docs)
    assert len(index.upsert_calls) == 0


def test_embedding_dimension_mismatch_fails_closed():
    provider = MockEmbeddingProvider(dimension=8)
    index = MockVectorIndex()
    cfg = IngestionConfig(expected_dimension=16)
    svc = IngestionService(provider, index, config=cfg)

    docs = [IngestionDocument(document_id="d1", text="Hello world!")]
    with pytest.raises(RetrievalContractError, match="Expected vector dimension 16, got 8"):
        svc.ingest_documents("tenant-1", docs)
    assert len(index.upsert_calls) == 0


def test_sanitization_masks_sensitive_errors():
    provider = MockEmbeddingProvider(raise_exception=True)
    index = MockVectorIndex()
    svc = IngestionService(provider, index)

    docs = [IngestionDocument(document_id="d1", text="Confidential text")]
    with pytest.raises(RetrievalError) as exc_info:
        svc.ingest_documents("tenant-1", docs)

    # Ensure secret string is not in the message of RetrievalError
    assert "sk-12345" not in str(exc_info.value)
    assert "Embedding provider call failed" in str(exc_info.value)


# --- Query Helper Tests ---


def test_query_success_and_scoping():
    provider = MockEmbeddingProvider(dimension=4)
    index = MockVectorIndex()
    cfg = IngestionConfig(expected_dimension=4, max_top_k=5)
    svc = IngestionService(provider, index, config=cfg)

    # Ingest document
    svc.ingest_documents(
        "tenant-1", [IngestionDocument(document_id="d1", text="Sample document text")]
    )

    results = svc.query("tenant-1", "search keyword", top_k=2)
    assert len(results) > 0
    assert results[0].tenant_id == "tenant-1"
    assert len(index.search_calls) == 1
    call = index.search_calls[0]
    assert call.tenant_id == "tenant-1"
    assert call.top_k == 2
    assert len(call.query_vector) == 4


def test_query_enforces_max_top_k():
    provider = MockEmbeddingProvider(dimension=4)
    index = MockVectorIndex()
    cfg = IngestionConfig(max_top_k=5)
    svc = IngestionService(provider, index, config=cfg)

    svc.query("tenant-1", "hello", top_k=50)
    assert index.search_calls[0].top_k == 5  # Clamped to max_top_k


def test_query_foreign_tenant_result_fails_closed():
    provider = MockEmbeddingProvider(dimension=4)
    index = MockVectorIndex(foreign_tenant_result=True)
    svc = IngestionService(provider, index)

    # Manually populate foreign doc in index
    index.docs["other:doc"] = VectorDocument(
        id="doc",
        vector=(0.1, 0.2, 0.3, 0.4),
        tenant_id="foreign_tenant",
        content="foreign content",
    )

    with pytest.raises(RetrievalContractError, match="Search result belongs to foreign tenant"):
        svc.query("tenant-1", "hello")


def test_query_filter_bounds():
    with pytest.raises(RetrievalContractError, match="maximum allowed filter keys"):
        validate_query_filter_bounds({f"k_{i}": i for i in range(25)})

    with pytest.raises(RetrievalContractError, match="exceeds max items"):
        validate_query_filter_bounds({"tag": list(range(105))})
