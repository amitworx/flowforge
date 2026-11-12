import unicodedata
from dataclasses import FrozenInstanceError

import pytest

from flowforge.ai.retrieval.chunking import (
    ChunkingConfig,
    TextChunk,
    chunk_text,
)
from flowforge.ai.retrieval.exceptions import (
    ChunkingError,
    RetrievalContractError,
)
from flowforge.ai.retrieval.protocols import (
    EmbeddingProvider,
    QueryBundle,
    SearchResult,
    VectorDocument,
    VectorIndex,
)
from flowforge.ai.retrieval.validation import (
    validate_chunk_bounds,
    validate_embedding_vector,
    validate_query_bundle,
    validate_search_results,
    validate_tenant_scope,
    validate_vector_documents,
)


class TestProtocolsAndDataStructures:
    def test_text_chunk_dataclass_immutability_and_fields(self):
        chunk = TextChunk(
            chunk_id="chunk-123",
            text="Hello world",
            start_offset=0,
            end_offset=11,
            index=0,
            metadata={"source": "doc1"},
        )
        assert chunk.chunk_id == "chunk-123"
        assert chunk.text == "Hello world"
        assert chunk.start_offset == 0
        assert chunk.end_offset == 11
        assert chunk.index == 0
        assert chunk.metadata == {"source": "doc1"}

        # TextChunk should be frozen
        with pytest.raises(FrozenInstanceError):
            chunk.text = "Changed"

    def test_vector_document_dataclass(self):
        doc = VectorDocument(
            id="doc-1",
            vector=(0.1, 0.2, 0.3),
            tenant_id="tenant-alpha",
            content="test document",
            metadata={"author": "alice"},
        )
        assert doc.id == "doc-1"
        assert doc.vector == (0.1, 0.2, 0.3)
        assert doc.tenant_id == "tenant-alpha"
        assert doc.content == "test document"
        assert doc.metadata == {"author": "alice"}

        # Immutability check
        with pytest.raises(FrozenInstanceError):
            doc.tenant_id = "tenant-beta"

    def test_query_bundle_dataclass(self):
        query = QueryBundle(
            query_text="search query",
            query_vector=(0.5, -0.5, 0.1),
            tenant_id="tenant-alpha",
            top_k=5,
            score_threshold=0.7,
            filter_metadata={"department": "finance"},
        )
        assert query.query_text == "search query"
        assert query.query_vector == (0.5, -0.5, 0.1)
        assert query.tenant_id == "tenant-alpha"
        assert query.top_k == 5
        assert query.score_threshold == 0.7
        assert query.filter_metadata == {"department": "finance"}

    def test_search_result_dataclass_ordering(self):
        res1 = SearchResult(
            id="doc-1",
            score=0.95,
            content="top result",
            tenant_id="tenant-alpha",
            metadata={},
        )
        res2 = SearchResult(
            id="doc-2",
            score=0.85,
            content="second result",
            tenant_id="tenant-alpha",
            metadata={},
        )
        assert res1.score > res2.score

    def test_mock_embedding_provider_satisfies_protocol(self):
        class DummyProvider:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[0.1 * (i + 1), 0.2 * (i + 1)] for i in range(len(texts))]

        provider = DummyProvider()
        assert isinstance(provider, EmbeddingProvider)
        assert provider.embed_texts(["hi", "there"]) == [[0.1, 0.2], [0.2, 0.4]]

    def test_mock_vector_index_satisfies_protocol(self):
        class DummyIndex:
            def upsert(
                self,
                tenant_id: str,  # noqa: ARG002
                documents: list[VectorDocument],
            ) -> int:
                return len(documents)

            def search(
                self,
                query: QueryBundle,  # noqa: ARG002
            ) -> list[SearchResult]:
                return []

        index = DummyIndex()
        assert isinstance(index, VectorIndex)
        assert index.upsert("tenant-alpha", []) == 0
        assert index.search(QueryBundle(tenant_id="tenant-alpha", top_k=5)) == []


class TestContractValidationHelpers:
    def test_validate_tenant_scope_rejects_empty_or_whitespace(self):
        with pytest.raises(RetrievalContractError, match="tenant_id cannot be empty"):
            validate_tenant_scope("")
        with pytest.raises(RetrievalContractError, match="tenant_id cannot be empty"):
            validate_tenant_scope("   ")
        with pytest.raises(RetrievalContractError, match="tenant_id must be a string"):
            validate_tenant_scope(None)
        assert validate_tenant_scope("tenant-123") == "tenant-123"
        assert validate_tenant_scope("  tenant-123  ") == "tenant-123"

    def test_validate_embedding_vector_rejects_nan_inf_and_dimension_mismatch(self):
        # Valid vector
        validate_embedding_vector([0.1, 0.2, 0.3], expected_dimension=3)

        # Dimension mismatch
        with pytest.raises(RetrievalContractError, match="Expected vector dimension 3, got 2"):
            validate_embedding_vector([0.1, 0.2], expected_dimension=3)

        # Non-numeric
        with pytest.raises(RetrievalContractError, match="Vector element.*must be float or int"):
            validate_embedding_vector([0.1, "bad", 0.3])

        # NaN
        with pytest.raises(RetrievalContractError, match="Vector contains non-finite"):
            validate_embedding_vector([0.1, float("nan"), 0.3])

        # Inf
        with pytest.raises(RetrievalContractError, match="Vector contains non-finite"):
            validate_embedding_vector([0.1, float("inf"), 0.3])
        with pytest.raises(RetrievalContractError, match="Vector contains non-finite"):
            validate_embedding_vector([0.1, float("-inf"), 0.3])

        # Empty vector
        with pytest.raises(RetrievalContractError, match="Vector cannot be empty"):
            validate_embedding_vector([])

    def test_validate_vector_documents_enforces_tenant_and_vectors(self):
        docs = [
            VectorDocument(
                id="d1",
                vector=(0.1, 0.2),
                tenant_id="t1",
                content="c1",
            ),
            VectorDocument(
                id="d2",
                vector=(0.3, 0.4),
                tenant_id="t1",
                content="c2",
            ),
        ]
        validate_vector_documents(docs, expected_tenant_id="t1", expected_dimension=2)

        # Cross-tenant mismatch in batch
        bad_docs = [
            VectorDocument(
                id="d1",
                vector=(0.1, 0.2),
                tenant_id="t1",
                content="c1",
            ),
            VectorDocument(
                id="d2",
                vector=(0.3, 0.4),
                tenant_id="foreign-tenant",
                content="c2",
            ),
        ]
        with pytest.raises(RetrievalContractError, match="Tenant isolation violation"):
            validate_vector_documents(bad_docs, expected_tenant_id="t1", expected_dimension=2)

        # Reject duplicate IDs in documents
        dup_docs = [
            VectorDocument(id="d1", vector=(0.1, 0.2), tenant_id="t1", content="c1"),
            VectorDocument(id="d1", vector=(0.3, 0.4), tenant_id="t1", content="c2"),
        ]
        with pytest.raises(RetrievalContractError, match="Duplicate document id"):
            validate_vector_documents(dup_docs, expected_tenant_id="t1", expected_dimension=2)

    def test_validate_query_bundle_enforces_positive_top_k_and_valid_score(self):
        # top_k must be >= 1
        with pytest.raises(
            RetrievalContractError, match="top_k must be greater than or equal to 1"
        ):
            validate_query_bundle(QueryBundle(tenant_id="t1", top_k=0))

        # score_threshold non-finite
        with pytest.raises(RetrievalContractError, match="score_threshold must be a finite"):
            validate_query_bundle(
                QueryBundle(tenant_id="t1", top_k=5, score_threshold=float("nan"))
            )

        # Empty tenant
        with pytest.raises(RetrievalContractError, match="tenant_id cannot be empty"):
            validate_query_bundle(QueryBundle(tenant_id="", top_k=5))

    def test_validate_search_results_enforces_deterministic_ordering_and_tenant(self):
        results = [
            SearchResult(id="d1", score=0.9, content="c1", tenant_id="t1"),
            SearchResult(id="d2", score=0.8, content="c2", tenant_id="t1"),
        ]
        validate_search_results(results, expected_tenant_id="t1")

        # Out of order results (must be sorted descending by score)
        unordered = [
            SearchResult(id="d1", score=0.7, content="c1", tenant_id="t1"),
            SearchResult(id="d2", score=0.9, content="c2", tenant_id="t1"),
        ]
        with pytest.raises(
            RetrievalContractError, match="Search results must be sorted descending by score"
        ):
            validate_search_results(unordered, expected_tenant_id="t1")

        # Non-finite score
        nan_score = [SearchResult(id="d1", score=float("nan"), content="c1", tenant_id="t1")]
        with pytest.raises(RetrievalContractError, match="Search result.*has non-finite score"):
            validate_search_results(nan_score, expected_tenant_id="t1")

        # Cross-tenant leak in search results
        leaked = [SearchResult(id="d1", score=0.9, content="c1", tenant_id="t2")]
        with pytest.raises(RetrievalContractError, match="Search result belongs to foreign tenant"):
            validate_search_results(leaked, expected_tenant_id="t1")


class TestDeterministicTextChunking:
    def test_empty_and_whitespace_only_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t  \r\n") == []
        assert chunk_text(None) == []

    def test_single_character_input(self):
        chunks = chunk_text("a", config=ChunkingConfig(max_chars=100, overlap=10))
        assert len(chunks) == 1
        assert chunks[0].text == "a"
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == 1
        assert chunks[0].index == 0

    def test_config_validation_rejects_invalid_limits(self):
        # max_chars must be >= 1
        with pytest.raises(ChunkingError, match="max_chars must be greater than or equal to 1"):
            ChunkingConfig(max_chars=0, overlap=0)

        # overlap cannot be negative
        with pytest.raises(ChunkingError, match="overlap cannot be negative"):
            ChunkingConfig(max_chars=100, overlap=-1)

        # overlap must be strictly less than max_chars (avoids infinite loop / zero progress)
        with pytest.raises(ChunkingError, match="overlap must be strictly less than max_chars"):
            ChunkingConfig(max_chars=50, overlap=50)

        with pytest.raises(ChunkingError, match="overlap must be strictly less than max_chars"):
            ChunkingConfig(max_chars=50, overlap=60)

    def test_chunking_bounds_and_safety_limits(self):
        # Max input length limit to prevent unbounded memory allocation
        with pytest.raises(ChunkingError, match="Input text length exceeds maximum allowed limit"):
            validate_chunk_bounds("x" * (10_000_001), max_input_length=10_000_000)

        # Max chunk count limit
        config = ChunkingConfig(max_chars=2, overlap=0, max_chunks=3)
        with pytest.raises(ChunkingError, match="Generated chunk count exceeds limit of 3"):
            chunk_text("abcdefgh", config=config)

    def test_deterministic_chunk_ids_hash_input_and_settings(self):
        text = "The quick brown fox jumps over the lazy dog."
        config1 = ChunkingConfig(max_chars=20, overlap=5)
        chunks1_first = chunk_text(text, config=config1)
        chunks1_second = chunk_text(text, config=config1)

        assert len(chunks1_first) > 0
        assert [c.chunk_id for c in chunks1_first] == [c.chunk_id for c in chunks1_second]

        # Different config yields different chunk IDs
        config2 = ChunkingConfig(max_chars=25, overlap=5)
        chunks2 = chunk_text(text, config=config2)
        assert chunks1_first[0].chunk_id != chunks2[0].chunk_id

    def test_paragraph_and_sentence_boundary_awareness(self):
        # When paragraph breaks exist within window, chunker prefers splitting at boundary
        para1 = "This is the first paragraph. It is quite descriptive."
        para2 = "This is the second paragraph. It continues the story."
        full_text = f"{para1}\n\n{para2}"

        # If max_chars is large enough to contain para1 but not both, without overlap
        config = ChunkingConfig(max_chars=70, overlap=0)
        chunks = chunk_text(full_text, config=config)

        assert len(chunks) == 2
        assert chunks[0].text == para1
        assert chunks[1].text == para2
        # Exact offsets in original text
        assert full_text[chunks[0].start_offset : chunks[0].end_offset] == chunks[0].text
        assert full_text[chunks[1].start_offset : chunks[1].end_offset] == chunks[1].text

    def test_exact_offsets_reproduce_slice_in_original_text(self):
        text = (
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi."
        )
        config = ChunkingConfig(max_chars=25, overlap=5)
        chunks = chunk_text(text, config=config)

        for c in chunks:
            assert text[c.start_offset : c.end_offset] == c.text
            assert len(c.text) <= 25

    def test_unicode_normalization_and_handling(self):
        # Combining accents / decomposed unicode
        # e.g., 'e' + combining acute accent vs precomposed 'é'
        text_decomposed = "Cafe\u0301 au lait. " * 5  # "Café au lait. "
        config = ChunkingConfig(max_chars=30, overlap=5, normalize_unicode="NFC")
        chunks = chunk_text(text_decomposed, config=config)

        assert len(chunks) > 0
        for c in chunks:
            assert unicodedata.is_normalized("NFC", c.text)
            assert len(c.text) <= 30

    def test_fallback_when_no_boundary_exists_within_window(self):
        # Single long word without spaces or paragraphs
        long_word = "a" * 100
        config = ChunkingConfig(max_chars=30, overlap=10)
        chunks = chunk_text(long_word, config=config)

        assert len(chunks) > 1
        assert chunks[0].text == "a" * 30
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == 30
        # Overlap step: 30 - 10 = 20
        assert chunks[1].start_offset == 20
        assert chunks[1].end_offset == 50
        assert chunks[1].text == "a" * 30
