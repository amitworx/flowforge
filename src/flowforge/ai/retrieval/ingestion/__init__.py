"""Deterministic, tenant-safe embedding ingestion and retrieval orchestration core."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from flowforge.ai.retrieval.chunking import ChunkingConfig, TextChunk, chunk_text
from flowforge.ai.retrieval.exceptions import RetrievalContractError, RetrievalError
from flowforge.ai.retrieval.pgvector.config import (
    MAX_BATCH_SIZE,
    MAX_METADATA_DEPTH,
    MAX_METADATA_SIZE_BYTES,
    MAX_TEXT_CONTENT_LENGTH,
    MAX_TOP_K,
)
from flowforge.ai.retrieval.protocols import (
    EmbeddingProvider,
    QueryBundle,
    SearchResult,
    VectorDocument,
    VectorIndex,
)
from flowforge.ai.retrieval.validation import (
    validate_embedding_vector,
    validate_search_results,
    validate_tenant_scope,
    validate_vector_documents,
)

# Limits and safety constants
MAX_DOCUMENT_ID_LENGTH: int = 128
MAX_DOCUMENTS_PER_BATCH: int = 1_000
MAX_CHUNKS_PER_INGEST: int = 10_000
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_MAX_TOP_K: int = 100
_DOC_ID_ALLOWED_RE = re.compile(r"^[a-zA-Z0-9_\-\.:#]+$")


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Configuration options for tenant-safe ingestion service."""

    batch_size: int = DEFAULT_BATCH_SIZE
    max_top_k: int = DEFAULT_MAX_TOP_K
    expected_dimension: int | None = None
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    max_documents_per_batch: int = MAX_DOCUMENTS_PER_BATCH
    max_chunks_per_ingest: int = MAX_CHUNKS_PER_INGEST

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.batch_size > MAX_BATCH_SIZE:
            raise RetrievalContractError(
                f"batch_size must be between 1 and {MAX_BATCH_SIZE}, got {self.batch_size}"
            )
        if self.max_top_k < 1 or self.max_top_k > MAX_TOP_K:
            raise RetrievalContractError(
                f"max_top_k must be between 1 and {MAX_TOP_K}, got {self.max_top_k}"
            )
        if self.expected_dimension is not None and self.expected_dimension < 1:
            raise RetrievalContractError(
                f"expected_dimension must be positive integer, got {self.expected_dimension}"
            )
        if self.max_documents_per_batch < 1:
            raise RetrievalContractError("max_documents_per_batch must be >= 1")
        if self.max_chunks_per_ingest < 1:
            raise RetrievalContractError("max_chunks_per_ingest must be >= 1")


@dataclass(frozen=True, slots=True)
class IngestionDocument:
    """Document input to ingestion service within a bounded tenant context."""

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Information regarding a generated and indexed chunk."""

    chunk_id: str
    document_id: str
    index: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Result of an ingestion operation."""

    tenant_id: str
    documents_processed: int
    chunks_created: int
    upserted_count: int
    chunk_records: list[ChunkRecord] = field(default_factory=list)


def build_tenant_document_id(
    tenant_id: str,
    raw_document_id: str,
    chunk_index: int | None = None,
    chunk_hash: str | None = None,
) -> str:
    """Construct a stable, deterministic tenant-scoped document ID.

    Format:
        If chunk_index is None:
            t:{tenant_hash}:{raw_document_id}
        If chunk_index is provided:
            t:{tenant_hash}:{raw_document_id}:c{chunk_index}[:{chunk_hash[:8]}]
    """
    valid_tenant = validate_tenant_scope(tenant_id)
    if not isinstance(raw_document_id, str):
        raise RetrievalContractError(
            f"document_id must be a string, got {type(raw_document_id).__name__}"
        )
    raw_doc_id = raw_document_id.strip()
    if not raw_doc_id:
        raise RetrievalContractError("document_id cannot be empty or whitespace-only")
    if len(raw_doc_id) > MAX_DOCUMENT_ID_LENGTH:
        raise RetrievalContractError(
            f"document_id exceeds maximum length of {MAX_DOCUMENT_ID_LENGTH} characters"
        )
    if not _DOC_ID_ALLOWED_RE.match(raw_doc_id):
        raise RetrievalContractError(f"document_id contains disallowed characters: '{raw_doc_id}'")

    # Deterministic short tenant hash (12 chars hex)
    t_hash = hashlib.sha256(valid_tenant.encode("utf-8")).hexdigest()[:12]

    if chunk_index is None:
        return f"t_{t_hash}_{raw_doc_id}"

    if chunk_index < 0:
        raise RetrievalContractError(f"chunk_index cannot be negative, got {chunk_index}")

    base = f"t_{t_hash}_{raw_doc_id}_c{chunk_index}"
    if chunk_hash:
        clean_hash = chunk_hash[:8]
        return f"{base}_{clean_hash}"
    return base


def sanitize_and_validate_metadata(
    metadata: dict[str, Any],
    max_depth: int = MAX_METADATA_DEPTH,
    max_size: int = MAX_METADATA_SIZE_BYTES,
) -> dict[str, Any]:
    """Validate and copy metadata ensuring bounding and no forbidden nesting/size."""
    if not isinstance(metadata, dict):
        raise RetrievalContractError(
            f"metadata must be a dictionary, got {type(metadata).__name__}"
        )

    def _check_depth(obj: Any, depth: int) -> None:
        if depth > max_depth:
            raise RetrievalContractError(
                f"Metadata nesting depth exceeds maximum allowed limit ({max_depth})"
            )
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise RetrievalContractError(
                        f"Metadata key must be a string, got {type(k).__name__}"
                    )
                if isinstance(v, (dict, list)):
                    _check_depth(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _check_depth(item, depth + 1)

    _check_depth(metadata, 1)

    try:
        encoded = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as err:
        raise RetrievalContractError(f"Metadata is not JSON-serializable: {err}") from err

    if len(encoded.encode("utf-8")) > max_size:
        raise RetrievalContractError(
            f"Metadata size ({len(encoded.encode('utf-8'))} bytes) exceeds maximum limit ({max_size} bytes)"
        )

    return json.loads(encoded)


def validate_query_filter_bounds(filter_metadata: dict[str, Any]) -> None:
    """Validate that query filter metadata does not contain malicious or oversized keys/structures."""
    if not isinstance(filter_metadata, dict):
        raise RetrievalContractError(
            f"filter_metadata must be a dict, got {type(filter_metadata).__name__}"
        )
    if len(filter_metadata) > 20:
        raise RetrievalContractError("filter_metadata exceeds maximum allowed filter keys (20)")
    for key, val in filter_metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise RetrievalContractError("filter_metadata keys must be non-empty strings")
        if len(key) > 64:
            raise RetrievalContractError(f"filter_metadata key '{key}' exceeds maximum length 64")
        if isinstance(val, (dict, list)):
            if isinstance(val, list) and len(val) > 100:
                raise RetrievalContractError("filter_metadata list value exceeds max items (100)")
            if isinstance(val, dict):
                # Only simple operator dicts allowed, depth 1
                for sub_k in val:
                    if not isinstance(sub_k, str) or len(sub_k) > 32:
                        raise RetrievalContractError("Nested filter operator key invalid")


def _is_zero_vector(vec: tuple[float, ...] | list[float], eps: float = 1e-12) -> bool:
    """Check if vector is zero (norm close to 0) which causes degenerate cosine distances."""
    norm_sq = sum(x * x for x in vec)
    return norm_sq < eps


class IngestionService:
    """Deterministic, injected EmbeddingProvider + VectorIndex orchestration service.

    Guarantees:
    - Multi-tenant isolation: caller must provide authenticated tenant_id.
    - Strict input bounds: maximum document length, batch sizing, chunk count.
    - Embeddings validation: vector dimension, finite values, non-zero cosine safety.
    - Deterministic chunk IDs and stable tenant document IDs.
    - Safe query helper with top_k clamping and tenant scoping.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        config: IngestionConfig | None = None,
    ) -> None:
        if embedding_provider is None:
            raise RetrievalContractError(
                "embedding_provider must be provided via dependency injection"
            )
        if vector_index is None:
            raise RetrievalContractError("vector_index must be provided via dependency injection")

        self._embedding_provider = embedding_provider
        self._vector_index = vector_index
        self._config = config or IngestionConfig()

    @property
    def config(self) -> IngestionConfig:
        return self._config

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider

    @property
    def vector_index(self) -> VectorIndex:
        return self._vector_index

    def ingest_documents(
        self,
        tenant_id: str,
        documents: list[IngestionDocument],
    ) -> IngestionResult:
        """Ingest, chunk, embed, and idempotently upsert documents for a specific tenant.

        Args:
            tenant_id: Verified tenant identifier.
            documents: List of IngestionDocument instances.

        Returns:
            IngestionResult summarizing processed documents and upserted vectors.

        Raises:
            RetrievalContractError: If boundaries, formats, dimensions, or tenant isolation are violated.
        """
        valid_tenant = validate_tenant_scope(tenant_id)

        if not isinstance(documents, list):
            raise RetrievalContractError(
                f"documents must be a list, got {type(documents).__name__}"
            )

        if len(documents) > self._config.max_documents_per_batch:
            raise RetrievalContractError(
                f"Document batch count ({len(documents)}) exceeds limit ({self._config.max_documents_per_batch})"
            )

        if not documents:
            return IngestionResult(
                tenant_id=valid_tenant,
                documents_processed=0,
                chunks_created=0,
                upserted_count=0,
                chunk_records=[],
            )

        # 1. Validation & Chunking
        seen_doc_ids: set[str] = set()
        all_chunks: list[
            tuple[str, TextChunk, dict[str, Any]]
        ] = []  # (raw_doc_id, chunk, safe_metadata)

        for doc in documents:
            if not isinstance(doc, IngestionDocument):
                raise RetrievalContractError(
                    f"Expected IngestionDocument instance, got {type(doc).__name__}"
                )
            if not doc.document_id or not isinstance(doc.document_id, str):
                raise RetrievalContractError(
                    "IngestionDocument document_id must be a non-empty string"
                )

            clean_doc_id = doc.document_id.strip()
            if clean_doc_id in seen_doc_ids:
                raise RetrievalContractError(f"Duplicate document_id in batch: '{clean_doc_id}'")
            seen_doc_ids.add(clean_doc_id)

            if len(clean_doc_id) > MAX_DOCUMENT_ID_LENGTH or not _DOC_ID_ALLOWED_RE.match(
                clean_doc_id
            ):
                raise RetrievalContractError(
                    f"Invalid or disallowed characters in document_id '{clean_doc_id}'"
                )

            if not isinstance(doc.text, str):
                raise RetrievalContractError(
                    f"Document text must be string, got {type(doc.text).__name__}"
                )

            if len(doc.text) > MAX_TEXT_CONTENT_LENGTH:
                raise RetrievalContractError(
                    f"Document text length ({len(doc.text)}) exceeds limit ({MAX_TEXT_CONTENT_LENGTH})"
                )

            safe_meta = sanitize_and_validate_metadata(doc.metadata or {})

            # Chunk document
            chunks = chunk_text(
                text=doc.text,
                config=self._config.chunking_config,
                metadata=safe_meta,
                document_id=clean_doc_id,
            )

            for chunk in chunks:
                all_chunks.append((clean_doc_id, chunk, safe_meta))

        if len(all_chunks) > self._config.max_chunks_per_ingest:
            raise RetrievalContractError(
                f"Total chunks created ({len(all_chunks)}) exceeds safety limit ({self._config.max_chunks_per_ingest})"
            )

        if not all_chunks:
            return IngestionResult(
                tenant_id=valid_tenant,
                documents_processed=len(documents),
                chunks_created=0,
                upserted_count=0,
                chunk_records=[],
            )

        # 2. Embedding in batches & VectorDocument creation
        chunk_texts = [item[1].text for item in all_chunks]
        embeddings: list[list[float]] = []
        batch_size = self._config.batch_size

        for i in range(0, len(chunk_texts), batch_size):
            batch_slice = chunk_texts[i : i + batch_size]
            try:
                batch_vectors = self._embedding_provider.embed_texts(batch_slice)
            except Exception as err:
                raise RetrievalError(
                    "Embedding provider call failed during batch processing"
                ) from err

            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch_slice):
                raise RetrievalContractError(
                    f"Embedding provider returned {len(batch_vectors) if isinstance(batch_vectors, list) else 'invalid'} "
                    f"vectors, expected {len(batch_slice)}"
                )

            for vec in batch_vectors:
                validate_embedding_vector(vec, expected_dimension=self._config.expected_dimension)
                if _is_zero_vector(vec):
                    raise RetrievalContractError(
                        "Embedding vector has zero norm (invalid for cosine similarity)"
                    )
                embeddings.append(vec)

        # 3. Build VectorDocuments
        vector_docs: list[VectorDocument] = []
        chunk_records: list[ChunkRecord] = []

        for (clean_doc_id, chunk, safe_meta), vec in zip(all_chunks, embeddings, strict=True):
            stable_doc_id = build_tenant_document_id(
                tenant_id=valid_tenant,
                raw_document_id=clean_doc_id,
                chunk_index=chunk.index,
                chunk_hash=chunk.chunk_id,
            )

            # Preserve safe metadata with provenance
            doc_meta = dict(safe_meta)
            doc_meta["_provenance"] = {
                "source_doc_id": clean_doc_id,
                "chunk_index": chunk.index,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "chunk_id": chunk.chunk_id,
            }

            v_doc = VectorDocument(
                id=stable_doc_id,
                vector=tuple(vec),
                tenant_id=valid_tenant,
                content=chunk.text,
                metadata=doc_meta,
            )
            vector_docs.append(v_doc)
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=clean_doc_id,
                    index=chunk.index,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                )
            )

        validate_vector_documents(
            documents=vector_docs,
            expected_tenant_id=valid_tenant,
            expected_dimension=self._config.expected_dimension,
        )

        # 4. Upsert into VectorIndex
        upserted_count = self._vector_index.upsert(
            tenant_id=valid_tenant,
            documents=vector_docs,
        )

        return IngestionResult(
            tenant_id=valid_tenant,
            documents_processed=len(documents),
            chunks_created=len(all_chunks),
            upserted_count=upserted_count,
            chunk_records=chunk_records,
        )

    def query(
        self,
        tenant_id: str,
        query_text: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query the vector index with text by generating embeddings and executing a tenant-safe search.

        Args:
            tenant_id: Verified tenant identifier.
            query_text: Raw query text.
            top_k: Number of results to retrieve.
            score_threshold: Optional similarity score threshold.
            filter_metadata: Optional key-value filter metadata.

        Returns:
            List of SearchResult items verified for tenant isolation and sorting.

        Raises:
            RetrievalContractError: If query parameters, filters, or returned results violate contracts.
        """
        valid_tenant = validate_tenant_scope(tenant_id)

        if not isinstance(query_text, str) or not query_text.strip():
            raise RetrievalContractError("query_text must be a non-empty string")

        if len(query_text) > MAX_TEXT_CONTENT_LENGTH:
            raise RetrievalContractError(
                f"query_text length ({len(query_text)}) exceeds maximum allowed limit"
            )

        if top_k < 1:
            raise RetrievalContractError(f"top_k must be >= 1, got {top_k}")

        bounded_top_k = min(top_k, self._config.max_top_k)

        if filter_metadata is not None:
            validate_query_filter_bounds(filter_metadata)
            safe_filters = sanitize_and_validate_metadata(filter_metadata)
        else:
            safe_filters = {}

        # Generate query embedding
        try:
            vectors = self._embedding_provider.embed_texts([query_text])
        except Exception as err:
            raise RetrievalError("Failed to generate embedding for query") from err

        if not isinstance(vectors, list) or len(vectors) != 1:
            raise RetrievalContractError("Embedding provider returned invalid vectors for query")

        q_vec = vectors[0]
        validate_embedding_vector(q_vec, expected_dimension=self._config.expected_dimension)
        if _is_zero_vector(q_vec):
            raise RetrievalContractError("Query embedding vector has zero norm")

        bundle = QueryBundle(
            tenant_id=valid_tenant,
            query_text=query_text,
            query_vector=tuple(q_vec),
            top_k=bounded_top_k,
            score_threshold=score_threshold,
            filter_metadata=safe_filters,
        )

        results = self._vector_index.search(bundle)

        if not isinstance(results, list):
            raise RetrievalContractError(
                f"VectorIndex search returned {type(results).__name__}, expected list"
            )

        # Validate that results strictly adhere to caller's tenant_id and invariants
        validate_search_results(results, expected_tenant_id=valid_tenant)

        return results
