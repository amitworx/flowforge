"""Deterministic, tenant-safe embedding ingestion and retrieval orchestration core."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from flowforge.ai.retrieval.chunking import ChunkingConfig, TextChunk, chunk_text
from flowforge.ai.retrieval.exceptions import RetrievalContractError, RetrievalError
from flowforge.ai.retrieval.pgvector.config import (
    MAX_BATCH_SIZE,
    MAX_FILTER_IN_ITEMS,
    MAX_METADATA_DEPTH,
    MAX_METADATA_SIZE_BYTES,
    MAX_PGVECTOR_DIMENSION,
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
_FILTER_KEY_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\.]{0,63}$")
_SUPPORTED_FILTER_OPERATORS = {"$eq", "$in", "$ne"}


def _validate_strict_int(
    val: Any, field_name: str, min_val: int = 1, max_val: int | None = None
) -> int:
    """Validate that val is a strict int (not bool, float, or str) within [min_val, max_val]."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise RetrievalContractError(
            f"{field_name} must be a strict integer, got {type(val).__name__}"
        )
    if val < min_val:
        raise RetrievalContractError(f"{field_name} must be >= {min_val}, got {val}")
    if max_val is not None and val > max_val:
        raise RetrievalContractError(f"{field_name} must be <= {max_val}, got {val}")
    return val


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
        _validate_strict_int(self.batch_size, "batch_size", min_val=1, max_val=MAX_BATCH_SIZE)
        _validate_strict_int(self.max_top_k, "max_top_k", min_val=1, max_val=MAX_TOP_K)
        _validate_strict_int(
            self.max_documents_per_batch,
            "max_documents_per_batch",
            min_val=1,
            max_val=MAX_DOCUMENTS_PER_BATCH,
        )
        _validate_strict_int(
            self.max_chunks_per_ingest,
            "max_chunks_per_ingest",
            min_val=1,
            max_val=MAX_CHUNKS_PER_INGEST,
        )

        if self.expected_dimension is not None:
            _validate_strict_int(
                self.expected_dimension,
                "expected_dimension",
                min_val=1,
                max_val=MAX_PGVECTOR_DIMENSION,
            )

        if not isinstance(self.chunking_config, ChunkingConfig):
            raise RetrievalContractError("chunking_config must be an instance of ChunkingConfig")


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
            t_{tenant_hash}_{raw_document_id}
        If chunk_index is provided:
            t_{tenant_hash}_{raw_document_id}_c{chunk_index}[_{chunk_hash[:8]}]
    """
    valid_tenant = validate_tenant_scope(tenant_id)
    if not isinstance(raw_document_id, str):
        raise RetrievalContractError("document_id must be a string")
    raw_doc_id = raw_document_id.strip()
    if not raw_doc_id:
        raise RetrievalContractError("document_id cannot be empty or whitespace-only")
    if len(raw_doc_id) > MAX_DOCUMENT_ID_LENGTH:
        raise RetrievalContractError("document_id exceeds maximum length limit")
    if not _DOC_ID_ALLOWED_RE.match(raw_doc_id):
        raise RetrievalContractError("document_id contains disallowed characters")

    # Deterministic short tenant hash (12 chars hex)
    t_hash = hashlib.sha256(valid_tenant.encode("utf-8")).hexdigest()[:12]

    if chunk_index is None:
        return f"t_{t_hash}_{raw_doc_id}"

    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise RetrievalContractError("chunk_index must be a non-negative integer")

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
        raise RetrievalContractError("metadata must be a dictionary")

    def _check_depth(obj: Any, depth: int) -> None:
        if depth > max_depth:
            raise RetrievalContractError("Metadata nesting depth exceeds maximum allowed limit")
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise RetrievalContractError("Metadata key must be a string")
                if isinstance(v, (dict, list)):
                    _check_depth(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _check_depth(item, depth + 1)

    _check_depth(metadata, 1)

    try:
        encoded = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        raise RetrievalContractError("Metadata is not JSON-serializable") from None

    if len(encoded.encode("utf-8")) > max_size:
        raise RetrievalContractError("Metadata size exceeds maximum limit")

    return json.loads(encoded)


def validate_query_filter_bounds(filter_metadata: dict[str, Any]) -> None:
    """Validate that query filter metadata matches safe pgvector adapter semantics and bounds."""
    if not isinstance(filter_metadata, dict):
        raise RetrievalContractError("filter_metadata must be a dictionary")
    if len(filter_metadata) > 20:
        raise RetrievalContractError("filter_metadata exceeds maximum allowed filter keys")
    for key, val in filter_metadata.items():
        if not isinstance(key, str) or not _FILTER_KEY_PATTERN.match(key):
            raise RetrievalContractError("Invalid filter metadata key format")
        if isinstance(val, dict):
            # Operator dict: only $eq, $ne, $in
            for op, op_val in val.items():
                if op not in _SUPPORTED_FILTER_OPERATORS:
                    raise RetrievalContractError("Unsupported filter operator in metadata query")
                if op == "$in":
                    if not isinstance(op_val, (list, tuple, set)):
                        raise RetrievalContractError("$in operator expects a list")
                    if len(op_val) > MAX_FILTER_IN_ITEMS:
                        raise RetrievalContractError("$in operator exceeds maximum allowed items")
                elif op in ("$eq", "$ne") and isinstance(op_val, (dict, list)):
                    raise RetrievalContractError("Nested complex value not supported for operator")
        elif isinstance(val, list):
            raise RetrievalContractError("Direct list value not supported without operator")


def _is_zero_vector(vec: tuple[float, ...] | list[float]) -> bool:
    """Check if vector is an exact zero vector (all elements equal 0.0).

    Does not falsely reject tiny valid finite embeddings with small norms.
    """
    return all(x == 0.0 for x in vec)


class IngestionService:
    """Deterministic, injected EmbeddingProvider + VectorIndex orchestration service.

    Authorization Contract:
    - Caller tenant identity MUST be authenticated and resolved by the caller.
    - This service does not perform standalone caller authentication; it enforces multi-tenant
      isolation based on the verified tenant_id parameter provided by the caller.
    - Fails closed on cross-tenant search responses or mismatched tenant documents.
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
            tenant_id: Verified caller tenant identifier.
            documents: List of IngestionDocument instances.

        Returns:
            IngestionResult summarizing processed documents and upserted vectors.

        Raises:
            RetrievalContractError: If boundaries, formats, dimensions, or tenant isolation are violated.
            RetrievalError: Generic unchained error if embedding provider call fails.
        """
        valid_tenant = validate_tenant_scope(tenant_id)

        if not isinstance(documents, list):
            raise RetrievalContractError("documents must be a list")

        if len(documents) > self._config.max_documents_per_batch:
            raise RetrievalContractError("Document batch count exceeds limit")

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
        all_chunks: list[tuple[str, TextChunk, dict[str, Any]]] = []

        for doc in documents:
            if not isinstance(doc, IngestionDocument):
                raise RetrievalContractError("Expected IngestionDocument instance")
            if not doc.document_id or not isinstance(doc.document_id, str):
                raise RetrievalContractError(
                    "IngestionDocument document_id must be a non-empty string"
                )

            clean_doc_id = doc.document_id.strip()
            if clean_doc_id in seen_doc_ids:
                raise RetrievalContractError("Duplicate document_id in batch")
            seen_doc_ids.add(clean_doc_id)

            if len(clean_doc_id) > MAX_DOCUMENT_ID_LENGTH or not _DOC_ID_ALLOWED_RE.match(
                clean_doc_id
            ):
                raise RetrievalContractError("Invalid or disallowed characters in document_id")

            if not isinstance(doc.text, str):
                raise RetrievalContractError("Document text must be a string")

            if len(doc.text) > MAX_TEXT_CONTENT_LENGTH:
                raise RetrievalContractError("Document text length exceeds limit")

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
            raise RetrievalContractError("Total chunks created exceeds safety limit")

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
            except Exception:
                # Privacy: fail closed with generic error unchained from provider internals
                raise RetrievalError(
                    "Embedding provider call failed during batch processing"
                ) from None

            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch_slice):
                raise RetrievalContractError("Embedding provider returned mismatched vector count")

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

            # Validate metadata including provenance
            sanitize_and_validate_metadata(doc_meta)

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
            tenant_id: Verified caller tenant identifier.
            query_text: Raw query text.
            top_k: Number of results to retrieve (must be strict positive integer).
            score_threshold: Optional similarity score threshold (must be finite float/int).
            filter_metadata: Optional key-value filter metadata matching pgvector semantics.

        Returns:
            List of SearchResult items verified for tenant isolation and sorting.

        Raises:
            RetrievalContractError: If query parameters, filters, or returned results violate contracts.
            RetrievalError: Generic unchained error if embedding generation fails.
        """
        valid_tenant = validate_tenant_scope(tenant_id)

        if not isinstance(query_text, str) or not query_text.strip():
            raise RetrievalContractError("query_text must be a non-empty string")

        if len(query_text) > MAX_TEXT_CONTENT_LENGTH:
            raise RetrievalContractError("query_text length exceeds maximum allowed limit")

        # Validate top_k strictly
        _validate_strict_int(top_k, "top_k", min_val=1)
        bounded_top_k = min(top_k, self._config.max_top_k)

        # Validate score_threshold strictly before any external calls
        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
                raise RetrievalContractError("score_threshold must be a numeric value")
            if not math.isfinite(score_threshold):
                raise RetrievalContractError("score_threshold must be a finite float")

        # Validate filter_metadata before external calls
        if filter_metadata is not None:
            validate_query_filter_bounds(filter_metadata)
            safe_filters = sanitize_and_validate_metadata(filter_metadata)
        else:
            safe_filters = {}

        # Generate query embedding
        try:
            vectors = self._embedding_provider.embed_texts([query_text])
        except Exception:
            # Privacy: unchain provider error to prevent secret/prompt leaks
            raise RetrievalError("Failed to generate embedding for query") from None

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
            raise RetrievalContractError("VectorIndex search returned invalid result type")

        # Validate that results strictly adhere to caller's tenant_id and invariants
        validate_search_results(results, expected_tenant_id=valid_tenant)

        return results
