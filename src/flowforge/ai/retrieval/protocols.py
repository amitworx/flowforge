"""Retrieval protocol definitions and data contracts.

These contracts provide stable dependency-injection interfaces for vector indices
and embedding providers without requiring live vector databases or network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class VectorDocument:
    """Represents a document embedded and ready to be upserted to a vector index.

    Attributes:
        id: Unique identifier for the document within the tenant scope.
        vector: Sequence of float embedding dimensions.
        tenant_id: Scope identifier enforcing multi-tenant isolation.
        content: Original raw or chunked text content.
        metadata: Arbitrary deterministic metadata associated with document.
    """

    id: str
    vector: tuple[float, ...]
    tenant_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.vector, list):
            # Ensure vector is converted to immutable tuple
            object.__setattr__(self, "vector", tuple(self.vector))


@dataclass(frozen=True, slots=True)
class QueryBundle:
    """Represents a vector or text query request scoped to a specific tenant.

    Attributes:
        tenant_id: Scope identifier enforcing multi-tenant isolation.
        query_text: Optional raw text query.
        query_vector: Optional query embedding vector.
        top_k: Number of most similar results to return (>= 1).
        score_threshold: Optional minimum similarity score threshold.
        filter_metadata: Optional key-value metadata filters.
    """

    tenant_id: str
    query_text: str | None = None
    query_vector: tuple[float, ...] | None = None
    top_k: int = 10
    score_threshold: float | None = None
    filter_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.query_vector, list):
            object.__setattr__(self, "query_vector", tuple(self.query_vector))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Represents a retrieved search result record.

    Attributes:
        id: Document ID.
        score: Normalized similarity score (higher is more similar).
        content: Document content string.
        tenant_id: Scope identifier verifying tenant safety.
        metadata: Document metadata.
    """

    id: str
    score: float
    content: str
    tenant_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of input text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vector embeddings corresponding to each input text.
        """
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for tenant-scoped vector search indices."""

    def upsert(
        self,
        tenant_id: str,
        documents: list[VectorDocument],
    ) -> int:
        """Upsert vector documents into the index under the given tenant scope.

        Args:
            tenant_id: Tenant scope identifier.
            documents: List of VectorDocument items to index.

        Returns:
            Number of documents successfully upserted.
        """
        ...

    def search(
        self,
        query: QueryBundle,
    ) -> list[SearchResult]:
        """Search the index for vector documents matching the query bundle.

        Args:
            query: QueryBundle specifying tenant scope, vector/text, and parameters.

        Returns:
            List of SearchResult items sorted descending by similarity score.
        """
        ...
