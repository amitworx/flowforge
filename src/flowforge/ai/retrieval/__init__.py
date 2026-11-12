"""Retrieval foundation contracts, protocols, and deterministic text chunking."""

from flowforge.ai.retrieval.chunking import (
    ChunkingConfig,
    TextChunk,
    chunk_text,
    compute_chunk_id,
)
from flowforge.ai.retrieval.exceptions import (
    ChunkingError,
    RetrievalContractError,
    RetrievalError,
)
from flowforge.ai.retrieval.protocols import (
    EmbeddingProvider,
    QueryBundle,
    SearchResult,
    VectorDocument,
    VectorIndex,
)
from flowforge.ai.retrieval.validation import (
    MAX_SAFE_INPUT_LENGTH,
    MAX_SAFE_VECTOR_DIMENSION,
    validate_chunk_bounds,
    validate_embedding_vector,
    validate_query_bundle,
    validate_search_results,
    validate_tenant_scope,
    validate_vector_documents,
)

__all__ = [
    "ChunkingConfig",
    "ChunkingError",
    "EmbeddingProvider",
    "MAX_SAFE_INPUT_LENGTH",
    "MAX_SAFE_VECTOR_DIMENSION",
    "QueryBundle",
    "RetrievalContractError",
    "RetrievalError",
    "SearchResult",
    "TextChunk",
    "VectorDocument",
    "VectorIndex",
    "chunk_text",
    "compute_chunk_id",
    "validate_chunk_bounds",
    "validate_embedding_vector",
    "validate_query_bundle",
    "validate_search_results",
    "validate_tenant_scope",
    "validate_vector_documents",
]
