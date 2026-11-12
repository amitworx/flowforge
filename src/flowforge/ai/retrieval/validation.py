"""Validation helpers enforcing tenant isolation, vector invariants, and data safety bounds."""

from __future__ import annotations

import math
from collections.abc import Sequence

from flowforge.ai.retrieval.exceptions import ChunkingError, RetrievalContractError
from flowforge.ai.retrieval.protocols import QueryBundle, SearchResult, VectorDocument

MAX_SAFE_INPUT_LENGTH: int = 10_000_000  # 10 MB text safeguard
MAX_SAFE_VECTOR_DIMENSION: int = 16_384  # Maximum reasonable embedding dimension


def validate_tenant_scope(tenant_id: str | None) -> str:
    """Validate that tenant_id is a non-empty, non-whitespace string.

    Args:
        tenant_id: The tenant identifier to validate.

    Returns:
        The stripped valid tenant identifier.

    Raises:
        RetrievalContractError: If tenant_id is invalid or empty.
    """
    if not isinstance(tenant_id, str):
        raise RetrievalContractError(f"tenant_id must be a string, got {type(tenant_id).__name__}")
    stripped = tenant_id.strip()
    if not stripped:
        raise RetrievalContractError("tenant_id cannot be empty or whitespace-only")
    return stripped


def validate_embedding_vector(
    vector: Sequence[float],
    expected_dimension: int | None = None,
) -> None:
    """Validate that vector consists of finite floats matching dimension constraints.

    Args:
        vector: Sequence of floats.
        expected_dimension: Optional dimension to enforce.

    Raises:
        RetrievalContractError: If vector is empty, contains non-numbers, non-finite values,
                                or fails dimension requirements.
    """
    if vector is None or len(vector) == 0:
        raise RetrievalContractError("Vector cannot be empty or None")

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise RetrievalContractError(
            f"Expected vector dimension {expected_dimension}, got {len(vector)}"
        )

    if len(vector) > MAX_SAFE_VECTOR_DIMENSION:
        raise RetrievalContractError(
            f"Vector dimension {len(vector)} exceeds maximum limit {MAX_SAFE_VECTOR_DIMENSION}"
        )

    for i, val in enumerate(vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise RetrievalContractError(
                f"Vector element at index {i} must be float or int, got {type(val).__name__}"
            )
        if not math.isfinite(val):
            raise RetrievalContractError(
                f"Vector contains non-finite value (NaN or Inf) at index {i}: {val}"
            )


def validate_vector_documents(
    documents: list[VectorDocument],
    expected_tenant_id: str,
    expected_dimension: int | None = None,
) -> None:
    """Validate a batch of VectorDocument instances for tenant isolation and vector validity.

    Args:
        documents: List of VectorDocument items.
        expected_tenant_id: Tenant id that all documents must belong to.
        expected_dimension: Optional embedding vector dimension requirement.

    Raises:
        RetrievalContractError: If documents violate tenant boundaries, contain duplicates,
                                or contain invalid vectors.
    """
    scope = validate_tenant_scope(expected_tenant_id)
    seen_ids: set[str] = set()

    for doc in documents:
        if not isinstance(doc, VectorDocument):
            raise RetrievalContractError(
                f"Expected VectorDocument instance, got {type(doc).__name__}"
            )

        if not doc.id or not isinstance(doc.id, str):
            raise RetrievalContractError("VectorDocument id must be a non-empty string")

        if doc.id in seen_ids:
            raise RetrievalContractError(f"Duplicate document id found in batch: '{doc.id}'")
        seen_ids.add(doc.id)

        doc_tenant = validate_tenant_scope(doc.tenant_id)
        if doc_tenant != scope:
            raise RetrievalContractError(
                f"Tenant isolation violation: document '{doc.id}' belongs to tenant '{doc_tenant}', "
                f"expected '{scope}'"
            )

        validate_embedding_vector(doc.vector, expected_dimension=expected_dimension)


def validate_query_bundle(query: QueryBundle) -> None:
    """Validate parameters of a QueryBundle.

    Args:
        query: QueryBundle instance to validate.

    Raises:
        RetrievalContractError: If query bundle parameters are invalid.
    """
    if not isinstance(query, QueryBundle):
        raise RetrievalContractError(f"Expected QueryBundle instance, got {type(query).__name__}")

    validate_tenant_scope(query.tenant_id)

    if query.top_k < 1:
        raise RetrievalContractError(f"top_k must be greater than or equal to 1, got {query.top_k}")

    if query.score_threshold is not None:
        if not isinstance(query.score_threshold, (int, float)) or isinstance(
            query.score_threshold, bool
        ):
            raise RetrievalContractError("score_threshold must be a numeric value")
        if not math.isfinite(query.score_threshold):
            raise RetrievalContractError("score_threshold must be a finite float")

    if query.query_vector is not None:
        validate_embedding_vector(query.query_vector)


def validate_search_results(
    results: list[SearchResult],
    expected_tenant_id: str,
) -> None:
    """Validate search results for tenant isolation, score finiteness, and descending ordering.

    Args:
        results: List of SearchResult items.
        expected_tenant_id: Expected tenant scope.

    Raises:
        RetrievalContractError: If results violate ordering, finiteness, or tenant scope.
    """
    scope = validate_tenant_scope(expected_tenant_id)
    prev_score: float | None = None

    for idx, res in enumerate(results):
        if not isinstance(res, SearchResult):
            raise RetrievalContractError(
                f"Expected SearchResult instance, got {type(res).__name__}"
            )

        res_tenant = validate_tenant_scope(res.tenant_id)
        if res_tenant != scope:
            raise RetrievalContractError(
                f"Search result belongs to foreign tenant '{res_tenant}', expected '{scope}'"
            )

        if (
            not isinstance(res.score, (int, float))
            or isinstance(res.score, bool)
            or not math.isfinite(res.score)
        ):
            raise RetrievalContractError(
                f"Search result '{res.id}' at index {idx} has non-finite score: {res.score}"
            )

        if prev_score is not None and res.score > prev_score:
            raise RetrievalContractError(
                f"Search results must be sorted descending by score: result {idx} score {res.score} "
                f"is greater than previous score {prev_score}"
            )
        prev_score = float(res.score)


def validate_chunk_bounds(
    text: str,
    max_input_length: int = MAX_SAFE_INPUT_LENGTH,
) -> None:
    """Validate that input text does not exceed memory safety limits.

    Args:
        text: Input text string.
        max_input_length: Maximum allowed characters.

    Raises:
        ChunkingError: If text length exceeds limit.
    """
    if text is not None and len(text) > max_input_length:
        raise ChunkingError(
            f"Input text length exceeds maximum allowed limit ({len(text)} > {max_input_length})"
        )
