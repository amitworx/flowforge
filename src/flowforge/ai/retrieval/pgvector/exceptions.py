"""Exceptions for pgvector retrieval adapter."""

from __future__ import annotations

from flowforge.ai.retrieval.exceptions import RetrievalError


class PgVectorError(RetrievalError):
    """Base exception for all pgvector adapter operations."""


class PgVectorExtensionUnavailableError(PgVectorError):
    """Raised when pgvector extension is absent or cannot be loaded."""


class PgVectorFilterError(PgVectorError):
    """Raised when an unsupported or invalid metadata filter expression is provided."""
