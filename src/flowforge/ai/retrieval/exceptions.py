"""Exceptions for AI retrieval contracts and chunking."""


class RetrievalError(Exception):
    """Base exception for all retrieval module errors."""


class RetrievalContractError(RetrievalError):
    """Raised when a retrieval protocol contract, tenant scope, or vector invariant is violated."""


class ChunkingError(RetrievalError):
    """Raised when text chunking parameters are invalid or processing fails safety bounds."""
