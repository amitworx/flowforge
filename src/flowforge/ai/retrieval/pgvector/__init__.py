"""PostgreSQL + pgvector tenant-isolated vector store adapter."""

from __future__ import annotations

from flowforge.ai.retrieval.pgvector.adapter import PgVectorIndex
from flowforge.ai.retrieval.pgvector.config import DistanceMetric, PgVectorConfig
from flowforge.ai.retrieval.pgvector.exceptions import (
    PgVectorError,
    PgVectorExtensionUnavailableError,
    PgVectorFilterError,
)
from flowforge.ai.retrieval.pgvector.filters import compile_metadata_filters
from flowforge.ai.retrieval.pgvector.schema import (
    CHECK_EXTENSION_SQL,
    CREATE_EXTENSION_SQL,
    SCHEMA_DDL,
    TABLE_NAME_DEFAULT,
)

__all__ = [
    "CHECK_EXTENSION_SQL",
    "CREATE_EXTENSION_SQL",
    "DistanceMetric",
    "PgVectorConfig",
    "PgVectorError",
    "PgVectorExtensionUnavailableError",
    "PgVectorFilterError",
    "PgVectorIndex",
    "SCHEMA_DDL",
    "TABLE_NAME_DEFAULT",
    "compile_metadata_filters",
]
