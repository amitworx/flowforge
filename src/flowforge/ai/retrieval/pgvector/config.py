"""Configuration options and metric definitions for pgvector adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from flowforge.ai.retrieval.exceptions import RetrievalContractError

_TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")

# pgvector standard vector(n) upper bound
MAX_PGVECTOR_DIMENSION: int = 2_000
MAX_BATCH_SIZE: int = 1_000
MAX_TOP_K: int = 1_000
MAX_FILTER_IN_ITEMS: int = 100
MAX_METADATA_SIZE_BYTES: int = 65_536  # 64 KB metadata limit
MAX_METADATA_DEPTH: int = 4
MAX_TEXT_CONTENT_LENGTH: int = 1_000_000  # 1 MB text length limit


class DistanceMetric(StrEnum):
    """Supported distance metrics for pgvector similarity search."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"

    @property
    def operator(self) -> str:
        """PostgreSQL operator corresponding to the metric."""
        if self == DistanceMetric.COSINE:
            return "<=>"
        if self == DistanceMetric.L2:
            return "<->"
        if self == DistanceMetric.INNER_PRODUCT:
            return "<#>"
        raise ValueError(f"Unknown metric: {self}")

    def score_from_distance(self, distance: float) -> float:
        """Convert PostgreSQL raw distance to a normalized similarity score (higher is better).

        - Cosine: distance in [0, 2]. Score: 1.0 - distance.
        - L2: distance in [0, inf). Score: 1.0 / (1.0 + distance).
        - Inner Product: pgvector returns negative dot product (<#> = - (x . y)).
          Raw distance is -dot_product. So score is -distance (i.e. dot_product).
        """
        if self == DistanceMetric.COSINE:
            return 1.0 - distance
        if self == DistanceMetric.L2:
            return 1.0 / (1.0 + distance)
        if self == DistanceMetric.INNER_PRODUCT:
            return -distance
        raise ValueError(f"Unknown metric: {self}")


@dataclass(frozen=True, slots=True)
class PgVectorConfig:
    """Configuration for PgVectorIndex.

    Attributes:
        table_name: Qualified table name (alphanumeric and underscores only, max 63 chars).
        dimension: Expected embedding vector dimension (1 to 2,000 for standard pgvector vector(n)).
        metric: Distance metric to use for similarity ranking.
        create_extension: Whether to attempt 'CREATE EXTENSION IF NOT EXISTS vector' during bootstrap.
                         Defaults to False for least privilege.
        auto_bootstrap: Whether to automatically bootstrap schema on first operation.
                        Defaults to False for least privilege.
    """

    table_name: str = "flowforge_ai_vector_documents"
    dimension: int = 1536
    metric: DistanceMetric = DistanceMetric.COSINE
    create_extension: bool = False
    auto_bootstrap: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not _TABLE_NAME_PATTERN.match(self.table_name):
            raise RetrievalContractError(
                f"Invalid table name '{self.table_name}'. Must be a valid SQL identifier (alphanumeric + underscore, <= 63 chars)."
            )
        # Disallow bool (since bool is a subclass of int in Python: isinstance(True, int) == True)
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise RetrievalContractError(
                f"dimension must be an integer, got {type(self.dimension).__name__}"
            )
        if self.dimension < 1 or self.dimension > MAX_PGVECTOR_DIMENSION:
            raise RetrievalContractError(
                f"Invalid dimension {self.dimension}. Must be an integer between 1 and {MAX_PGVECTOR_DIMENSION:,} for pgvector vector(n)."
            )
        if isinstance(self.metric, str):
            try:
                object.__setattr__(self, "metric", DistanceMetric(self.metric))
            except ValueError as err:
                raise RetrievalContractError(
                    f"Unsupported distance metric '{self.metric}'. Supported metrics: {[m.value for m in DistanceMetric]}"
                ) from err
        elif not isinstance(self.metric, DistanceMetric):
            raise RetrievalContractError(
                f"metric must be a DistanceMetric instance or valid string, got {type(self.metric).__name__}"
            )
