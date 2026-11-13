"""Tenant-isolated PostgreSQL + pgvector repository adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg import sql

from flowforge.ai.retrieval.exceptions import RetrievalContractError
from flowforge.ai.retrieval.pgvector.config import (
    MAX_BATCH_SIZE,
    MAX_METADATA_DEPTH,
    MAX_METADATA_SIZE_BYTES,
    MAX_TEXT_CONTENT_LENGTH,
    MAX_TOP_K,
    PgVectorConfig,
)
from flowforge.ai.retrieval.pgvector.exceptions import (
    PgVectorError,
    PgVectorExtensionUnavailableError,
)
from flowforge.ai.retrieval.pgvector.filters import compile_metadata_filters
from flowforge.ai.retrieval.pgvector.schema import (
    CHECK_EXTENSION_SQL,
    CREATE_EXTENSION_SQL,
    SCHEMA_DDL,
)
from flowforge.ai.retrieval.protocols import QueryBundle, SearchResult, VectorDocument, VectorIndex
from flowforge.ai.retrieval.validation import (
    validate_query_bundle,
    validate_search_results,
    validate_tenant_scope,
    validate_vector_documents,
)


def _check_metadata_depth(obj: Any, current_depth: int = 1) -> None:
    """Validate that nested metadata does not exceed MAX_METADATA_DEPTH."""
    if current_depth > MAX_METADATA_DEPTH:
        raise RetrievalContractError(
            f"Metadata nesting depth exceeds maximum allowed limit ({MAX_METADATA_DEPTH})"
        )
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _check_metadata_depth(v, current_depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _check_metadata_depth(item, current_depth + 1)


class PgVectorIndex(VectorIndex):
    """PostgreSQL + pgvector implementation of VectorIndex protocol.

    Features:
    - Pure tenant isolation enforced strictly at the database SQL query level (WHERE tenant_id = %s).
    - Note on authorization: tenant_id parameter scopes SQL data boundaries. Callers MUST ensure
      tenant_id is authenticated and resolved from authorized session/user context before passing here.
    - Parameterized queries throughout (no dynamic SQL interpolation).
    - Full floating-point precision vector serialization (no precision truncation).
    - Rejects zero-magnitude vectors for Cosine metric to prevent NaN/Inf cosine distance.
    - Strict bounds on top_k, batch size, text length, and metadata depth/size.
    - Least privilege: auto_bootstrap and create_extension default to False.
    - Graceful pgvector extension detection and bootstrap handling.
    - Idempotent upsert via ON CONFLICT (tenant_id, id) DO UPDATE.
    - Deterministic score calculation and tie-breaking by document ID.
    - Explicit delete method with strict tenant isolation.
    """

    def __init__(
        self,
        connection_factory: Callable[[], psycopg.Connection[Any]] | str,
        config: PgVectorConfig | None = None,
    ) -> None:
        """Initialize PgVectorIndex.

        Args:
            connection_factory: Either a psycopg connection string (DSN) or a callable returning a new connection.
            config: Optional PgVectorConfig specifying table name, dimension, and distance metric.
        """
        if isinstance(connection_factory, str):
            conn_str = connection_factory
            self._connection_factory = lambda: psycopg.connect(conn_str)
        elif callable(connection_factory):
            self._connection_factory = connection_factory
        else:
            raise RetrievalContractError(
                "connection_factory must be a connection string (DSN) or a callable returning a psycopg.Connection"
            )

        self._config = config or PgVectorConfig()
        self._bootstrapped = False

    @property
    def config(self) -> PgVectorConfig:
        return self._config

    def bootstrap(self, conn: psycopg.Connection[Any] | None = None) -> None:
        """Ensure pgvector extension is available and storage table exists.

        Raises:
            PgVectorExtensionUnavailableError: If pgvector extension is absent and cannot be created.
            PgVectorError: On database DDL failure.
        """
        should_close = False
        if conn is None:
            conn = self._connection_factory()
            should_close = True

        try:
            with conn.cursor() as cur:
                # 1. Check or create extension
                cur.execute(CHECK_EXTENSION_SQL)
                ext_exists = cur.fetchone() is not None

                if not ext_exists:
                    if self._config.create_extension:
                        try:
                            cur.execute(CREATE_EXTENSION_SQL)
                            conn.commit()
                        except Exception as e:
                            conn.rollback()
                            raise PgVectorExtensionUnavailableError(
                                f"Failed to create pgvector extension: {e}"
                            ) from e
                    else:
                        raise PgVectorExtensionUnavailableError(
                            "pgvector extension is not installed in the target database"
                        )

                # 2. Verify extension is indeed present
                cur.execute(CHECK_EXTENSION_SQL)
                if cur.fetchone() is None:
                    raise PgVectorExtensionUnavailableError(
                        "pgvector extension is not available in the database"
                    )

                # 3. Create schema table and indexes
                ddl = SCHEMA_DDL.format(
                    table_name=self._config.table_name,
                    dimension=self._config.dimension,
                )
                cur.execute(ddl)
                conn.commit()
                self._bootstrapped = True
        except (PgVectorExtensionUnavailableError, RetrievalContractError):
            raise
        except Exception as e:
            conn.rollback()
            raise PgVectorError(f"Bootstrap failed: {e}") from e
        finally:
            if should_close:
                conn.close()

    def _ensure_bootstrapped(self, conn: psycopg.Connection[Any]) -> None:
        if not self._bootstrapped and self._config.auto_bootstrap:
            self.bootstrap(conn=conn)

    @staticmethod
    def _format_vector(vector: tuple[float, ...]) -> str:
        """Format a sequence of floats as a pgvector literal string '[v1,v2,...]' with full precision."""
        return "[" + ",".join(repr(float(v)) for v in vector) + "]"

    def _validate_vector_magnitude(self, vector: tuple[float, ...]) -> None:
        """Verify vector magnitude is non-zero for cosine similarity metric."""
        if self._config.metric.value == "cosine":
            norm_sq = sum(float(x) ** 2 for x in vector)
            if norm_sq == 0.0 or math.isclose(norm_sq, 0.0, abs_tol=1e-12):
                raise RetrievalContractError(
                    "Zero-magnitude vector is invalid for cosine distance metric (causes division by zero)"
                )

    def upsert(
        self,
        tenant_id: str,
        documents: list[VectorDocument],
    ) -> int:
        """Upsert vector documents into the index under the given tenant scope.

        Enforces:
        - Strict tenant isolation: document.tenant_id must match parameter tenant_id.
        - Dimension check against config.dimension.
        - Batch size capped at MAX_BATCH_SIZE.
        - Finite float values and non-zero magnitude check for cosine metric.
        - Text length and metadata depth/size bounds.
        - Parameterized INSERT ... ON CONFLICT (tenant_id, id) DO UPDATE.

        Args:
            tenant_id: Tenant scope identifier.
            documents: List of VectorDocument items to index.

        Returns:
            Number of documents successfully upserted.
        """
        valid_tenant = validate_tenant_scope(tenant_id)

        if not documents:
            return 0

        if len(documents) > MAX_BATCH_SIZE:
            raise RetrievalContractError(
                f"Batch size exceeds maximum allowed limit ({len(documents)} > {MAX_BATCH_SIZE})"
            )

        # Validate documents using retrieval validation helper
        validate_vector_documents(
            documents,
            expected_tenant_id=valid_tenant,
            expected_dimension=self._config.dimension,
        )

        # Validate magnitudes, text content length, and metadata bounds
        upsert_params = []
        for doc in documents:
            self._validate_vector_magnitude(doc.vector)

            if len(doc.content) > MAX_TEXT_CONTENT_LENGTH:
                raise RetrievalContractError(
                    f"Document '{doc.id}' content length exceeds maximum limit ({len(doc.content)} > {MAX_TEXT_CONTENT_LENGTH})"
                )

            meta = dict(doc.metadata) if doc.metadata else {}
            _check_metadata_depth(meta)
            meta_json = json.dumps(meta)
            if len(meta_json.encode("utf-8")) > MAX_METADATA_SIZE_BYTES:
                raise RetrievalContractError(
                    f"Document '{doc.id}' metadata size exceeds maximum limit of {MAX_METADATA_SIZE_BYTES} bytes"
                )

            upsert_params.append(
                (
                    valid_tenant,
                    doc.id,
                    self._format_vector(doc.vector),
                    doc.content,
                    meta_json,
                )
            )

        with self._connection_factory() as conn:
            self._ensure_bootstrapped(conn)
            with conn.cursor() as cur:
                table = sql.Identifier(self._config.table_name)
                upsert_sql = sql.SQL(
                    """
                    INSERT INTO {table} (tenant_id, id, vector, content, metadata, updated_at)
                    VALUES (%s, %s, %s::vector, %s, %s::jsonb, CURRENT_TIMESTAMP)
                    ON CONFLICT (tenant_id, id) DO UPDATE SET
                        vector = EXCLUDED.vector,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ).format(table=table)

                cur.executemany(upsert_sql, upsert_params)
                conn.commit()
                return len(documents)

    def search(
        self,
        query: QueryBundle,
    ) -> list[SearchResult]:
        """Search the index for vector documents matching the query bundle.

        Enforces:
        - Strict tenant isolation: query is executed with WHERE tenant_id = %s.
        - top_k bounded by MAX_TOP_K.
        - Dimension validation of query_vector.
        - Non-zero vector magnitude check for cosine similarity.
        - Operator-based distance metric calculation with parameterized query vector.
        - Safe parameterized metadata filters.
        - Deterministic ordering: primary score descending, tie-breaker id ascending.
        - Score threshold filtering.

        Args:
            query: QueryBundle specifying tenant scope, query vector, top_k, threshold, filters.

        Returns:
            List of SearchResult items sorted descending by similarity score.
        """
        validate_query_bundle(query)
        tenant_id = validate_tenant_scope(query.tenant_id)

        if query.top_k > MAX_TOP_K:
            raise RetrievalContractError(
                f"top_k exceeds maximum allowed limit ({query.top_k} > {MAX_TOP_K})"
            )

        if query.query_vector is None:
            raise RetrievalContractError("PgVectorIndex requires query_vector in QueryBundle")

        if len(query.query_vector) != self._config.dimension:
            raise RetrievalContractError(
                f"Query vector dimension {len(query.query_vector)} does not match configured dimension {self._config.dimension}"
            )

        self._validate_vector_magnitude(query.query_vector)

        filter_sql, filter_params = compile_metadata_filters(query.filter_metadata)
        vector_str = self._format_vector(query.query_vector)
        dist_op = self._config.metric.operator

        with self._connection_factory() as conn:
            self._ensure_bootstrapped(conn)
            with conn.cursor() as cur:
                table = sql.Identifier(self._config.table_name)
                # Raw distance query
                search_sql_raw = f"""
                    SELECT id, content, metadata, (vector {dist_op} %s::vector) AS dist
                    FROM {{table}}
                    WHERE tenant_id = %s {filter_sql}
                    ORDER BY dist ASC, id ASC
                    LIMIT %s
                """

                fetch_limit = max(query.top_k, 1)
                search_sql = sql.SQL(search_sql_raw).format(table=table)
                all_params = [vector_str, tenant_id, *filter_params, fetch_limit]

                cur.execute(search_sql, all_params)
                rows = cur.fetchall()

                results: list[SearchResult] = []
                for row_id, content, metadata_raw, dist in rows:
                    dist_float = float(dist)
                    if math.isnan(dist_float) or math.isinf(dist_float):
                        # Safeguard against unexpected NaN/Inf in distance
                        continue

                    score = self._config.metric.score_from_distance(dist_float)
                    if query.score_threshold is not None and score < query.score_threshold:
                        continue

                    if isinstance(metadata_raw, str):
                        meta = json.loads(metadata_raw)
                    elif isinstance(metadata_raw, dict):
                        meta = dict(metadata_raw)
                    else:
                        meta = {}

                    results.append(
                        SearchResult(
                            id=str(row_id),
                            score=score,
                            content=str(content),
                            tenant_id=tenant_id,
                            metadata=meta,
                        )
                    )

                # Final sort by score desc, id asc
                results.sort(key=lambda r: (-r.score, r.id))
                results = results[: query.top_k]

                validate_search_results(results, expected_tenant_id=tenant_id)
                return results

    def delete(
        self,
        tenant_id: str,
        document_ids: list[str],
    ) -> int:
        """Delete specific documents strictly scoped to the tenant.

        Args:
            tenant_id: Tenant scope identifier.
            document_ids: List of document IDs to delete.

        Returns:
            Number of rows deleted.
        """
        valid_tenant = validate_tenant_scope(tenant_id)

        if not isinstance(document_ids, list):
            raise RetrievalContractError(
                f"document_ids must be a list of strings, got {type(document_ids).__name__}"
            )

        if not document_ids:
            return 0

        clean_ids: list[str] = []
        for item in document_ids:
            if not isinstance(item, str):
                raise RetrievalContractError(
                    f"Each document ID in document_ids must be a string, got {type(item).__name__}"
                )
            stripped = item.strip()
            if stripped:
                clean_ids.append(stripped)

        if not clean_ids:
            return 0

        with self._connection_factory() as conn:
            self._ensure_bootstrapped(conn)
            with conn.cursor() as cur:
                table = sql.Identifier(self._config.table_name)
                del_sql = sql.SQL(
                    """
                    DELETE FROM {table}
                    WHERE tenant_id = %s AND id = ANY(%s)
                    """
                ).format(table=table)
                cur.execute(del_sql, [valid_tenant, clean_ids])
                deleted = cur.rowcount
                conn.commit()
                return deleted

    def delete_tenant(self, tenant_id: str) -> int:
        """Delete all documents belonging to a tenant (e.g. for GDPR / tenant offboarding).

        Args:
            tenant_id: Tenant scope identifier.

        Returns:
            Number of rows deleted.
        """
        valid_tenant = validate_tenant_scope(tenant_id)
        with self._connection_factory() as conn:
            self._ensure_bootstrapped(conn)
            with conn.cursor() as cur:
                table = sql.Identifier(self._config.table_name)
                del_sql = sql.SQL(
                    """
                    DELETE FROM {table}
                    WHERE tenant_id = %s
                    """
                ).format(table=table)
                cur.execute(del_sql, [valid_tenant])
                deleted = cur.rowcount
                conn.commit()
                return deleted
