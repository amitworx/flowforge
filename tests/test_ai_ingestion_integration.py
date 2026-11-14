"""Real pgvector live integration test for IngestionService (M0-017B)."""

from __future__ import annotations

import uuid

import psycopg

from flowforge.ai.retrieval.ingestion import (
    IngestionConfig,
    IngestionDocument,
    IngestionService,
)
from flowforge.ai.retrieval.pgvector import (
    DistanceMetric,
    PgVectorConfig,
    PgVectorIndex,
)
from tests.test_ai_ingestion_unit import MockEmbeddingProvider
from tests.test_ai_pgvector_integration import PGVECTOR_DSN, requires_pgvector


@requires_pgvector
def test_ingestion_service_with_real_pgvector():
    """Verify end-to-end ingestion and query against a live PostgreSQL + pgvector database."""
    table_name = f"test_ingest_{uuid.uuid4().hex[:12]}"
    dim = 4

    pg_cfg = PgVectorConfig(
        table_name=table_name,
        dimension=dim,
        metric=DistanceMetric.COSINE,
        create_extension=True,
    )
    vector_index = PgVectorIndex(connection_factory=PGVECTOR_DSN, config=pg_cfg)
    vector_index.bootstrap()

    try:
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        ingest_cfg = IngestionConfig(
            batch_size=10,
            expected_dimension=dim,
        )
        service = IngestionService(
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            config=ingest_cfg,
        )

        tenant = "tenant-prod-42"
        docs = [
            IngestionDocument(
                document_id="policy-manual",
                text="Standard operating procedures for team members. Safety first and security compliance.",
                metadata={"department": "compliance", "version": 1},
            ),
            IngestionDocument(
                document_id="runbook-deployment",
                text="Deployment instructions for production servers. Ensure backups are verified.",
                metadata={"department": "devops", "version": 2},
            ),
        ]

        # 1. Ingest
        result = service.ingest_documents(tenant_id=tenant, documents=docs)
        assert result.documents_processed == 2
        assert result.chunks_created >= 2
        assert result.upserted_count == result.chunks_created

        # 2. Query
        search_results = service.query(
            tenant_id=tenant,
            query_text="operating procedures",
            top_k=5,
        )
        assert len(search_results) > 0
        for res in search_results:
            assert res.tenant_id == tenant
            assert res.score >= -1.0 and res.score <= 1.0
            assert "_provenance" in res.metadata

        # 3. Cross-tenant isolation verification
        other_tenant_results = service.query(
            tenant_id="tenant-foreign-99",
            query_text="operating procedures",
            top_k=5,
        )
        assert len(other_tenant_results) == 0

    finally:
        # Cleanup table
        try:
            with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn, conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        except Exception:
            pass
