"""Schema constants and DDL definitions for tenant-isolated pgvector storage."""

from __future__ import annotations

TABLE_NAME_DEFAULT = "flowforge_ai_vector_documents"

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    tenant_id VARCHAR(255) NOT NULL,
    id VARCHAR(255) NOT NULL,
    vector vector({dimension}) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_tenant_id
ON {table_name} (tenant_id);

CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata
ON {table_name} USING gin (metadata);
"""

CHECK_EXTENSION_SQL = """
SELECT 1 FROM pg_extension WHERE extname = 'vector';
"""

CREATE_EXTENSION_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
"""
