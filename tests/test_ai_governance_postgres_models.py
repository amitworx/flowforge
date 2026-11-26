"""RED test suite for FlowForge AI Governance PostgreSQL Models and Migrations."""

import pytest
from django.db import connection


@pytest.mark.django_db
def test_governance_models_and_tables_exist():
    """Verify that ai_governance models exist and their tables are present in PostgreSQL."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name LIKE 'ai_governance_%'
            """
        )
        tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {
        "ai_governance_scope_counter",
        "ai_governance_budget_settings",
        "ai_governance_reservation",
        "ai_governance_settlement",
        "ai_governance_ledger",
    }
    assert expected_tables.issubset(tables)
