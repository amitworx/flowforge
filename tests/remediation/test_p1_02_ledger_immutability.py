"""RED regression tests for P1-02: Enforce Append-Only Ledger Database Constraint."""

import datetime

import pytest
from django.db import transaction

from flowforge.ai.governance.db.models import AIUsageLedgerModel


@pytest.mark.django_db(transaction=True)
def test_p1_02_database_level_ledger_immutability():
    """Verify that PostgreSQL rejects direct UPDATE and DELETE statements on ai_governance_ledger."""
    AIUsageLedgerModel.objects.create(
        entry_id="ledger-immutable-1",
        event_type="reservation",
        idempotency_key="idemp-immutable-1",
        scope_key="type=tenant:tenant=t1",
        scope_type="tenant",
        tenant_id="t1",
        provider="ollama",
        model="llama3",
        timestamp=datetime.datetime.now(datetime.UTC),
        tokens_delta=100,
        cost_micros_delta=1000,
        calls_delta=1,
    )

    # 1. Attempt UPDATE via raw SQL / ORM
    with pytest.raises(Exception) as exc_update, transaction.atomic():
        AIUsageLedgerModel.objects.filter(entry_id="ledger-immutable-1").update(tokens_delta=999)

    assert (
        "ai_governance_ledger is append-only" in str(exc_update.value).lower()
        or "protected" in str(exc_update.value).lower()
    )

    # 2. Attempt DELETE via raw SQL / ORM
    with pytest.raises(Exception) as exc_delete, transaction.atomic():
        AIUsageLedgerModel.objects.filter(entry_id="ledger-immutable-1").delete()

    assert (
        "ai_governance_ledger is append-only" in str(exc_delete.value).lower()
        or "protected" in str(exc_delete.value).lower()
    )
