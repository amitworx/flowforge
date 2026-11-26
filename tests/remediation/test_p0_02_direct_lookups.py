"""RED regression tests for P0-02: Direct Identifier Lookups Authorization."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    SettlementType,
)


@pytest.fixture(params=["in_memory", "postgres"])
def repo(request):
    if request.param == "in_memory":
        return InMemoryBudgetRepository()
    return PostgreSQLBudgetRepository()


@pytest.mark.django_db(transaction=True)
def test_p0_02_cross_tenant_reservation_lookup(repo):
    """Direct reservation lookup by ID must return None when queried with mismatched scope or omitted scope for tenant record."""
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-B")

    res, _ = repo.reserve_atomic(
        reservation_id="res-lookup-a",
        idempotency_key="idemp-lookup-a",
        scope=scope_a,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=200,
        estimated_total_tokens=300,
        estimated_cost_micros=5000,
        attempt_number=1,
        created_at=now,
    )

    # Scoped with Tenant A must succeed
    assert repo.get_reservation_by_id("res-lookup-a", scope=scope_a) is not None

    # Scoped with Tenant B must return None
    assert repo.get_reservation_by_id("res-lookup-a", scope=scope_b) is None

    # Scoped with None (omitted or unprivileged) must return None for tenant-owned records
    assert repo.get_reservation_by_id("res-lookup-a", scope=None) is None


@pytest.mark.django_db(transaction=True)
def test_p0_02_cross_tenant_settlement_lookups(repo):
    """Direct settlement lookups by ID and by reservation ID must return None for cross-tenant access."""
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-B")

    repo.reserve_atomic(
        reservation_id="res-stl-lookup-a",
        idempotency_key="idemp-stl-lookup-a",
        scope=scope_a,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=200,
        estimated_total_tokens=300,
        estimated_cost_micros=5000,
        attempt_number=1,
        created_at=now,
    )

    stl, _ = repo.settle_reservation(
        settlement_id="stl-lookup-a",
        reservation_id="res-stl-lookup-a",
        idempotency_key="idemp-stl-act-a",
        settlement_type=SettlementType.EXACT,
        actual_prompt_tokens=100,
        actual_completion_tokens=200,
        actual_total_tokens=300,
        actual_cost_micros=5000,
        attempt_number=1,
        settled_at=now,
    )

    # Scoped lookup with matching tenant
    assert repo.get_settlement_by_id("stl-lookup-a", scope=scope_a) is not None
    assert repo.get_settlement_by_reservation_id("res-stl-lookup-a", scope=scope_a) is not None

    # Cross-tenant lookups
    assert repo.get_settlement_by_id("stl-lookup-a", scope=scope_b) is None
    assert repo.get_settlement_by_reservation_id("res-stl-lookup-a", scope=scope_b) is None

    # Unscoped / scope=None lookups
    assert repo.get_settlement_by_id("stl-lookup-a", scope=None) is None
    assert repo.get_settlement_by_reservation_id("res-stl-lookup-a", scope=None) is None
