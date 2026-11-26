"""RED regression tests for FIX2-01: Eliminate Policy-Layer Authorization Bypass."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import AIGovernanceError, AIReservationNotFoundError
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    ReservationStatus,
    SettlementType,
)
from flowforge.ai.governance.policy import AIBudgetPolicyService


@pytest.fixture(params=["in_memory", "postgres"])
def policy_and_repo(request):
    if request.param == "in_memory":
        repo = InMemoryBudgetRepository()
    else:
        repo = PostgreSQLBudgetRepository()
    service = AIBudgetPolicyService(repository=repo)
    return service, repo


@pytest.mark.django_db(transaction=True)
def test_fix2_01_unauthenticated_release_with_scope_none_rejected(policy_and_repo):
    """An unauthenticated caller passing scope=None to policy.release() must NOT bypass tenant isolation."""
    service, repo = policy_and_repo
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")

    res, _ = repo.reserve_atomic(
        reservation_id="res-tenant-a-bypass-1",
        idempotency_key="idemp-tenant-a-bypass-1",
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

    # Caller omits scope (scope=None). Currently policy service fetches reservation with internal_privileged=True!
    # Under FIX2-01, this must fail closed and NOT release the reservation!
    with pytest.raises((AIReservationNotFoundError, AIGovernanceError)):
        service.release(
            reservation_id="res-tenant-a-bypass-1",
            scope=None,
        )

    # Verify reservation is still pending
    rec = repo.get_reservation_by_id("res-tenant-a-bypass-1", scope=scope_a)
    assert rec.status == ReservationStatus.PENDING


@pytest.mark.django_db(transaction=True)
def test_fix2_01_unauthenticated_settle_with_scope_none_rejected(policy_and_repo):
    """An unauthenticated caller passing scope=None to policy.settle() must NOT bypass tenant isolation."""
    service, repo = policy_and_repo
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")

    res, _ = repo.reserve_atomic(
        reservation_id="res-tenant-a-settle-bypass-1",
        idempotency_key="idemp-tenant-a-settle-bypass-1",
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

    # Caller omits scope (scope=None). Must fail closed.
    with pytest.raises((AIReservationNotFoundError, AIGovernanceError)):
        service.settle(
            reservation_id="res-tenant-a-settle-bypass-1",
            scope=None,
            actual_prompt_tokens=50,
            actual_completion_tokens=50,
            actual_cost_micros=2000,
            settlement_type=SettlementType.EXACT,
        )

    # Verify reservation is still pending
    rec = repo.get_reservation_by_id("res-tenant-a-settle-bypass-1", scope=scope_a)
    assert rec.status == ReservationStatus.PENDING


@pytest.mark.django_db(transaction=True)
def test_fix2_01_tenant_b_cannot_release_tenant_a_via_policy(policy_and_repo):
    """Tenant B passing scope_b cannot release Tenant A's reservation via policy."""
    service, repo = policy_and_repo
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-B")

    repo.reserve_atomic(
        reservation_id="res-tenant-a-cross-1",
        idempotency_key="idemp-tenant-a-cross-1",
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

    with pytest.raises((AIReservationNotFoundError, AIGovernanceError)):
        service.release(
            reservation_id="res-tenant-a-cross-1",
            scope=scope_b,
        )

    rec = repo.get_reservation_by_id("res-tenant-a-cross-1", scope=scope_a)
    assert rec.status == ReservationStatus.PENDING


@pytest.mark.django_db(transaction=True)
def test_fix2_01_tenant_b_cannot_settle_tenant_a_via_policy(policy_and_repo):
    """Tenant B passing scope_b cannot settle Tenant A's reservation via policy."""
    service, repo = policy_and_repo
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-B")

    repo.reserve_atomic(
        reservation_id="res-tenant-a-cross-settle-1",
        idempotency_key="idemp-tenant-a-cross-settle-1",
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

    with pytest.raises((AIReservationNotFoundError, AIGovernanceError)):
        service.settle(
            reservation_id="res-tenant-a-cross-settle-1",
            scope=scope_b,
            actual_prompt_tokens=50,
            actual_completion_tokens=50,
            actual_cost_micros=2000,
            settlement_type=SettlementType.EXACT,
        )

    rec = repo.get_reservation_by_id("res-tenant-a-cross-settle-1", scope=scope_a)
    assert rec.status == ReservationStatus.PENDING
