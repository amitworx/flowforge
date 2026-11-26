"""RED regression tests for P0-01: Cross-Tenant Reservation Release Vulnerability."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import (
    AIGovernanceError,
    AIReservationNotFoundError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    ReservationStatus,
)


@pytest.fixture(params=["in_memory", "postgres"])
def repo(request):
    if request.param == "in_memory":
        return InMemoryBudgetRepository()
    return PostgreSQLBudgetRepository()


@pytest.mark.django_db(transaction=True)
def test_p0_01_cross_tenant_release_rejected(repo):
    """Tenant B must not be able to release Tenant A's reservation."""
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-B")

    res, _ = repo.reserve_atomic(
        reservation_id="res-tenant-a-1",
        idempotency_key="idemp-tenant-a-1",
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

    # Tenant B attempts to release Tenant A's reservation using scope_b
    with pytest.raises((AIReservationNotFoundError, AIGovernanceError)):
        repo.release_reservation(
            reservation_id="res-tenant-a-1",
            idempotency_key="idemp-rel-b-1",
            scope=scope_b,
            released_at=now,
        )

    # Verify reservation is STILL PENDING and counters are intact
    rec = repo.get_reservation_by_id("res-tenant-a-1", scope=scope_a)
    assert rec.status == ReservationStatus.PENDING
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope_a)
    assert (active_calls, active_tokens, active_cost) == (1, 300, 5000)


@pytest.mark.django_db(transaction=True)
def test_p0_01_mismatched_hierarchy_release_rejected(repo):
    """Same tenant but wrong workflow or run must be rejected."""
    now = datetime.datetime.now(datetime.UTC)
    scope_wf1 = BudgetScope(
        scope_type=BudgetScopeType.WORKFLOW,
        tenant_id="tenant-A",
        workflow_id="wf-1",
    )
    scope_wf2 = BudgetScope(
        scope_type=BudgetScopeType.WORKFLOW,
        tenant_id="tenant-A",
        workflow_id="wf-2",
    )

    repo.reserve_atomic(
        reservation_id="res-wf1-1",
        idempotency_key="idemp-wf1-1",
        scope=scope_wf1,
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
        repo.release_reservation(
            reservation_id="res-wf1-1",
            idempotency_key="idemp-rel-wf2",
            scope=scope_wf2,
            released_at=now,
        )


@pytest.mark.django_db(transaction=True)
def test_p0_01_missing_scope_release_rejected(repo):
    """Release without scope must be rejected or raise TypeError / AIGovernanceError."""
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")

    repo.reserve_atomic(
        reservation_id="res-tenant-a-missing-scope",
        idempotency_key="idemp-tenant-a-missing-scope",
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

    # Calling with scope=None must fail closed
    with pytest.raises(AIGovernanceError):
        repo.release_reservation(
            reservation_id="res-tenant-a-missing-scope",
            idempotency_key="idemp-rel-missing",
            scope=None,
            released_at=now,
        )


@pytest.mark.django_db(transaction=True)
def test_p0_01_release_idempotency_replay_and_conflict(repo):
    """Identical release is idempotent; conflicting release or replay with different params is handled safely."""
    now = datetime.datetime.now(datetime.UTC)
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")

    repo.reserve_atomic(
        reservation_id="res-tenant-a-idem",
        idempotency_key="idemp-tenant-a-idem",
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

    # First release
    rel1, replayed1 = repo.release_reservation(
        reservation_id="res-tenant-a-idem",
        idempotency_key="idemp-rel-1",
        scope=scope_a,
        released_at=now,
    )
    assert not replayed1
    assert rel1.status == ReservationStatus.RELEASED

    # Replay with same idempotency key
    rel2, replayed2 = repo.release_reservation(
        reservation_id="res-tenant-a-idem",
        idempotency_key="idemp-rel-1",
        scope=scope_a,
        released_at=now,
    )
    assert replayed2
    assert rel2.status == ReservationStatus.RELEASED

    # Counters must not be decremented twice!
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope_a)
    assert (active_calls, active_tokens, active_cost) == (0, 0, 0)
