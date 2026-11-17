"""RED regression tests for FIX2-02: Compare-and-Swap Dispatch Transitions."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import (
    AIInvalidReservationStateError,
    AIStaleFencedAttemptError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    DispatchStatus,
)


@pytest.fixture(params=["in_memory", "postgres"])
def repo(request):
    if request.param == "in_memory":
        return InMemoryBudgetRepository()
    return PostgreSQLBudgetRepository()


@pytest.mark.django_db(transaction=True)
def test_fix2_02_transition_stale_from_status_rejected(repo):
    """When a worker attempts a transition with a stale from_status, it MUST raise AIInvalidReservationStateError."""
    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-cas-1")

    res, _ = repo.reserve_atomic(
        reservation_id="res-cas-1",
        idempotency_key="idemp-cas-1",
        scope=scope,
        ancestor_scopes=[],
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=1000,
        attempt_number=1,
        created_at=now,
    )
    assert res.dispatch_status == DispatchStatus.NOT_DISPATCHED

    # Worker A transitions from NOT_DISPATCHED to DISPATCH_STARTED
    res_a = repo.transition_dispatch_status(
        reservation_id="res-cas-1",
        scope=scope,
        from_status=DispatchStatus.NOT_DISPATCHED,
        to_status=DispatchStatus.DISPATCH_STARTED,
        attempt_number=1,
    )
    assert res_a.dispatch_status == DispatchStatus.DISPATCH_STARTED

    # Worker B (having stale expectation from_status=NOT_DISPATCHED) attempts to transition to RESPONSE_RECEIVED or DISPATCH_STARTED
    with pytest.raises(AIInvalidReservationStateError):
        repo.transition_dispatch_status(
            reservation_id="res-cas-1",
            scope=scope,
            from_status=DispatchStatus.NOT_DISPATCHED,
            to_status=DispatchStatus.RESPONSE_RECEIVED,
            attempt_number=1,
        )

    # State must remain DISPATCH_STARTED
    rec = repo.get_reservation_by_id("res-cas-1", scope=scope)
    assert rec.dispatch_status == DispatchStatus.DISPATCH_STARTED


@pytest.mark.django_db(transaction=True)
def test_fix2_02_transition_matching_from_status_succeeds(repo):
    """When from_status matches persisted status, valid legal transition succeeds."""
    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-cas-2")

    repo.reserve_atomic(
        reservation_id="res-cas-2",
        idempotency_key="idemp-cas-2",
        scope=scope,
        ancestor_scopes=[],
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=1000,
        attempt_number=1,
        created_at=now,
    )

    # 1. NOT_DISPATCHED -> DISPATCH_STARTED
    rec1 = repo.transition_dispatch_status(
        reservation_id="res-cas-2",
        scope=scope,
        from_status=DispatchStatus.NOT_DISPATCHED,
        to_status=DispatchStatus.DISPATCH_STARTED,
        attempt_number=1,
    )
    assert rec1.dispatch_status == DispatchStatus.DISPATCH_STARTED

    # 2. DISPATCH_STARTED -> RESPONSE_RECEIVED
    rec2 = repo.transition_dispatch_status(
        reservation_id="res-cas-2",
        scope=scope,
        from_status=DispatchStatus.DISPATCH_STARTED,
        to_status=DispatchStatus.RESPONSE_RECEIVED,
        attempt_number=1,
    )
    assert rec2.dispatch_status == DispatchStatus.RESPONSE_RECEIVED


@pytest.mark.django_db(transaction=True)
def test_fix2_02_transition_stale_fencing_attempt_rejected(repo):
    """Mismatched attempt number must be rejected with AIStaleFencedAttemptError without altering state."""
    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-cas-3")

    repo.reserve_atomic(
        reservation_id="res-cas-3",
        idempotency_key="idemp-cas-3",
        scope=scope,
        ancestor_scopes=[],
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=1000,
        attempt_number=2,
        created_at=now,
    )

    with pytest.raises(AIStaleFencedAttemptError):
        repo.transition_dispatch_status(
            reservation_id="res-cas-3",
            scope=scope,
            from_status=DispatchStatus.NOT_DISPATCHED,
            to_status=DispatchStatus.DISPATCH_STARTED,
            attempt_number=1,  # Stale attempt
        )


@pytest.mark.django_db(transaction=True)
def test_fix2_02_transition_after_release_rejected(repo):
    """Transitioning dispatch status of an already-released reservation must be rejected."""
    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-cas-4")

    repo.reserve_atomic(
        reservation_id="res-cas-4",
        idempotency_key="idemp-cas-4",
        scope=scope,
        ancestor_scopes=[],
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=1000,
        attempt_number=1,
        created_at=now,
    )

    # Release reservation
    repo.release_reservation(
        reservation_id="res-cas-4",
        idempotency_key="idemp-cas-rel-4",
        scope=scope,
        released_at=now,
    )

    with pytest.raises(AIInvalidReservationStateError):
        repo.transition_dispatch_status(
            reservation_id="res-cas-4",
            scope=scope,
            from_status=DispatchStatus.NOT_DISPATCHED,
            to_status=DispatchStatus.DISPATCH_STARTED,
            attempt_number=1,
        )
