"""RED regression tests for P0-03: Provider Dispatch Lifecycle State Machine."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import (
    AIGovernanceError,
    AIInvalidReservationStateError,
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
def test_p0_03_dispatch_lifecycle_transitions(repo):
    """Verify dispatch lifecycle states: NOT_DISPATCHED -> DISPATCH_STARTED -> RESPONSE_RECEIVED."""
    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-dispatch")

    res, _ = repo.reserve_atomic(
        reservation_id="res-disp-1",
        idempotency_key="idemp-disp-1",
        scope=scope,
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
    # Default dispatch status must be NOT_DISPATCHED
    assert res.dispatch_status == DispatchStatus.NOT_DISPATCHED

    # Transition to DISPATCH_STARTED
    res_started = repo.transition_dispatch_status(
        reservation_id="res-disp-1",
        scope=scope,
        from_status=DispatchStatus.NOT_DISPATCHED,
        to_status=DispatchStatus.DISPATCH_STARTED,
        attempt_number=1,
    )
    assert res_started.dispatch_status == DispatchStatus.DISPATCH_STARTED

    # Attempting to release a DISPATCH_STARTED reservation must FAIL closed!
    with pytest.raises(AIInvalidReservationStateError):
        repo.release_reservation(
            reservation_id="res-disp-1",
            idempotency_key="idemp-rel-disp-1",
            scope=scope,
            released_at=now,
        )

    # Transition DISPATCH_STARTED -> RESPONSE_RECEIVED
    res_received = repo.transition_dispatch_status(
        reservation_id="res-disp-1",
        scope=scope,
        from_status=DispatchStatus.DISPATCH_STARTED,
        to_status=DispatchStatus.RESPONSE_RECEIVED,
        attempt_number=1,
    )
    assert res_received.dispatch_status == DispatchStatus.RESPONSE_RECEIVED


@pytest.mark.django_db(transaction=True)
def test_p0_03_invalid_dispatch_transition_and_fencing(repo):
    """Invalid transitions or wrong attempt fencing token must fail closed."""
    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-dispatch-fence")

    repo.reserve_atomic(
        reservation_id="res-disp-fence",
        idempotency_key="idemp-disp-fence",
        scope=scope,
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

    # Attempt wrong attempt_number
    with pytest.raises(AIGovernanceError):
        repo.transition_dispatch_status(
            reservation_id="res-disp-fence",
            scope=scope,
            from_status=DispatchStatus.NOT_DISPATCHED,
            to_status=DispatchStatus.DISPATCH_STARTED,
            attempt_number=99,
        )

    # Attempt invalid jump: NOT_DISPATCHED -> RESPONSE_RECEIVED
    with pytest.raises(AIInvalidReservationStateError):
        repo.transition_dispatch_status(
            reservation_id="res-disp-fence",
            scope=scope,
            from_status=DispatchStatus.NOT_DISPATCHED,
            to_status=DispatchStatus.RESPONSE_RECEIVED,
            attempt_number=1,
        )
