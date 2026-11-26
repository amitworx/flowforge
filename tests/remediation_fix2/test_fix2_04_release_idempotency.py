"""RED regression tests for FIX2-04: Enforce Release Idempotency-Key Semantics."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import (
    AIDuplicateSettlementError,
    AIGovernanceError,
    AIInvalidReservationStateError,
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
def test_fix2_04_release_idempotent_replay_and_conflict(repo):
    """Initial release succeeds, same-key replay returns replayed=True, conflicting key raises AIDuplicateSettlementError."""
    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-rel-idem-1")

    repo.reserve_atomic(
        reservation_id="res-rel-idem-1",
        idempotency_key="idemp-create-1",
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

    # 1. First release with key-A succeeds with replayed=False
    rec1, replayed1 = repo.release_reservation(
        reservation_id="res-rel-idem-1",
        idempotency_key="release-key-A",
        scope=scope,
        released_at=now,
        reason="cancelled",
    )
    assert rec1.status == ReservationStatus.RELEASED
    assert replayed1 is False

    # 2. Same key release-key-A replay succeeds with replayed=True
    rec2, replayed2 = repo.release_reservation(
        reservation_id="res-rel-idem-1",
        idempotency_key="release-key-A",
        scope=scope,
        released_at=now,
        reason="cancelled",
    )
    assert rec2.status == ReservationStatus.RELEASED
    assert replayed2 is True

    # 3. Conflicting release with different release-key-B must FAIL and NOT return replayed=True
    with pytest.raises(
        (AIDuplicateSettlementError, AIInvalidReservationStateError, AIGovernanceError)
    ):
        repo.release_reservation(
            reservation_id="res-rel-idem-1",
            idempotency_key="release-key-B",
            scope=scope,
            released_at=now,
            reason="cancelled",
        )
