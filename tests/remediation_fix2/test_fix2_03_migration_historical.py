"""RED regression tests for FIX2-03: Safe Handling of Historical Pending Reservations During Migration."""

import datetime

import pytest

from flowforge.ai.governance.db.models import AIReservationModel
from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import AIInvalidReservationStateError
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    DispatchStatus,
    ReservationStatus,
)


@pytest.mark.django_db(transaction=True)
def test_fix2_03_historical_pending_reservation_cannot_be_released_as_not_dispatched():
    """Historical pending reservations with uncertain dispatch history must NOT be released as unused.

    If an existing database had pending reservations created before safe dispatch tracking was implemented,
    or during an upgrade from 0001, their dispatch history is uncertain.
    They must be classified as BILLING_UNCERTAIN (or fenced from release), preventing premature refund.
    """
    now = datetime.datetime.now(datetime.UTC)
    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-hist-1")

    # Simulate a historical reservation in the database that has dispatch_status='billing_uncertain'
    # or an unverified historical record.
    AIReservationModel.objects.create(
        reservation_id="res-hist-pending-1",
        idempotency_key="idemp-hist-1",
        scope_key=scope.key,
        scope_type=scope.scope_type.value,
        tenant_id=scope.tenant_id,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=1000,
        attempt_number=1,
        status=ReservationStatus.PENDING.value,
        dispatch_status=DispatchStatus.BILLING_UNCERTAIN.value,
        created_at=now,
    )

    # Attempting to release this historical reservation MUST raise AIInvalidReservationStateError
    with pytest.raises(AIInvalidReservationStateError):
        repo.release_reservation(
            reservation_id="res-hist-pending-1",
            idempotency_key="idemp-rel-hist-1",
            scope=scope,
            released_at=now,
        )

    # Verify reservation is still pending in DB
    rec = repo.get_reservation_by_id("res-hist-pending-1", scope=scope)
    assert rec.status == ReservationStatus.PENDING
    assert rec.dispatch_status == DispatchStatus.BILLING_UNCERTAIN
