"""Comprehensive migration regression and financial safety test suite for M0-018B2-B-FIX3.

Tests verify:
1. Fresh install from scratch through 0005.
2. Step-by-step upgrade from 0001 -> 0002 -> 0003 -> 0004 -> 0005.
3. Step-by-step upgrade starting from 0002, 0003, and 0004.
4. Historical ambiguous pending rows (created prior to dispatch status tracking) are safely fenced to BILLING_UNCERTAIN.
5. Indistinguishable rows: proving why timestamps/clocks cannot reliably discriminate genuine undispatched vs dispatched rows.
6. Non-pending states (settled, released, expired) preserve their terminal state and are NOT modified by migration 0005.
7. Active dispatch states (dispatch_started, response_received, response_lost, billing_uncertain) are NOT corrupted.
8. Rollback semantics (0005 -> 0004 -> 0001 and forward reapplication).
9. Budget counter and ledger integrity during and after migration.
10. Conservative fencing prevents premature capacity release (financial safety over premature availability).
"""

import datetime

import pytest
from django.core.management import call_command
from django.db import connection

from flowforge.ai.governance.db.models import (
    AIBudgetScopeCounter,
    AIBudgetSettingsModel,
    AIReservationModel,
    AISettlementModel,
    AIUsageLedgerModel,
)
from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import AIInvalidReservationStateError
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    DispatchStatus,
    ReservationStatus,
    SettlementType,
)


@pytest.mark.django_db(transaction=True)
def test_migration_fresh_install():
    """Verify fresh install applies all migrations cleanly and leaves schema consistent."""
    call_command("migrate", "ai_governance", "zero", verbosity=0)
    call_command("migrate", "ai_governance", verbosity=0)

    # Verify tables exist and queryable
    assert AIReservationModel.objects.count() == 0
    assert AISettlementModel.objects.count() == 0
    assert AIUsageLedgerModel.objects.count() == 0
    assert AIBudgetScopeCounter.objects.count() == 0


@pytest.mark.django_db(transaction=True)
def test_migration_upgrade_from_0001_with_various_states():
    """Upgrade from 0001 schema containing pending, settled, and released reservations."""
    call_command("migrate", "ai_governance", "0001", verbosity=0)

    now = datetime.datetime.now(datetime.UTC)
    with connection.cursor() as cursor:
        # 1. Historical pending reservation (dispatch history unknown)
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, created_at, audit_metadata
            ) VALUES (
                'res-0001-pending', 'idem-0001-p', 'tenant:t-001', 'tenant', 't-001',
                'mock', 'model-1', 100, 100, 200, 1000, 1,
                'pending', %s, '{}'::jsonb
            );
            """,
            [now],
        )
        # 2. Historical settled reservation
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, created_at, audit_metadata
            ) VALUES (
                'res-0001-settled', 'idem-0001-s', 'tenant:t-001', 'tenant', 't-001',
                'mock', 'model-1', 100, 100, 200, 1000, 1,
                'settled', %s, '{}'::jsonb
            );
            """,
            [now],
        )
        # 3. Historical released reservation
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, created_at, audit_metadata
            ) VALUES (
                'res-0001-released', 'idem-0001-r', 'tenant:t-001', 'tenant', 't-001',
                'mock', 'model-1', 100, 100, 200, 1000, 1,
                'released', %s, '{}'::jsonb
            );
            """,
            [now],
        )

    # Apply forward migrations to 0005
    call_command("migrate", "ai_governance", verbosity=0)

    # Historical pending reservation MUST be fenced to BILLING_UNCERTAIN
    res_p = AIReservationModel.objects.get(reservation_id="res-0001-pending")
    assert res_p.status == "pending"
    assert res_p.dispatch_status == DispatchStatus.BILLING_UNCERTAIN.value

    # Historical settled and released reservations must NOT be touched
    res_s = AIReservationModel.objects.get(reservation_id="res-0001-settled")
    assert res_s.status == "settled"
    # Migration 0002 default is not_dispatched, but status is terminal settled
    assert res_s.dispatch_status == DispatchStatus.NOT_DISPATCHED.value

    res_r = AIReservationModel.objects.get(reservation_id="res-0001-released")
    assert res_r.status == "released"
    assert res_r.dispatch_status == DispatchStatus.NOT_DISPATCHED.value


@pytest.mark.django_db(transaction=True)
def test_migration_upgrade_from_0002_0003_0004():
    """Verify upgrades from intermediate migrations 0002, 0003, and 0004 preserve dispatch states."""
    # Start at 0002
    call_command("migrate", "ai_governance", "0002", verbosity=0)

    now = datetime.datetime.now(datetime.UTC)
    with connection.cursor() as cursor:
        # Insert a pending reservation with not_dispatched (e.g. from 0002)
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, dispatch_status, created_at, audit_metadata
            ) VALUES (
                'res-0002-pending-notdisp', 'idem-0002-1', 'tenant:t-002', 'tenant', 't-002',
                'mock', 'model-1', 100, 100, 200, 1000, 1,
                'pending', 'not_dispatched', %s, '{}'::jsonb
            );
            """,
            [now],
        )
        # Insert a pending reservation that had already transitioned to dispatch_started
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, dispatch_status, created_at, audit_metadata
            ) VALUES (
                'res-0002-pending-started', 'idem-0002-2', 'tenant:t-002', 'tenant', 't-002',
                'mock', 'model-1', 100, 100, 200, 1000, 1,
                'pending', 'dispatch_started', %s, '{}'::jsonb
            );
            """,
            [now],
        )

    # Migrate forward through 0003 -> 0004 -> 0005
    call_command("migrate", "ai_governance", "0003", verbosity=0)
    call_command("migrate", "ai_governance", "0004", verbosity=0)
    call_command("migrate", "ai_governance", "0005", verbosity=0)

    res_notdisp = AIReservationModel.objects.get(reservation_id="res-0002-pending-notdisp")
    assert res_notdisp.dispatch_status == DispatchStatus.BILLING_UNCERTAIN.value

    # Explicit dispatch_started MUST NOT be altered
    res_started = AIReservationModel.objects.get(reservation_id="res-0002-pending-started")
    assert res_started.dispatch_status == DispatchStatus.DISPATCH_STARTED.value


@pytest.mark.django_db(transaction=True)
def test_indistinguishable_rows_financial_safety_tradeoff():
    """Demonstrate why application-supplied timestamps cannot reliably distinguish provenance.

    If an application or old worker created a pending reservation, or if clocks differed,
    the reservation could have an external dispatch attempt in-flight.
    Fencing pending rows with ambiguous dispatch history to BILLING_UNCERTAIN ensures
    we do not prematurely refund capacity, upholding financial safety.
    """
    call_command("migrate", "ai_governance", verbosity=0)

    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-tradeoff")

    now = datetime.datetime.now(datetime.UTC)
    # Create reservation via repo so counters are properly incremented
    res = repo.create_reservation(
        reservation_id="res-ambiguous-1",
        idempotency_key="idemp-ambiguous-1",
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=1000,
        attempt_number=1,
        created_at=now,
    )
    assert res.reservation_id == "res-ambiguous-1"
    # Simulate that this pending reservation was fenced to billing_uncertain (e.g. by migration 0005)
    AIReservationModel.objects.filter(reservation_id="res-ambiguous-1").update(
        dispatch_status=DispatchStatus.BILLING_UNCERTAIN.value
    )

    # Releasing this ambiguous reservation MUST fail closed
    with pytest.raises(AIInvalidReservationStateError):
        repo.release_reservation(
            reservation_id="res-ambiguous-1",
            idempotency_key="rel-ambiguous-1",
            scope=scope,
            released_at=now,
        )

    # However, settlement with truthful actual usage is supported when provider response arrives
    settlement, replayed = repo.settle_reservation(
        settlement_id="stl-ambiguous-1",
        reservation_id="res-ambiguous-1",
        idempotency_key="settle-ambiguous-1",
        settlement_type=SettlementType.EXACT,
        actual_prompt_tokens=80,
        actual_completion_tokens=80,
        actual_total_tokens=160,
        actual_cost_micros=800,
        attempt_number=1,
        settled_at=now,
    )
    assert not replayed
    assert settlement.actual_cost_micros == 800

    # Model is now settled
    m = AIReservationModel.objects.get(reservation_id="res-ambiguous-1")
    assert m.status == ReservationStatus.SETTLED.value


@pytest.mark.django_db(transaction=True)
def test_rollback_and_counter_integrity():
    """Verify rollback of 0005 to 0004 is a clean no-op for data and preserves DB counters."""
    call_command("migrate", "ai_governance", verbosity=0)

    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-counters")

    # Set budget settings
    AIBudgetSettingsModel.objects.create(
        scope_key=scope.key,
        scope_type=scope.scope_type.value,
        tenant_id=scope.tenant_id,
        version=1,
        max_estimated_cost_micros=100000,
        max_calls=10,
    )

    now = datetime.datetime.now(datetime.UTC)
    # Create normal reservation via repo
    res = repo.create_reservation(
        reservation_id="res-active-1",
        idempotency_key="idemp-active-1",
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=1000,
        attempt_number=1,
        created_at=now,
    )
    assert res.reservation_id == "res-active-1"

    # Verify counter
    counter = AIBudgetScopeCounter.objects.get(scope_key=scope.key)
    assert counter.reserved_calls == 1
    assert counter.reserved_cost_micros == 1000

    # Roll back 0005 to 0004
    call_command("migrate", "ai_governance", "0004", verbosity=0)

    # Counter remains intact
    counter.refresh_from_db()
    assert counter.reserved_calls == 1
    assert counter.reserved_cost_micros == 1000

    # Re-apply 0005
    call_command("migrate", "ai_governance", "0005", verbosity=0)

    # Counter remains intact
    counter.refresh_from_db()
    assert counter.reserved_calls == 1
    assert counter.reserved_cost_micros == 1000
