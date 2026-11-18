"""Comprehensive migration regression and financial safety test suite for M0-018B2-B-FIX3 and FIX3-FOLLOWUP.

Tests verify:
1. Fresh install from scratch through 0005.
2. Parameterized step-by-step upgrade starting independently from 0001, 0002, 0003, and 0004 through 0005.
3. Every actual supported dispatch state (not_dispatched, dispatch_started, response_received, response_lost, billing_uncertain)
   and terminal status (pending, settled, released, expired) across upgrades.
4. Execution of real migration 0005 against pending not_dispatched rows (including those created after 0002) proving conservative fencing.
5. End-to-end authorized release rejection and evidence-based settlement on actual migration-fenced rows.
6. Non-empty counters and audit ledger entries preserved across migration and rollback/reapplication.
7. Verification that newly created reservations after upgrade remain fully releasable if not dispatched.
8. Teardown safety: schema always restored to latest even on assertion failure.
"""

import contextlib
import datetime

import pytest
from django.core.management import call_command
from django.db import connection

from flowforge.ai.governance.db.models import (
    AIBudgetScopeCounter,
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


@pytest.fixture(autouse=True)
def restore_latest_schema():
    """Ensure the latest migration state is restored after each test, even if assertions fail."""
    yield
    with contextlib.suppress(Exception):
        call_command("migrate", "ai_governance", verbosity=0)


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
@pytest.mark.parametrize("start_migration", ["0001", "0002", "0003", "0004"])
def test_parameterized_upgrade_from_prior_migrations(start_migration):
    """Independently start at 0001, 0002, 0003, and 0004, insert schema-valid fixtures, and apply through 0005.

    Verifies:
    - pending + not_dispatched is fenced to billing_uncertain by 0005.
    - pending + dispatch_started remains dispatch_started.
    - pending + response_received remains response_received.
    - pending + response_lost remains response_lost.
    - pending + billing_uncertain remains billing_uncertain.
    - settled, released, and expired states remain intact.
    """
    call_command("migrate", "ai_governance", start_migration, verbosity=0)

    now = datetime.datetime.now(datetime.UTC)
    with connection.cursor() as cursor:
        if start_migration == "0001":
            # 0001 has no dispatch_status column and no release_idempotency_key column
            # Insert pending
            cursor.execute(
                """
                INSERT INTO ai_governance_reservation (
                    reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                    provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                    estimated_total_tokens, estimated_cost_micros, attempt_number,
                    status, created_at, audit_metadata
                ) VALUES (
                    'res-0001-pending', 'idem-0001-p', 'tenant:t-001', 'tenant', 't-001',
                    'mock', 'm1', 100, 100, 200, 1000, 1,
                    'pending', %s, '{}'::jsonb
                );
                """,
                [now],
            )
            # Insert settled
            cursor.execute(
                """
                INSERT INTO ai_governance_reservation (
                    reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                    provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                    estimated_total_tokens, estimated_cost_micros, attempt_number,
                    status, created_at, audit_metadata
                ) VALUES (
                    'res-0001-settled', 'idem-0001-s', 'tenant:t-001', 'tenant', 't-001',
                    'mock', 'm1', 100, 100, 200, 1000, 1,
                    'settled', %s, '{}'::jsonb
                );
                """,
                [now],
            )
            # Insert released
            cursor.execute(
                """
                INSERT INTO ai_governance_reservation (
                    reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                    provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                    estimated_total_tokens, estimated_cost_micros, attempt_number,
                    status, created_at, audit_metadata
                ) VALUES (
                    'res-0001-released', 'idem-0001-r', 'tenant:t-001', 'tenant', 't-001',
                    'mock', 'm1', 100, 100, 200, 1000, 1,
                    'released', %s, '{}'::jsonb
                );
                """,
                [now],
            )
            # Insert expired
            cursor.execute(
                """
                INSERT INTO ai_governance_reservation (
                    reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                    provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                    estimated_total_tokens, estimated_cost_micros, attempt_number,
                    status, created_at, audit_metadata
                ) VALUES (
                    'res-0001-expired', 'idem-0001-e', 'tenant:t-001', 'tenant', 't-001',
                    'mock', 'm1', 100, 100, 200, 1000, 1,
                    'expired', %s, '{}'::jsonb
                );
                """,
                [now],
            )
        else:
            # In 0002, 0003, and 0004, dispatch_status exists.
            # In 0004, release_idempotency_key exists.
            has_release_key = start_migration == "0004"
            cols = (
                "reservation_id, idempotency_key, scope_key, scope_type, tenant_id, "
                "provider, model, estimated_prompt_tokens, estimated_completion_tokens, "
                "estimated_total_tokens, estimated_cost_micros, attempt_number, "
                "status, dispatch_status, created_at, audit_metadata"
            )
            if has_release_key:
                cols += ", release_idempotency_key"

            dispatch_states = [
                ("res-p-notdisp", "pending", "not_dispatched"),
                ("res-p-started", "pending", "dispatch_started"),
                ("res-p-received", "pending", "response_received"),
                ("res-p-lost", "pending", "response_lost"),
                ("res-p-uncertain", "pending", "billing_uncertain"),
                ("res-t-settled", "settled", "response_received"),
                ("res-t-released", "released", "not_dispatched"),
                ("res-t-expired", "expired", "not_dispatched"),
            ]
            for r_id, status_val, disp_val in dispatch_states:
                val_placeholders = (
                    "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '{}'::jsonb"
                )
                vals = [
                    f"{r_id}-{start_migration}",
                    f"idem-{r_id}-{start_migration}",
                    "tenant:t-param",
                    "tenant",
                    "t-param",
                    "mock",
                    "m1",
                    100,
                    100,
                    200,
                    1000,
                    1,
                    status_val,
                    disp_val,
                    now,
                ]
                if has_release_key:
                    val_placeholders += ", %s"
                    vals.append(None if status_val != "released" else f"rel-key-{r_id}")

                cursor.execute(
                    f"INSERT INTO ai_governance_reservation ({cols}) VALUES ({val_placeholders});",
                    vals,
                )

    # Apply migration forward to 0005
    call_command("migrate", "ai_governance", verbosity=0)

    if start_migration == "0001":
        # 0001 pending gets default 'not_dispatched' in 0002, then fenced to 'billing_uncertain' in 0005
        p = AIReservationModel.objects.get(reservation_id="res-0001-pending")
        assert p.status == ReservationStatus.PENDING.value
        assert p.dispatch_status == DispatchStatus.BILLING_UNCERTAIN.value

        s = AIReservationModel.objects.get(reservation_id="res-0001-settled")
        assert s.status == ReservationStatus.SETTLED.value
        assert s.dispatch_status == DispatchStatus.NOT_DISPATCHED.value

        r = AIReservationModel.objects.get(reservation_id="res-0001-released")
        assert r.status == ReservationStatus.RELEASED.value
        assert r.dispatch_status == DispatchStatus.NOT_DISPATCHED.value

        e = AIReservationModel.objects.get(reservation_id="res-0001-expired")
        assert e.status == ReservationStatus.EXPIRED.value
        assert e.dispatch_status == DispatchStatus.NOT_DISPATCHED.value
    else:
        # Check every dispatch state
        notdisp = AIReservationModel.objects.get(reservation_id=f"res-p-notdisp-{start_migration}")
        assert notdisp.status == ReservationStatus.PENDING.value
        assert notdisp.dispatch_status == DispatchStatus.BILLING_UNCERTAIN.value

        started = AIReservationModel.objects.get(reservation_id=f"res-p-started-{start_migration}")
        assert started.status == ReservationStatus.PENDING.value
        assert started.dispatch_status == DispatchStatus.DISPATCH_STARTED.value

        received = AIReservationModel.objects.get(
            reservation_id=f"res-p-received-{start_migration}"
        )
        assert received.status == ReservationStatus.PENDING.value
        assert received.dispatch_status == DispatchStatus.RESPONSE_RECEIVED.value

        lost = AIReservationModel.objects.get(reservation_id=f"res-p-lost-{start_migration}")
        assert lost.status == ReservationStatus.PENDING.value
        assert lost.dispatch_status == DispatchStatus.RESPONSE_LOST.value

        uncertain = AIReservationModel.objects.get(
            reservation_id=f"res-p-uncertain-{start_migration}"
        )
        assert uncertain.status == ReservationStatus.PENDING.value
        assert uncertain.dispatch_status == DispatchStatus.BILLING_UNCERTAIN.value

        # Check terminal states
        stl = AIReservationModel.objects.get(reservation_id=f"res-t-settled-{start_migration}")
        assert stl.status == ReservationStatus.SETTLED.value
        assert stl.dispatch_status == DispatchStatus.RESPONSE_RECEIVED.value

        rel = AIReservationModel.objects.get(reservation_id=f"res-t-released-{start_migration}")
        assert rel.status == ReservationStatus.RELEASED.value
        assert rel.dispatch_status == DispatchStatus.NOT_DISPATCHED.value

        exp = AIReservationModel.objects.get(reservation_id=f"res-t-expired-{start_migration}")
        assert exp.status == ReservationStatus.EXPIRED.value
        assert exp.dispatch_status == DispatchStatus.NOT_DISPATCHED.value


@pytest.mark.django_db(transaction=True)
def test_real_migration_fencing_and_lifecycle_operations():
    """Migrate to 0004, seed genuine undispatched pending reservation, apply 0005.

    Proves:
    1. Migration 0005 actively transitions pending not_dispatched to billing_uncertain.
    2. Even if a reservation was never dispatched in reality, insufficient persistent provenance
       causes 0005 to fence it to billing_uncertain (the documented conservative availability tradeoff).
    3. Attempted release through repo.release_reservation() is strictly rejected fail-closed.
    4. Legitimate settlement with actual evidence through repo.settle_reservation() succeeds.
    """
    call_command("migrate", "ai_governance", "0004", verbosity=0)

    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-e2e")

    # Insert schema-valid reservation in 0004 state with initial non-empty counters for tenant AND global scope
    global_scope = BudgetScope(scope_type=BudgetScopeType.GLOBAL)
    with connection.cursor() as cursor:
        for s in (scope, global_scope):
            cursor.execute(
                """
                INSERT INTO ai_governance_scope_counter (
                    scope_key, scope_type, tenant_id,
                    settled_calls, settled_tokens, settled_cost_micros,
                    reserved_calls, reserved_tokens, reserved_cost_micros,
                    updated_at
                ) VALUES (
                    %s, %s, %s,
                    0, 0, 0,
                    1, 200, 1000,
                    %s
                );
                """,
                [s.key, s.scope_type.value, s.tenant_id, now],
            )
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, dispatch_status, created_at, audit_metadata
            ) VALUES (
                'res-e2e-fenced', 'idem-e2e-1', %s, %s, %s,
                'mock', 'm1', 100, 100, 200, 1000, 1,
                'pending', 'not_dispatched', %s, '{}'::jsonb
            );
            """,
            [scope.key, scope.scope_type.value, scope.tenant_id, now],
        )

    # Confirm before 0005 it was not_dispatched
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT dispatch_status FROM ai_governance_reservation WHERE reservation_id = 'res-e2e-fenced';"
        )
        assert cursor.fetchone()[0] == "not_dispatched"

    # Apply real migration 0005
    call_command("migrate", "ai_governance", "0005", verbosity=0)

    # Verify migration 0005 modified the row
    m = AIReservationModel.objects.get(reservation_id="res-e2e-fenced")
    assert m.dispatch_status == DispatchStatus.BILLING_UNCERTAIN.value

    repo = PostgreSQLBudgetRepository()

    # Verify release is rejected fail-closed
    with pytest.raises(AIInvalidReservationStateError):
        repo.release_reservation(
            reservation_id="res-e2e-fenced",
            idempotency_key="rel-e2e-attempt",
            scope=scope,
            released_at=now,
        )

    # Verify settlement with truthful evidence succeeds
    settlement, replayed = repo.settle_reservation(
        settlement_id="stl-e2e-fenced",
        reservation_id="res-e2e-fenced",
        idempotency_key="settle-e2e-1",
        settlement_type=SettlementType.EXACT,
        actual_prompt_tokens=90,
        actual_completion_tokens=90,
        actual_total_tokens=180,
        actual_cost_micros=900,
        attempt_number=1,
        settled_at=now,
    )
    assert not replayed
    assert settlement.actual_cost_micros == 900

    # Verify reservation is settled in database
    m.refresh_from_db()
    assert m.status == ReservationStatus.SETTLED.value

    # Verify scope counter was accurately updated (reserved cleared, settled booked)
    counter = AIBudgetScopeCounter.objects.get(scope_key=scope.key)
    assert counter.reserved_calls == 0
    assert counter.reserved_tokens == 0
    assert counter.reserved_cost_micros == 0
    assert counter.settled_calls == 1
    assert counter.settled_tokens == 180
    assert counter.settled_cost_micros == 900


@pytest.mark.django_db(transaction=True)
def test_nonempty_counters_and_ledger_integrity_across_rollback_and_reapply():
    """Seed consistent nonempty counters and ledger entries.

    Verify:
    1. Migration 0005 does not alter counter values or duplicate ledger entries.
    2. Rollback 0005 -> 0004 preserves counter and ledger entries exactly.
    3. Reapplication 0005 preserves counter and ledger entries exactly.
    4. New normal reservation created after upgrade is NOT fenced and CAN be released.
    """
    call_command("migrate", "ai_governance", "0004", verbosity=0)

    now = datetime.datetime.now(datetime.UTC)
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-integrity")

    # Seed budget settings, counter, and ledger entries
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ai_governance_budget_settings (
                scope_key, scope_type, tenant_id, version,
                max_calls, max_estimated_cost_micros,
                allowed_providers, denied_providers, allow_overrun,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, 1,
                100, 1000000,
                '[]'::jsonb, '[]'::jsonb, false,
                %s, %s
            );
            """,
            [scope.key, scope.scope_type.value, scope.tenant_id, now, now],
        )
        cursor.execute(
            """
            INSERT INTO ai_governance_scope_counter (
                scope_key, scope_type, tenant_id,
                settled_calls, settled_tokens, settled_cost_micros,
                reserved_calls, reserved_tokens, reserved_cost_micros,
                updated_at
            ) VALUES (
                %s, %s, %s,
                5, 1000, 5000,
                2, 400, 2000,
                %s
            );
            """,
            [scope.key, scope.scope_type.value, scope.tenant_id, now],
        )
        cursor.execute(
            """
            INSERT INTO ai_governance_ledger (
                entry_id, event_type, idempotency_key,
                scope_key, scope_type, tenant_id,
                provider, model, timestamp,
                tokens_delta, cost_micros_delta, calls_delta,
                overrun_marker, notes, audit_metadata
            ) VALUES (
                'entry-seed-1', 'settlement', 'idem-ledger-seed-1',
                %s, %s, %s,
                'mock', 'm1', %s,
                1000, 5000, 5,
                'none', 'Initial seed entry', '{}'::jsonb
            );
            """,
            [scope.key, scope.scope_type.value, scope.tenant_id, now],
        )

    # Forward migration to 0005
    call_command("migrate", "ai_governance", "0005", verbosity=0)

    # Verify counter unchanged
    c_0005 = AIBudgetScopeCounter.objects.get(scope_key=scope.key)
    assert (c_0005.settled_calls, c_0005.settled_cost_micros) == (5, 5000)
    assert (c_0005.reserved_calls, c_0005.reserved_cost_micros) == (2, 2000)

    # Verify ledger entries count and content unchanged
    assert AIUsageLedgerModel.objects.count() == 1
    l_0005 = AIUsageLedgerModel.objects.get(entry_id="entry-seed-1")
    assert l_0005.cost_micros_delta == 5000

    # Roll back 0005 -> 0004
    call_command("migrate", "ai_governance", "0004", verbosity=0)

    c_0004 = AIBudgetScopeCounter.objects.get(scope_key=scope.key)
    assert (c_0004.settled_calls, c_0004.settled_cost_micros) == (5, 5000)
    assert (c_0004.reserved_calls, c_0004.reserved_cost_micros) == (2, 2000)
    assert AIUsageLedgerModel.objects.count() == 1

    # Reapply 0005
    call_command("migrate", "ai_governance", "0005", verbosity=0)

    c_reapply = AIBudgetScopeCounter.objects.get(scope_key=scope.key)
    assert (c_reapply.settled_calls, c_reapply.settled_cost_micros) == (5, 5000)
    assert (c_reapply.reserved_calls, c_reapply.reserved_cost_micros) == (2, 2000)
    assert AIUsageLedgerModel.objects.count() == 1

    # Verify that a brand new reservation created on latest schema:
    # 1. Defaults to NOT_DISPATCHED
    # 2. Can be normally released at zero cost
    repo = PostgreSQLBudgetRepository()
    new_res = repo.create_reservation(
        reservation_id="res-post-upgrade-active",
        idempotency_key="idem-post-upgrade-1",
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=50,
        estimated_completion_tokens=50,
        estimated_total_tokens=100,
        estimated_cost_micros=500,
        attempt_number=1,
        created_at=now,
    )
    assert new_res.dispatch_status == DispatchStatus.NOT_DISPATCHED

    # Release succeeding without error
    rel_rec, replayed = repo.release_reservation(
        reservation_id="res-post-upgrade-active",
        idempotency_key="rel-post-upgrade-1",
        scope=scope,
        released_at=now,
    )
    assert not replayed
    assert rel_rec.status == ReservationStatus.RELEASED
