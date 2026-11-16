"""Test suite for crash recovery, uncertain provider billing, and reconciliation invariants."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    ReservationStatus,
    SettlementType,
    UsageLedgerEntry,
)


@pytest.mark.django_db(transaction=True)
def test_uncertain_provider_billing_conservative_settlement():
    """Verify that when a provider response is lost or billing is uncertain,

    the reservation settles with ESTIMATED_EXPOSURE, deducting reserved exposure
    and booking conservative estimates into settled counters without financial leakage.
    """
    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-uncertain")
    now = datetime.datetime.now(datetime.UTC)

    # 1. Create reservation for 1000 prompt tokens + 2000 completion tokens = 3000 tokens ($0.15)
    res, _ = repo.reserve_atomic(
        reservation_id="res-timeout-01",
        idempotency_key="idemp-timeout-01",
        scope=scope,
        ancestor_scopes=[],
        provider="openai",
        model="gpt-4o",
        estimated_prompt_tokens=1000,
        estimated_completion_tokens=2000,
        estimated_total_tokens=3000,
        estimated_cost_micros=150000,
        attempt_number=1,
        created_at=now,
    )

    # Check active reserved tally
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope)
    assert (active_calls, active_tokens, active_cost) == (1, 3000, 150000)

    # 2. Settle with ESTIMATED_EXPOSURE due to network timeout after dispatch
    settle_time = datetime.datetime.now(datetime.UTC)
    settlement, is_replayed = repo.settle_reservation(
        settlement_id="stl-uncertain-01",
        reservation_id="res-timeout-01",
        idempotency_key="idemp-stl-uncertain-01",
        settlement_type=SettlementType.ESTIMATED_EXPOSURE,
        actual_prompt_tokens=1000,
        actual_completion_tokens=2000,
        actual_total_tokens=3000,
        actual_cost_micros=150000,
        attempt_number=1,
        settled_at=settle_time,
        audit_metadata={"status": "failed", "reason": "timeout"},
    )

    assert not is_replayed
    assert settlement.settlement_type == SettlementType.ESTIMATED_EXPOSURE

    # Reserved exposure is cleared
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope)
    assert (active_calls, active_tokens, active_cost) == (0, 0, 0)

    # Settled usage is truthfully booked to conservative exposure
    settled_calls, settled_tokens, settled_cost = repo.get_aggregate_usage(scope)
    assert (settled_calls, settled_tokens, settled_cost) == (1, 3000, 150000)


@pytest.mark.django_db(transaction=True)
def test_pre_dispatch_cancellation_zero_cost_release():
    """Verify that a pre-dispatch rejection or local cancellation safely releases

    reserved capacity without charging settled counters.
    """
    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-predispatch")
    now = datetime.datetime.now(datetime.UTC)

    repo.reserve_atomic(
        reservation_id="res-cancel-01",
        idempotency_key="idemp-cancel-01",
        scope=scope,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=500,
        estimated_completion_tokens=500,
        estimated_total_tokens=1000,
        estimated_cost_micros=10000,
        attempt_number=1,
        created_at=now,
    )

    # Pre-dispatch release
    released, _ = repo.release_reservation(
        reservation_id="res-cancel-01",
        idempotency_key="idemp-cancel-01",
        released_at=datetime.datetime.now(datetime.UTC),
        reason="user_cancelled",
    )
    assert released.status == ReservationStatus.RELEASED

    # Verify counters: zero reserved, zero settled
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope)
    settled_calls, settled_tokens, settled_cost = repo.get_aggregate_usage(scope)
    assert (active_calls, active_tokens, active_cost) == (0, 0, 0)
    assert (settled_calls, settled_tokens, settled_cost) == (0, 0, 0)


@pytest.mark.django_db(transaction=True)
def test_counter_and_ledger_reconciliation_invariant():
    """Verify that after multiple operations, the sum of ledger deltas equals the counter tallies."""
    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-reconcile")
    now = datetime.datetime.now(datetime.UTC)

    # Operation 1: Reservation + Exact Settlement
    repo.reserve_atomic(
        reservation_id="res-rec-1",
        idempotency_key="idemp-rec-1",
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
    ledger_stl_1 = UsageLedgerEntry(
        entry_id="led-stl-1",
        event_type="settlement",
        idempotency_key="idemp-stl-rec-1",
        scope=scope,
        provider="ollama",
        model="llama3",
        timestamp=now,
        tokens_delta=300,
        cost_micros_delta=5000,
        calls_delta=1,
    )
    repo.settle_reservation(
        settlement_id="stl-rec-1",
        reservation_id="res-rec-1",
        idempotency_key="idemp-stl-rec-1",
        settlement_type=SettlementType.EXACT,
        actual_prompt_tokens=100,
        actual_completion_tokens=200,
        actual_total_tokens=300,
        actual_cost_micros=5000,
        attempt_number=1,
        settled_at=now,
        ledger_entry=ledger_stl_1,
    )

    # Operation 2: Reservation + Overrun Settlement
    repo.reserve_atomic(
        reservation_id="res-rec-2",
        idempotency_key="idemp-rec-2",
        scope=scope,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        estimated_total_tokens=200,
        estimated_cost_micros=3000,
        attempt_number=1,
        created_at=now,
    )
    ledger_stl_2 = UsageLedgerEntry(
        entry_id="led-stl-2",
        event_type="settlement",
        idempotency_key="idemp-stl-rec-2",
        scope=scope,
        provider="ollama",
        model="llama3",
        timestamp=now,
        tokens_delta=350,  # 150 overrun tokens
        cost_micros_delta=5500,  # 2500 overrun cost
        calls_delta=1,
    )
    repo.settle_reservation(
        settlement_id="stl-rec-2",
        reservation_id="res-rec-2",
        idempotency_key="idemp-stl-rec-2",
        settlement_type=SettlementType.EXACT,
        actual_prompt_tokens=150,
        actual_completion_tokens=200,
        actual_total_tokens=350,
        actual_cost_micros=5500,
        attempt_number=1,
        settled_at=now,
        ledger_entry=ledger_stl_2,
    )

    # Operation 3: Reservation + Release
    repo.reserve_atomic(
        reservation_id="res-rec-3",
        idempotency_key="idemp-rec-3",
        scope=scope,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=500,
        estimated_completion_tokens=500,
        estimated_total_tokens=1000,
        estimated_cost_micros=15000,
        attempt_number=1,
        created_at=now,
    )
    repo.release_reservation(
        reservation_id="res-rec-3",
        idempotency_key="idemp-rec-3",
        released_at=now,
    )

    # Invariant Verification
    settled_calls, settled_tokens, settled_cost = repo.get_aggregate_usage(scope)
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope)

    assert active_calls == 0
    assert active_tokens == 0
    assert active_cost == 0

    assert settled_calls == 2
    assert settled_tokens == 300 + 350
    assert settled_cost == 5000 + 5500

    # Query ledger and sum deltas
    ledger_entries = repo.get_ledger_entries(scope)
    sum_tokens = sum(e.tokens_delta for e in ledger_entries)
    sum_cost = sum(e.cost_micros_delta for e in ledger_entries)
    sum_calls = sum(e.calls_delta for e in ledger_entries)

    assert sum_tokens == settled_tokens
    assert sum_cost == settled_cost
    assert sum_calls == settled_calls
