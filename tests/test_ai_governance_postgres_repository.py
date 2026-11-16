"""RED test suite verifying PostgreSQLBudgetRepository parity with InMemoryBudgetRepository."""

import datetime

import pytest

from flowforge.ai.governance.exceptions import (
    AIDuplicateSettlementError,
)
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
    ReservationStatus,
    SettlementType,
)


@pytest.mark.django_db(transaction=True)
def test_postgres_budget_repository_basic_lifecycle():
    from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository

    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-alpha")

    # 1. Settings
    settings = BudgetSettings(
        scope=scope,
        max_calls=10,
        max_total_tokens=1000,
        max_estimated_cost_micros=50000,
    )
    repo.set_budget_settings(settings)
    fetched_settings = repo.get_budget_settings(scope)
    assert fetched_settings is not None
    assert fetched_settings.max_calls == 10
    assert fetched_settings.max_total_tokens == 1000

    # 2. Reservation
    now = datetime.datetime.now(datetime.UTC)
    res, is_replayed = repo.reserve_atomic(
        reservation_id="res-001",
        idempotency_key="idemp-res-001",
        scope=scope,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=200,
        estimated_total_tokens=300,
        estimated_cost_micros=10000,
        attempt_number=1,
        created_at=now,
    )
    assert not is_replayed
    assert res.reservation_id == "res-001"
    assert res.status == ReservationStatus.PENDING

    # Counters check
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope)
    assert (active_calls, active_tokens, active_cost) == (1, 300, 10000)

    # Replay
    res_replayed, is_replayed_2 = repo.reserve_atomic(
        reservation_id="res-001-alt",
        idempotency_key="idemp-res-001",
        scope=scope,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=200,
        estimated_total_tokens=300,
        estimated_cost_micros=10000,
        attempt_number=1,
        created_at=now,
    )
    assert is_replayed_2
    assert res_replayed.reservation_id == "res-001"

    # Conflicting replay
    with pytest.raises(AIDuplicateSettlementError):
        repo.reserve_atomic(
            reservation_id="res-002",
            idempotency_key="idemp-res-001",
            scope=scope,
            ancestor_scopes=[],
            provider="ollama",
            model="llama3-different",
            estimated_prompt_tokens=100,
            estimated_completion_tokens=200,
            estimated_total_tokens=300,
            estimated_cost_micros=10000,
            attempt_number=1,
            created_at=now,
        )

    # 3. Settle
    settle_time = datetime.datetime.now(datetime.UTC)
    settlement, s_replayed = repo.settle_reservation(
        settlement_id="stl-001",
        reservation_id="res-001",
        idempotency_key="idemp-stl-001",
        settlement_type=SettlementType.EXACT,
        actual_prompt_tokens=90,
        actual_completion_tokens=210,
        actual_total_tokens=300,
        actual_cost_micros=9500,
        attempt_number=1,
        settled_at=settle_time,
    )
    assert not s_replayed
    assert settlement.actual_cost_micros == 9500

    # Counters after settlement
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope)
    assert (active_calls, active_tokens, active_cost) == (0, 0, 0)
    total_calls, total_tokens, total_cost = repo.get_aggregate_usage(scope)
    assert (total_calls, total_tokens, total_cost) == (1, 300, 9500)
