"""Comprehensive parity test suite comparing PostgreSQLBudgetRepository and InMemoryBudgetRepository."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import (
    AIBudgetExhaustedError,
    AIDuplicateSettlementError,
    AIInvalidReservationStateError,
    AIStaleFencedAttemptError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
    ReservationStatus,
    SettlementType,
    UsageLedgerEntry,
)


@pytest.fixture(params=["in_memory", "postgres"])
def repo(request):
    if request.param == "in_memory":
        return InMemoryBudgetRepository()
    return PostgreSQLBudgetRepository()


@pytest.mark.django_db(transaction=True)
def test_budget_settings_crud(repo):
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-settings")
    assert repo.get_budget_settings(scope) is None

    settings = BudgetSettings(
        scope=scope,
        version=2,
        max_calls=100,
        max_total_tokens=50000,
        max_prompt_tokens=20000,
        max_completion_tokens=30000,
        max_estimated_cost_micros=1000000,
        max_duration_seconds=30.0,
        allowed_providers=("mock", "ollama"),
        denied_providers=("openai",),
        allow_overrun=True,
    )
    repo.set_budget_settings(settings)

    fetched = repo.get_budget_settings(scope)
    assert fetched is not None
    assert fetched.version == 2
    assert fetched.max_calls == 100
    assert fetched.max_total_tokens == 50000
    assert fetched.max_prompt_tokens == 20000
    assert fetched.max_completion_tokens == 30000
    assert fetched.max_estimated_cost_micros == 1000000
    assert fetched.max_duration_seconds == 30.0
    assert fetched.allowed_providers == ("mock", "ollama")
    assert fetched.denied_providers == ("openai",)
    assert fetched.allow_overrun is True


@pytest.mark.django_db(transaction=True)
def test_budget_limits_exceeded(repo):
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-limits")
    settings = BudgetSettings(scope=scope, max_calls=1, max_total_tokens=500)
    repo.set_budget_settings(settings)

    now = datetime.datetime.now(datetime.UTC)
    # First reservation succeeds
    repo.reserve_atomic(
        reservation_id="res-1",
        idempotency_key="idemp-1",
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

    # Second reservation exceeds max_calls
    with pytest.raises(AIBudgetExhaustedError):
        repo.reserve_atomic(
            reservation_id="res-2",
            idempotency_key="idemp-2",
            scope=scope,
            ancestor_scopes=[],
            provider="ollama",
            model="llama3",
            estimated_prompt_tokens=100,
            estimated_completion_tokens=50,
            estimated_total_tokens=150,
            estimated_cost_micros=2000,
            attempt_number=1,
            created_at=now,
        )


@pytest.mark.django_db(transaction=True)
def test_idempotency_strict_replay_and_conflict(repo):
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-idemp")
    now = datetime.datetime.now(datetime.UTC)

    res, is_replayed = repo.reserve_atomic(
        reservation_id="res-orig",
        idempotency_key="idemp-key-shared",
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
    assert not is_replayed

    # Exact replay
    replayed_res, is_replayed = repo.reserve_atomic(
        reservation_id="res-different-id",
        idempotency_key="idemp-key-shared",
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
    assert is_replayed
    assert replayed_res.reservation_id == "res-orig"

    # Conflicting replay (tokens modified)
    with pytest.raises(AIDuplicateSettlementError):
        repo.reserve_atomic(
            reservation_id="res-diff-id-2",
            idempotency_key="idemp-key-shared",
            scope=scope,
            ancestor_scopes=[],
            provider="ollama",
            model="llama3",
            estimated_prompt_tokens=101,  # modified
            estimated_completion_tokens=200,
            estimated_total_tokens=301,
            estimated_cost_micros=5000,
            attempt_number=1,
            created_at=now,
        )


@pytest.mark.django_db(transaction=True)
def test_tenant_isolation_cross_tenant_rejection(repo):
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-alpha")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-beta")
    now = datetime.datetime.now(datetime.UTC)

    repo.reserve_atomic(
        reservation_id="res-tenant-a",
        idempotency_key="idemp-shared-token",
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

    # Tenant B tries to use the same idempotency key
    with pytest.raises(AIDuplicateSettlementError):
        repo.reserve_atomic(
            reservation_id="res-tenant-b",
            idempotency_key="idemp-shared-token",
            scope=scope_b,
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

    # Scoped lookup by tenant B must return None to prevent cross-tenant enumeration
    assert repo.get_reservation_by_idempotency_key("idemp-shared-token", scope=scope_b) is None


@pytest.mark.django_db(transaction=True)
def test_settlement_fencing_token_mismatch(repo):
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-fencing")
    now = datetime.datetime.now(datetime.UTC)

    repo.reserve_atomic(
        reservation_id="res-fence",
        idempotency_key="idemp-fence",
        scope=scope,
        ancestor_scopes=[],
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=200,
        estimated_total_tokens=300,
        estimated_cost_micros=5000,
        attempt_number=2,  # Attempt 2
        created_at=now,
    )

    # Attempt number 1 should fail fencing check
    with pytest.raises(AIStaleFencedAttemptError):
        repo.settle_reservation(
            settlement_id="stl-fence-1",
            reservation_id="res-fence",
            idempotency_key="idemp-stl-fence-1",
            settlement_type=SettlementType.EXACT,
            actual_prompt_tokens=100,
            actual_completion_tokens=200,
            actual_total_tokens=300,
            actual_cost_micros=5000,
            attempt_number=1,  # Stale attempt
            settled_at=now,
        )


@pytest.mark.django_db(transaction=True)
def test_release_and_state_invalidation(repo):
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-release")
    now = datetime.datetime.now(datetime.UTC)

    res, _ = repo.reserve_atomic(
        reservation_id="res-rel",
        idempotency_key="idemp-rel",
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
    assert res.status == ReservationStatus.PENDING

    # Release
    released, is_replayed = repo.release_reservation(
        reservation_id="res-rel",
        idempotency_key="idemp-rel",
        scope=scope,
        released_at=now,
    )
    assert not is_replayed
    assert released.status == ReservationStatus.RELEASED

    # Release replay
    released_2, is_replayed_2 = repo.release_reservation(
        reservation_id="res-rel",
        idempotency_key="idemp-rel",
        scope=scope,
        released_at=now,
    )
    assert is_replayed_2
    assert released_2.status == ReservationStatus.RELEASED

    # Cannot settle after release
    with pytest.raises(AIInvalidReservationStateError):
        repo.settle_reservation(
            settlement_id="stl-after-rel",
            reservation_id="res-rel",
            idempotency_key="idemp-stl-rel",
            settlement_type=SettlementType.EXACT,
            actual_prompt_tokens=100,
            actual_completion_tokens=200,
            actual_total_tokens=300,
            actual_cost_micros=5000,
            attempt_number=1,
            settled_at=now,
        )


@pytest.mark.django_db(transaction=True)
def test_immutable_ledger_entries_and_ordering(repo):
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-ledger")
    t1 = datetime.datetime(2026, 11, 1, 10, 0, 0, tzinfo=datetime.UTC)
    t2 = datetime.datetime(2026, 11, 1, 10, 5, 0, tzinfo=datetime.UTC)

    e1 = UsageLedgerEntry(
        entry_id="e-1",
        event_type="reservation",
        idempotency_key="k-1",
        scope=scope,
        provider="ollama",
        model="llama3",
        timestamp=t1,
        tokens_delta=300,
        cost_micros_delta=5000,
        calls_delta=1,
        notes="First entry",
    )
    e2 = UsageLedgerEntry(
        entry_id="e-2",
        event_type="settlement",
        idempotency_key="k-2",
        scope=scope,
        provider="ollama",
        model="llama3",
        timestamp=t2,
        tokens_delta=300,
        cost_micros_delta=5000,
        calls_delta=1,
        notes="Second entry",
    )
    repo.append_ledger_entry(e1)
    repo.append_ledger_entry(e2)

    entries = repo.get_ledger_entries(scope)
    assert len(entries) == 2
    assert entries[0].entry_id == "e-1"
    assert entries[1].entry_id == "e-2"
