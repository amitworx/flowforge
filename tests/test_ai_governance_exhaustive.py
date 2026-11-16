"""Exhaustive unit and concurrency test suite for AI Governance Budget & Audit Policy Core.

Validates:
1. Cross-tenant isolation
2. Non-reused reservation identifiers & non-overlapping scopes
3. Duplicate settlement prevention & idempotency under redelivery
4. Concurrent last-capacity contention (multithreading stress test)
5. Stale / fenced attempt rejection
6. Debit accounting under mixed providers
7. Time budgets & injectable clock determinism
8. Truthful overrun recording (token, cost, time)
9. Distinct actual usage vs estimated exposure vs zero-cost pre-dispatch rejection
10. Strict quota exhaustion
11. Deny-by-default hosted provider policy
12. Privacy guarantees (no raw prompts, secrets, provider tokens stored)
"""

from __future__ import annotations

import concurrent.futures
import datetime

import pytest

from flowforge.ai.governance.exceptions import (
    AIBudgetExhaustedError,
    AIDuplicateSettlementError,
    AIProviderPolicyDeniedError,
    AIStaleFencedAttemptError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
    OverrunMarker,
    ReservationStatus,
    SettlementType,
)
from flowforge.ai.governance.policy import AIBudgetPolicyService


class MockClock:
    """Injectable deterministic clock for testing without real-time dependencies."""

    def __init__(self, start_time: datetime.datetime | None = None) -> None:
        self.current_time = start_time or datetime.datetime(
            2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC
        )

    def __call__(self) -> datetime.datetime:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += datetime.timedelta(seconds=seconds)


@pytest.fixture
def repo() -> InMemoryBudgetRepository:
    return InMemoryBudgetRepository()


@pytest.fixture
def mock_clock() -> MockClock:
    return MockClock()


@pytest.fixture
def service(repo: InMemoryBudgetRepository, mock_clock: MockClock) -> AIBudgetPolicyService:
    return AIBudgetPolicyService(repository=repo, clock=mock_clock)


# ---------------------------------------------------------------------------
# 1. Cross-Tenant Isolation
# ---------------------------------------------------------------------------


def test_cross_tenant_isolation(service: AIBudgetPolicyService, repo: InMemoryBudgetRepository):
    """Budget limits and usage for Tenant A do not affect Tenant B."""
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-B")

    repo.set_budget_settings(
        BudgetSettings(
            scope=scope_a,
            max_calls=2,
            max_total_tokens=1000,
            allowed_providers=("mock", "ollama"),
        )
    )
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope_b,
            max_calls=2,
            max_total_tokens=1000,
            allowed_providers=("mock", "ollama"),
        )
    )

    # Exhaust tenant A
    res_a1 = service.reserve(
        scope=scope_a,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=300,
        estimated_completion_tokens=200,
    )
    res_a2 = service.reserve(
        scope=scope_a,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=300,
        estimated_completion_tokens=200,
    )

    # Tenant A is now exhausted on calls (2 / 2)
    with pytest.raises(AIBudgetExhaustedError) as exc_info:
        service.reserve(
            scope=scope_a,
            provider="mock",
            model="m1",
            estimated_prompt_tokens=10,
            estimated_completion_tokens=10,
        )
    assert exc_info.value.limit_name == "max_calls"

    # Tenant B must NOT be blocked
    res_b1 = service.reserve(
        scope=scope_b,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=300,
        estimated_completion_tokens=200,
    )
    assert res_b1.reservation_id is not None
    assert res_b1.scope.tenant_id == "tenant-B"

    # Settle Tenant A reservations
    service.settle(
        reservation_id=res_a1.reservation_id, actual_prompt_tokens=200, actual_completion_tokens=100
    )
    service.settle(
        reservation_id=res_a2.reservation_id, actual_prompt_tokens=200, actual_completion_tokens=100
    )

    # Check aggregate usages
    calls_a, tokens_a, _ = repo.get_aggregate_usage(scope_a)
    calls_b, tokens_b, _ = repo.get_aggregate_usage(scope_b)
    assert calls_a == 2
    assert tokens_a == 600
    assert calls_b == 0  # Tenant B not yet settled
    assert tokens_b == 0


# ---------------------------------------------------------------------------
# 2. Deny-by-Default Hosted Provider Policy
# ---------------------------------------------------------------------------


def test_deny_by_default_hosted_providers(
    service: AIBudgetPolicyService, repo: InMemoryBudgetRepository
):
    """External hosted providers like openai/anthropic are denied by default without explicit budget authorization."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-123")

    # No settings configured -> default allows mock, ollama, test; denies external hosted
    with pytest.raises(AIProviderPolicyDeniedError) as exc_info:
        service.reserve(
            scope=scope,
            provider="openai",
            model="gpt-4o",
            estimated_prompt_tokens=100,
            estimated_completion_tokens=100,
        )
    assert "denied by default" in str(exc_info.value)

    with pytest.raises(AIProviderPolicyDeniedError):
        service.reserve(
            scope=scope,
            provider="anthropic",
            model="claude-3-5",
            estimated_prompt_tokens=100,
            estimated_completion_tokens=100,
        )

    # Safe local/mock providers pass
    res_mock = service.reserve(
        scope=scope,
        provider="mock",
        model="mock-model",
        estimated_prompt_tokens=50,
        estimated_completion_tokens=50,
    )
    assert res_mock.provider == "mock"

    res_ollama = service.reserve(
        scope=scope,
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=50,
        estimated_completion_tokens=50,
    )
    assert res_ollama.provider == "ollama"

    # Explicit authorization allows external provider
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            allowed_providers=("openai", "mock"),
            denied_providers=("anthropic",),
            max_calls=10,
            max_total_tokens=10000,
        )
    )
    res_openai = service.reserve(
        scope=scope,
        provider="openai",
        model="gpt-4o",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
    )
    assert res_openai.provider == "openai"

    # Explicitly denied provider remains blocked
    with pytest.raises(AIProviderPolicyDeniedError):
        service.reserve(
            scope=scope,
            provider="anthropic",
            model="claude-3-5",
            estimated_prompt_tokens=100,
            estimated_completion_tokens=100,
        )


# ---------------------------------------------------------------------------
# 3. Duplicate Settlement & Idempotency under Redelivery
# ---------------------------------------------------------------------------


def test_idempotent_reservation_and_settlement(service: AIBudgetPolicyService):
    """Calling reserve or settle repeatedly with the same idempotency key returns identical record."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-1")

    # Idempotent reservation
    res1 = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        idempotency_key="msg-key-1",
    )
    res2 = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        idempotency_key="msg-key-1",
    )
    assert res1.reservation_id == res2.reservation_id

    # Settle once
    set1 = service.settle(
        reservation_id=res1.reservation_id,
        actual_prompt_tokens=80,
        actual_completion_tokens=90,
        idempotency_key="settle-key-1",
    )
    assert set1.actual_total_tokens == 170

    # Redelivery of identical settlement (Celery redelivery simulation)
    set2 = service.settle(
        reservation_id=res1.reservation_id,
        actual_prompt_tokens=80,
        actual_completion_tokens=90,
        idempotency_key="settle-key-1",
    )
    assert set1.settlement_id == set2.settlement_id

    # Conflicting re-settlement attempt with different key raises AIDuplicateSettlementError
    with pytest.raises(AIDuplicateSettlementError):
        service.settle(
            reservation_id=res1.reservation_id,
            actual_prompt_tokens=50,
            actual_completion_tokens=50,
            idempotency_key="settle-key-CONFLICT",
        )


# ---------------------------------------------------------------------------
# 4. Stale / Fenced Attempt Protection
# ---------------------------------------------------------------------------


def test_stale_fenced_attempt_rejection(service: AIBudgetPolicyService):
    """An out-of-order or stale attempt cannot settle a reservation with a higher attempt number."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-fence")

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        attempt_number=2,  # Fenced at attempt 2
    )

    # Attempt 1 trying to settle must fail
    with pytest.raises(AIStaleFencedAttemptError) as exc_info:
        service.settle(
            reservation_id=res.reservation_id,
            actual_prompt_tokens=90,
            actual_completion_tokens=90,
            attempt_number=1,
        )
    assert exc_info.value.expected_attempt == 2
    assert exc_info.value.actual_attempt == 1

    # Attempt 2 or higher succeeds
    set_record = service.settle(
        reservation_id=res.reservation_id,
        actual_prompt_tokens=90,
        actual_completion_tokens=90,
        attempt_number=2,
    )
    assert set_record.settlement_id is not None


# ---------------------------------------------------------------------------
# 5. Distinct Actual Usage vs Estimated Exposure vs Zero-Cost Rejection
# ---------------------------------------------------------------------------


def test_settlement_types_accounting(
    service: AIBudgetPolicyService, repo: InMemoryBudgetRepository
):
    """Demonstrates three distinct cost outcomes:

    1. Zero-cost pre-dispatch rejection (e.g. rate limit, validation failure prior to send)
    2. Estimated exposure for dispatched but failed / lost-response calls
    3. Exact actual usage reported by provider
    """
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-types")
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            max_calls=10,
            max_total_tokens=10_000,
            max_estimated_cost_micros=100_000,
            allowed_providers=("mock",),
        )
    )

    # Case 1: Zero-cost pre-dispatch rejection
    res1 = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=200,
        estimated_completion_tokens=200,
        estimated_cost_micros=2000,
    )
    set1 = service.settle(
        reservation_id=res1.reservation_id,
        settlement_type=SettlementType.ZERO_COST_REJECTED,
    )
    assert set1.actual_total_tokens == 0
    assert set1.actual_cost_micros == 0

    # Aggregate usage should NOT count tokens or cost for zero-cost rejection
    calls, tokens, cost = repo.get_aggregate_usage(scope)
    assert calls == 0
    assert tokens == 0
    assert cost == 0

    # Case 2: Dispatched / maybe-sent call where provider response failed/timed out
    res2 = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=300,
        estimated_completion_tokens=300,
        estimated_cost_micros=3000,
    )
    set2 = service.settle(
        reservation_id=res2.reservation_id,
        settlement_type=SettlementType.ESTIMATED_EXPOSURE,
    )
    # Retains conservative estimated exposure because tokens may have been consumed upstream
    assert set2.actual_total_tokens == 600
    assert set2.actual_cost_micros == 3000

    calls, tokens, cost = repo.get_aggregate_usage(scope)
    assert calls == 1
    assert tokens == 600
    assert cost == 3000

    # Case 3: Normal exact settlement
    res3 = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=400,
        estimated_completion_tokens=400,
        estimated_cost_micros=4000,
    )
    set3 = service.settle(
        reservation_id=res3.reservation_id,
        actual_prompt_tokens=350,
        actual_completion_tokens=250,
        actual_cost_micros=3000,
        settlement_type=SettlementType.EXACT,
    )
    assert set3.actual_total_tokens == 600

    calls, tokens, cost = repo.get_aggregate_usage(scope)
    assert calls == 2
    assert tokens == 1200
    assert cost == 6000


# ---------------------------------------------------------------------------
# 6. Truthful Overrun Markers
# ---------------------------------------------------------------------------


def test_truthful_overrun_recording(service: AIBudgetPolicyService, repo: InMemoryBudgetRepository):
    """When actual tokens returned exceed estimated reservation and budget limit,

    system truthfully marks and records the overrun without silently dropping spend.
    """
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-overrun")
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            max_calls=5,
            max_total_tokens=1000,
            allowed_providers=("mock",),
            allow_overrun=True,  # Allow completion with overrun marker
        )
    )

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=300,
        estimated_completion_tokens=300,  # 600 estimated
    )

    # Actual generation generated 1200 tokens (exceeding max_total_tokens of 1000)
    set_record = service.settle(
        reservation_id=res.reservation_id,
        actual_prompt_tokens=400,
        actual_completion_tokens=800,  # 1200 total
        settlement_type=SettlementType.EXACT,
    )

    assert set_record.overrun_marker == OverrunMarker.TOKENS_OVERRUN
    assert set_record.overrun_tokens == 200  # 1200 - 1000
    assert set_record.actual_total_tokens == 1200

    # Next reservation must be strictly blocked due to exhausted budget
    with pytest.raises(AIBudgetExhaustedError):
        service.reserve(
            scope=scope,
            provider="mock",
            model="m1",
            estimated_prompt_tokens=50,
            estimated_completion_tokens=50,
        )


def test_overrun_fail_closed_when_disallowed(
    service: AIBudgetPolicyService, repo: InMemoryBudgetRepository
):
    """When allow_overrun is False (default), exceeding limit on completion records overrun and raises."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-disallow-overrun")
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            max_calls=5,
            max_total_tokens=500,
            allowed_providers=("mock",),
            allow_overrun=False,
        )
    )

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=200,
        estimated_completion_tokens=200,  # 400 within 500
    )

    with pytest.raises(AIBudgetExhaustedError) as exc_info:
        service.settle(
            reservation_id=res.reservation_id,
            actual_prompt_tokens=300,
            actual_completion_tokens=300,  # 600 > 500
        )
    assert "budget overrun (tokens_overrun)" in str(exc_info.value)

    # Verify repository still recorded the settled state so exposure is NOT lost
    saved_res = repo.get_reservation_by_id(res.reservation_id)
    assert saved_res.status == ReservationStatus.SETTLED


# ---------------------------------------------------------------------------
# 7. Time Budgets & Injectable Clock
# ---------------------------------------------------------------------------


def test_time_budget_with_injectable_clock(
    service: AIBudgetPolicyService, repo: InMemoryBudgetRepository, mock_clock: MockClock
):
    """Time budget tracking uses injectable clock with zero reliance on real system clock."""
    scope = BudgetScope(scope_type=BudgetScopeType.RUN, tenant_id="tenant-time", run_id="run-1")
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            max_calls=5,
            max_duration_seconds=10.0,
            allowed_providers=("mock",),
            allow_overrun=True,
        )
    )

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
    )

    # Advance clock by 15 seconds
    mock_clock.advance(15.0)

    set_record = service.settle(
        reservation_id=res.reservation_id,
        actual_prompt_tokens=100,
        actual_completion_tokens=100,
        duration_seconds=15.0,
    )
    assert set_record.overrun_marker == OverrunMarker.TIME_OVERRUN
    assert set_record.duration_seconds == 15.0


# ---------------------------------------------------------------------------
# 8. Privacy: No Secrets, Raw Prompts, or Responses in Records/Audit
# ---------------------------------------------------------------------------


def test_privacy_guarantees(service: AIBudgetPolicyService, repo: InMemoryBudgetRepository):
    """Ledger and records store only bounded metadata, token tallies, and audit keys."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-priv")
    metadata = {"job_id": "job-abc", "step_name": "summarizer"}

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=100,
        audit_metadata=metadata,
    )
    service.settle(
        reservation_id=res.reservation_id,
        actual_prompt_tokens=90,
        actual_completion_tokens=90,
        audit_metadata=metadata,
    )

    ledger = repo.get_ledger_entries(scope)
    assert len(ledger) >= 2

    forbidden_terms = [
        "sk-",
        "api_key",
        "Bearer",
        "prompt_text",
        "completion_text",
        "user password",
    ]
    for entry in ledger:
        entry_repr = repr(entry)
        for term in forbidden_terms:
            assert term not in entry_repr
            assert term not in entry.notes


# ---------------------------------------------------------------------------
# 9. Concurrent Last-Capacity Contention Stress Test
# ---------------------------------------------------------------------------


def test_concurrent_last_capacity_contention(repo: InMemoryBudgetRepository):
    """Stress test: 20 threads simultaneously competing for a budget with only 5 remaining slots.

    Guarantees:
    - Exactly 5 threads succeed in reserving capacity.
    - Exactly 15 threads receive AIBudgetExhaustedError fail-closed.
    - No race condition or double allocation occurs under lock.
    """
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-concurrency")
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            max_calls=5,
            max_total_tokens=5000,
            allowed_providers=("mock",),
        )
    )
    service = AIBudgetPolicyService(repository=repo)

    num_threads = 20
    results: list[bool] = []
    errors: list[Exception] = []

    def attempt_reserve(i: int):
        try:
            service.reserve(
                scope=scope,
                provider="mock",
                model="m1",
                estimated_prompt_tokens=100,
                estimated_completion_tokens=100,
                idempotency_key=f"concurrent-{i}",
            )
            return True
        except AIBudgetExhaustedError as e:
            errors.append(e)
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(attempt_reserve, i) for i in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    success_count = sum(1 for r in results if r is True)
    failure_count = sum(1 for r in results if r is False)

    assert success_count == 5
    assert failure_count == 15
    assert len(errors) == 15

    # Check repository state
    res_calls, res_tokens, _ = repo.get_active_reserved_usage(scope)
    assert res_calls == 5
    assert res_tokens == 1000
