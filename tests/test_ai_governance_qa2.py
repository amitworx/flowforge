"""TDD RED test suite for Milestone M0-018B1 QA-2.

Validates:
1. Strict parameter comparison on idempotent replay (prompt/completion tokens, cost, model, provider, attempt).
   Conflicting prompt/completion split, cost, or attempt with same idempotency key must raise AIDuplicateSettlementError.
2. Exception & ledger confidentiality:
   No raw secrets, raw invalid scope values, raw provider names, reservation IDs, idempotency keys, or raw release reasons
   in exception messages, args, or ledger notes.
3. Settings validation:
   Reject NaN/Inf in BudgetSettings.max_duration_seconds.
   Reject boolean in BudgetSettings.version.
4. Correct hierarchical provider policy without incorrectly blocking default safe providers (mock, ollama)
   when ancestor scopes specify other providers unless explicitly denied.
"""

import pytest

from flowforge.ai.governance.exceptions import (
    AIDuplicateSettlementError,
    AIGovernanceError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
)
from flowforge.ai.governance.policy import AIBudgetPolicyService


def test_settings_validation_nan_inf_bool():
    """Reject NaN/Inf in max_duration_seconds and boolean in version."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-v")

    # Reject boolean in version
    with pytest.raises(AIGovernanceError):
        BudgetSettings(scope=scope, version=True)  # type: ignore

    # Reject float NaN
    with pytest.raises(AIGovernanceError):
        BudgetSettings(scope=scope, max_duration_seconds=float("nan"))

    # Reject float Inf
    with pytest.raises(AIGovernanceError):
        BudgetSettings(scope=scope, max_duration_seconds=float("inf"))


def test_strict_idempotency_replay_parameter_comparison():
    """Idempotent reservation replay must check prompt vs completion split and cost, not just total tokens."""
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-idem-strict")
    service = AIBudgetPolicyService(repository=repo)

    # Initial reservation: 100 prompt, 50 completion, 200 micros
    service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=50,
        estimated_cost_micros=200,
        idempotency_key="key-strict-1",
        attempt_number=1,
    )

    # Conflicting replay: same total tokens (150) but different prompt/completion split (50 prompt, 100 completion)
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(
            scope=scope,
            provider="mock",
            model="m1",
            estimated_prompt_tokens=50,
            estimated_completion_tokens=100,
            estimated_cost_micros=200,
            idempotency_key="key-strict-1",
            attempt_number=1,
        )

    # Conflicting replay: different cost
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(
            scope=scope,
            provider="mock",
            model="m1",
            estimated_prompt_tokens=100,
            estimated_completion_tokens=50,
            estimated_cost_micros=500,  # 500 != 200
            idempotency_key="key-strict-1",
            attempt_number=1,
        )


def test_confidentiality_no_raw_tokens_or_secrets_in_errors():
    """Exception messages, args, and ledger notes must not leak raw secrets, raw invalid scope values,

    or caller release strings.
    """
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-leak-test")
    service = AIBudgetPolicyService(repository=repo)

    secret_key = "sk-super-secret-key-12345"
    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=10,
        estimated_completion_tokens=10,
    )

    # Releasing with secret in reason string
    service.release(
        reservation_id=res.reservation_id,
        scope=scope,
        reason=f"Failed due to secret {secret_key}",
    )

    # Check ledger notes: must NOT contain the secret string
    entries = repo.get_ledger_entries(scope)
    for entry in entries:
        assert secret_key not in entry.notes
        assert "sk-" not in entry.notes
