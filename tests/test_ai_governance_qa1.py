"""TDD Red test suite for Milestone M0-018B1 QA-1.

Validates the confirmed issues to be fixed:
1. Check-then-write race in reserve: atomic reserve under one lock in repository.
   Simultaneous 1-token requests when only 1 token remains: exactly 1 succeeds, other fails.
2. Cross-scope settings & hierarchy: enforcing all ancestor caps (e.g. tenant cap limits workflow scope,
   global limits tenant), deny-by-default at all levels, cross-tenant isolation with no bypass via child scope.
3. Idempotency & tenant confusion: scoped idempotency keys, reject cross-tenant or mismatched key reuse,
   reject stale (<) AND future mismatched (>) fencing attempts unless properly sequence-matched.
4. Confidentiality & audit idempotency: allowlisted bounded metadata fields, generic error messages,
   atomic state transition + ledger entry with no duplicate ledger entries on replay,
   strict non-negative finite bounds validation on settlement tokens/cost/duration.
"""

import concurrent.futures

import pytest

from flowforge.ai.governance.exceptions import (
    AIBudgetExhaustedError,
    AIDuplicateSettlementError,
    AIStaleFencedAttemptError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
)
from flowforge.ai.governance.policy import AIBudgetPolicyService


def test_atomic_reserve_single_token_race():
    """When exactly 1 token remains, two concurrent reserve requests for 1 token

    must result in exactly 1 success and 1 fail-closed AIBudgetExhaustedError.
    """
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-race-1")
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            max_calls=10,
            max_total_tokens=1,  # Only 1 token allowed!
            allowed_providers=("mock",),
        )
    )
    service = AIBudgetPolicyService(repository=repo)

    results = []
    errors = []

    def try_reserve(thread_idx: int):
        try:
            service.reserve(
                scope=scope,
                provider="mock",
                model="m1",
                estimated_prompt_tokens=1,
                estimated_completion_tokens=0,
                idempotency_key=f"race-{thread_idx}",
            )
            return True
        except AIBudgetExhaustedError as exc:
            errors.append(exc)
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(try_reserve, 1)
        f2 = executor.submit(try_reserve, 2)
        results = [f1.result(), f2.result()]

    assert sum(results) == 1
    assert len(errors) == 1


def test_ancestor_hierarchy_caps_enforced():
    """A workflow scope under a tenant cannot exceed the tenant's max_total_tokens

    even if the workflow budget settings specify a higher limit or are unset.
    """
    repo = InMemoryBudgetRepository()
    tenant_scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-h1")
    workflow_scope = BudgetScope(
        scope_type=BudgetScopeType.WORKFLOW,
        tenant_id="tenant-h1",
        workflow_id="wf-1",
    )

    # Tenant allows at most 500 tokens
    repo.set_budget_settings(
        BudgetSettings(
            scope=tenant_scope,
            max_calls=10,
            max_total_tokens=500,
            allowed_providers=("mock",),
        )
    )
    # Workflow specifies 1000 tokens
    repo.set_budget_settings(
        BudgetSettings(
            scope=workflow_scope,
            max_calls=10,
            max_total_tokens=1000,
            allowed_providers=("mock",),
        )
    )
    service = AIBudgetPolicyService(repository=repo)

    # Reserving 600 tokens on the workflow scope MUST be rejected because tenant cap is 500
    with pytest.raises(AIBudgetExhaustedError) as exc_info:
        service.reserve(
            scope=workflow_scope,
            provider="mock",
            model="m1",
            estimated_prompt_tokens=300,
            estimated_completion_tokens=300,  # 600 > 500
        )
    assert exc_info.value.limit_name == "max_total_tokens"


def test_cross_tenant_idempotency_key_isolation():
    """An idempotency key used by Tenant A cannot be reused by or returned to Tenant B."""
    repo = InMemoryBudgetRepository()
    scope_a = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-A")
    scope_b = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-B")

    repo.set_budget_settings(
        BudgetSettings(
            scope=scope_a, max_calls=10, max_total_tokens=1000, allowed_providers=("mock",)
        )
    )
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope_b, max_calls=10, max_total_tokens=1000, allowed_providers=("mock",)
        )
    )

    service = AIBudgetPolicyService(repository=repo)

    res_a = service.reserve(
        scope=scope_a,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=50,
        estimated_completion_tokens=50,
        idempotency_key="shared-key-1",
    )
    assert res_a.scope.tenant_id == "tenant-A"

    # Attempting to reserve with same idempotency key under Tenant B must fail with error,
    # and never return Tenant A's reservation!
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(
            scope=scope_b,
            provider="mock",
            model="m1",
            estimated_prompt_tokens=50,
            estimated_completion_tokens=50,
            idempotency_key="shared-key-1",
        )


def test_fencing_attempt_validation():
    """Settlement must reject both stale (less than expected) and mismatched future attempts."""
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-fence")
    service = AIBudgetPolicyService(repository=repo)

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=50,
        estimated_completion_tokens=50,
        attempt_number=2,
    )

    # Lower attempt (stale)
    with pytest.raises(AIStaleFencedAttemptError):
        service.settle(
            reservation_id=res.reservation_id,
            scope=scope,
            actual_prompt_tokens=50,
            actual_completion_tokens=50,
            attempt_number=1,
        )

    # Mismatched unexpected higher attempt
    with pytest.raises(AIStaleFencedAttemptError):
        service.settle(
            reservation_id=res.reservation_id,
            scope=scope,
            actual_prompt_tokens=50,
            actual_completion_tokens=50,
            attempt_number=5,
        )


def test_metadata_allowlist_filtering():
    """Audit metadata must sanitize and filter out non-allowlisted or sensitive fields."""
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-priv-qa1")
    service = AIBudgetPolicyService(repository=repo)

    dirty_metadata = {
        "job_id": "job-123",
        "step_name": "ai_inference",
        "authorization": "Bearer secret-token-xyz",
        "prompt": "SELECT * FROM users",
        "api_key": "sk-1234567890",
    }

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=50,
        estimated_completion_tokens=50,
        audit_metadata=dirty_metadata,
    )

    # Disallowed keys must be stripped
    assert "job_id" in res.audit_metadata
    assert "step_name" in res.audit_metadata
    assert "authorization" not in res.audit_metadata
    assert "prompt" not in res.audit_metadata
    assert "api_key" not in res.audit_metadata


def test_idempotent_settlement_does_not_duplicate_ledger():
    """Settling idempotently multiple times must NOT append duplicate ledger entries."""
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-ledger-idem")
    service = AIBudgetPolicyService(repository=repo)

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=50,
        estimated_completion_tokens=50,
    )

    ledger_before = len(repo.get_ledger_entries(scope))

    s1 = service.settle(
        reservation_id=res.reservation_id,
        scope=scope,
        actual_prompt_tokens=40,
        actual_completion_tokens=40,
        idempotency_key="settle-idem-1",
    )
    ledger_after_1 = len(repo.get_ledger_entries(scope))
    assert ledger_after_1 == ledger_before + 1

    # Replay identical settlement
    s2 = service.settle(
        reservation_id=res.reservation_id,
        scope=scope,
        actual_prompt_tokens=40,
        actual_completion_tokens=40,
        idempotency_key="settle-idem-1",
    )
    assert s1.settlement_id == s2.settlement_id
    ledger_after_2 = len(repo.get_ledger_entries(scope))
    # Must NOT have appended a duplicate ledger entry!
    assert ledger_after_2 == ledger_after_1
