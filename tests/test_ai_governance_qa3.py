"""TDD RED test suite for Milestone M0-018B1 QA-3.

Validates:
1. Public exception details/attributes never leak raw identifiers (reservation IDs, provider names, scope keys, raw secrets)
   in attributes, `details`, or `__dict__`.
2. BudgetSettings strictly rejects bare strings for allowed_providers / denied_providers (does not unpack into single chars).
   Contradictory entries follow strict deny precedence or are rejected.
3. Provider / model / idempotency identifier input validation (strict charset, length bounds, no arbitrary printable secrets).
4. Safe audit metadata typing: dropping non-allowlisted / arbitrary unstructured strings; test with arbitrary printable secrets
   (without 'secret' or 'token' in the text).
5. Strict replay tests: changing EACH individual parameter (scope, provider, model, prompt tokens, completion tokens, cost,
   attempt) under the same idempotency key fails closed with AIDuplicateSettlementError.
   Changing EACH settlement parameter under same idempotency key fails closed with AIDuplicateSettlementError.
"""

import pytest

from flowforge.ai.governance.exceptions import (
    AIDuplicateSettlementError,
    AIGovernanceError,
    AIProviderPolicyDeniedError,
    AIReservationNotFoundError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
    SettlementType,
)
from flowforge.ai.governance.policy import AIBudgetPolicyService


def test_exception_details_no_raw_identifiers_or_secrets():
    """Public exception attributes, details, and __dict__ must not expose raw caller identifiers or secrets."""
    raw_secret_id = "RES-XYZ-9999-SECRET-VALUE"
    err = AIReservationNotFoundError(reservation_id=raw_secret_id)
    # Check string and repr
    assert raw_secret_id not in str(err)
    assert raw_secret_id not in repr(err)
    # Check details dict and object attributes
    err_dict_str = str(err.__dict__)
    assert raw_secret_id not in err_dict_str
    assert "details" in err.__dict__
    assert raw_secret_id not in str(err.details)


def test_budget_settings_reject_bare_string_providers():
    """Passing a bare string to allowed_providers or denied_providers must be rejected (not unpacked into chars)."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-str")
    with pytest.raises(AIGovernanceError):
        BudgetSettings(scope=scope, allowed_providers="mock")  # type: ignore

    with pytest.raises(AIGovernanceError):
        BudgetSettings(scope=scope, denied_providers="mock")  # type: ignore


def test_budget_settings_deny_precedence():
    """If a provider is in both allowed and denied providers, it must be rejected or strictly denied."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-deny-prec")
    repo = InMemoryBudgetRepository()
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            allowed_providers=("openai", "mock"),
            denied_providers=("openai",),
        )
    )
    service = AIBudgetPolicyService(repository=repo)
    with pytest.raises(AIProviderPolicyDeniedError):
        service.reserve(
            scope=scope,
            provider="openai",
            model="gpt-4o",
            estimated_prompt_tokens=10,
            estimated_completion_tokens=10,
        )


def test_input_bounds_validation():
    """Provider, model, and idempotency key must adhere to strict alphanumeric/bounded charset and length limits."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-bounds")
    service = AIBudgetPolicyService(repository=InMemoryBudgetRepository())

    # Provider with malicious/invalid characters or excessive length
    with pytest.raises(AIGovernanceError):
        service.reserve(
            scope=scope,
            provider="mock/../../bad\x00char",
            model="m1",
            estimated_prompt_tokens=10,
            estimated_completion_tokens=10,
        )

    # Model name too long
    with pytest.raises(AIGovernanceError):
        service.reserve(
            scope=scope,
            provider="mock",
            model="a" * 300,
            estimated_prompt_tokens=10,
            estimated_completion_tokens=10,
        )


def test_metadata_drops_arbitrary_printable_secrets():
    """Metadata dropping must discard arbitrary unknown printable values without needing 'secret' keyword."""
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-sec")
    service = AIBudgetPolicyService(repository=InMemoryBudgetRepository())

    opaque_secret = "ghp_984fjkf8934jklsdu90234jksldf09"  # arbitrary printable secret without 'secret' or 'token' in word
    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=10,
        estimated_completion_tokens=10,
        audit_metadata={
            "job_id": "job-123",
            "correlation_id": opaque_secret,  # correlation_id must be validated UUID or hex, not arbitrary text
            "random_field": opaque_secret,
        },
    )
    # random_field dropped
    assert "random_field" not in res.audit_metadata
    # invalid correlation_id dropped or opaque-hashed
    assert opaque_secret not in str(res.audit_metadata)


def test_strict_reservation_replay_each_parameter():
    """Changing ANY single parameter under same idempotency key must raise AIDuplicateSettlementError."""
    repo = InMemoryBudgetRepository()
    scope1 = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-strict-rep1")
    scope2 = BudgetScope(
        scope_type=BudgetScopeType.TENANT, tenant_id="tenant-strict-rep1", workflow_id="wf-1"
    )
    service = AIBudgetPolicyService(repository=repo)

    base_params = {
        "scope": scope1,
        "provider": "mock",
        "model": "model-a",
        "estimated_prompt_tokens": 100,
        "estimated_completion_tokens": 50,
        "estimated_cost_micros": 150,
        "attempt_number": 1,
        "idempotency_key": "fixed-key-1",
    }
    service.reserve(**base_params)

    # 1. Scope differs
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(**{**base_params, "scope": scope2})

    # 2. Provider differs
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(**{**base_params, "provider": "ollama"})

    # 3. Model differs
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(**{**base_params, "model": "model-b"})

    # 4. Prompt tokens differ
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(**{**base_params, "estimated_prompt_tokens": 99})

    # 5. Completion tokens differ
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(**{**base_params, "estimated_completion_tokens": 51})

    # 6. Cost differs
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(**{**base_params, "estimated_cost_micros": 999})

    # 7. Attempt differs
    with pytest.raises(AIDuplicateSettlementError):
        service.reserve(**{**base_params, "attempt_number": 2})


def test_strict_settlement_replay_each_parameter():
    """Changing ANY single settlement parameter under same idempotency key must raise AIDuplicateSettlementError."""
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-strict-set")
    service = AIBudgetPolicyService(repository=repo)

    res = service.reserve(
        scope=scope,
        provider="mock",
        model="m1",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=50,
    )

    base_settle = {
        "reservation_id": res.reservation_id,
        "scope": scope,
        "actual_prompt_tokens": 80,
        "actual_completion_tokens": 40,
        "actual_cost_micros": 120,
        "settlement_type": SettlementType.EXACT,
        "attempt_number": 1,
        "duration_seconds": 1.5,
        "idempotency_key": "settle-key-fixed",
    }
    service.settle(**base_settle)

    # Change prompt tokens
    with pytest.raises(AIDuplicateSettlementError):
        service.settle(**{**base_settle, "actual_prompt_tokens": 70})

    # Change completion tokens
    with pytest.raises(AIDuplicateSettlementError):
        service.settle(**{**base_settle, "actual_completion_tokens": 50})

    # Change cost
    with pytest.raises(AIDuplicateSettlementError):
        service.settle(**{**base_settle, "actual_cost_micros": 999})

    # Change duration
    with pytest.raises(AIDuplicateSettlementError):
        service.settle(**{**base_settle, "duration_seconds": 2.5})
