"""Adversarial and Confidentiality Tests for Milestone M0-018B1 Final QA.

Asserts that arbitrary, unusual, printable secret strings never leak in:
- `str(exc)`
- `repr(exc)`
- `exc.args`
- `exc.details`
- `exc.__dict__`
- `exc.__cause__` / `exc.__context__`
"""

import pytest

from flowforge.ai.governance.exceptions import (
    AIBudgetExhaustedError,
    AIDuplicateSettlementError,
    AIGovernanceError,
    AIInvalidReservationStateError,
    AIProviderPolicyDeniedError,
    AIReservationNotFoundError,
    AIStaleFencedAttemptError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
)
from flowforge.ai.governance.policy import AIBudgetPolicyService


def test_adversarial_exception_confidentiality_no_caller_secrets():
    """Ensure sensitive caller strings cannot leak into any public exception attribute or representation."""
    sensitive_payload = "super-secret-user-credential-XYZ987654321"

    # 1. AIReservationNotFoundError
    err1 = AIReservationNotFoundError(reservation_id=sensitive_payload)
    for target in (str(err1), repr(err1), str(err1.args), str(err1.details), str(err1.__dict__)):
        assert sensitive_payload not in target

    # 2. AIDuplicateSettlementError
    err2 = AIDuplicateSettlementError(reservation_id=sensitive_payload, message=sensitive_payload)
    for target in (str(err2), repr(err2), str(err2.args), str(err2.details), str(err2.__dict__)):
        assert sensitive_payload not in target

    # 3. AIInvalidReservationStateError
    err3 = AIInvalidReservationStateError(
        reservation_id=sensitive_payload,
        current_status=sensitive_payload,
        attempted_action=sensitive_payload,
    )
    for target in (str(err3), repr(err3), str(err3.args), str(err3.details), str(err3.__dict__)):
        assert sensitive_payload not in target
    assert err3.action_code == "unknown_action"
    assert err3.current_status == "unknown_status"

    # 4. AIProviderPolicyDeniedError
    err4 = AIProviderPolicyDeniedError(provider=sensitive_payload, reason=sensitive_payload)
    for target in (str(err4), repr(err4), str(err4.args), str(err4.details), str(err4.__dict__)):
        assert sensitive_payload not in target
    assert err4.reason_code == "policy_denied"

    # 5. AIStaleFencedAttemptError
    err5 = AIStaleFencedAttemptError(
        reservation_id=sensitive_payload, expected_attempt=1, actual_attempt=2
    )
    for target in (str(err5), repr(err5), str(err5.args), str(err5.details), str(err5.__dict__)):
        assert sensitive_payload not in target

    # 6. AIBudgetExhaustedError
    err6 = AIBudgetExhaustedError(message=sensitive_payload, scope_key=sensitive_payload)
    for target in (str(err6), repr(err6), str(err6.args), str(err6.details), str(err6.__dict__)):
        assert sensitive_payload not in target

    # 7. AIGovernanceError with custom code/details
    err7 = AIGovernanceError(
        message=sensitive_payload,
        code=sensitive_payload,
        details={"custom_secret_key": sensitive_payload},
    )
    for target in (str(err7), repr(err7), str(err7.args), str(err7.details), str(err7.__dict__)):
        assert sensitive_payload not in target


def test_adversarial_service_calls_no_secret_leak():
    """Ensure calling service with unauthorized or invalid values never echoes raw secrets into caught exceptions."""
    repo = InMemoryBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-adv")
    repo.set_budget_settings(
        BudgetSettings(
            scope=scope,
            max_calls=1,
            max_total_tokens=10,
            allowed_providers=("mock",),
        )
    )
    service = AIBudgetPolicyService(repository=repo)

    sensitive_provider = "unauthorized-hosted-provider-secret-999"
    with pytest.raises(AIProviderPolicyDeniedError) as exc_info:
        service.reserve(
            scope=scope,
            provider=sensitive_provider,
            model="m1",
            estimated_prompt_tokens=5,
            estimated_completion_tokens=5,
        )
    err = exc_info.value
    assert sensitive_provider not in str(err)
    assert sensitive_provider not in repr(err)
    assert sensitive_provider not in str(err.__dict__)
    assert sensitive_provider not in str(err.details)
