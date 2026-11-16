"""AI Budget and Governance Exception Hierarchy.

Strict fail-closed exceptions. Never exposes secrets, raw prompts, provider responses,
caller identifiers, or sensitive credentials in exception messages, attributes, or details.
Uses fixed generic allowlisted reason codes and safe policy metrics.
"""

from collections.abc import Mapping
from typing import Any

# Allowlisted, fixed reason and action codes
ALLOWED_ACTION_CODES = frozenset({"reserve", "settle", "release", "expire", "unknown_action"})
ALLOWED_REASON_CODES = frozenset(
    {
        "provider_denied_by_default",
        "provider_explicitly_denied",
        "provider_not_authorized",
        "policy_denied",
    }
)


def _safe_action_code(action: str | None) -> str:
    """Returns an allowlisted action code, mapping unknown caller inputs to 'unknown_action'."""
    if not action:
        return "unknown_action"
    cleaned = action.lower().strip()
    return cleaned if cleaned in ALLOWED_ACTION_CODES else "unknown_action"


def _safe_reason_code(reason: str | None) -> str:
    """Returns an allowlisted reason code, mapping unknown caller inputs to 'policy_denied'."""
    if not reason:
        return "policy_denied"
    cleaned = reason.lower().strip()
    return cleaned if cleaned in ALLOWED_REASON_CODES else "policy_denied"


class AIGovernanceError(Exception):
    """Base exception for all AI governance, budget, and audit errors."""

    def __init__(
        self,
        message: str = "AI governance error",
        *,
        code: str = "GOVERNANCE_ERROR",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = dict(details) if details else {}

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


class AIBudgetExhaustedError(AIGovernanceError):
    """Raised when a reservation or recording exceeds a configured budget limit."""

    def __init__(
        self,
        message: str = "AI budget limit exhausted",
        *,
        limit_name: str = "budget_limit",
        requested_value: int | float = 0,
        current_value: int | float = 0,
        limit_value: int | float = 0,
        scope_key: str | None = None,
    ) -> None:
        del scope_key
        details = {
            "limit_name": limit_name,
            "requested_value": requested_value,
            "current_value": current_value,
            "limit_value": limit_value,
        }
        super().__init__(message, code="BUDGET_EXHAUSTED", details=details)
        self.limit_name = limit_name
        self.requested_value = requested_value
        self.current_value = current_value
        self.limit_value = limit_value


class AIReservationNotFoundError(AIGovernanceError):
    """Raised when operating on a reservation ID that does not exist."""

    def __init__(self, reservation_id: str = "") -> None:
        del reservation_id
        super().__init__(
            "Reservation not found",
            code="RESERVATION_NOT_FOUND",
            details={},
        )


class AIDuplicateSettlementError(AIGovernanceError):
    """Raised when attempting duplicate or conflicting reservation / settlement operations."""

    def __init__(
        self, reservation_id: str = "", message: str = "Duplicate or conflicting operation"
    ) -> None:
        del reservation_id
        super().__init__(
            message,
            code="DUPLICATE_OR_CONFLICTING_OPERATION",
            details={},
        )


class AIInvalidReservationStateError(AIGovernanceError):
    """Raised when an operation is invalid for the current reservation status."""

    def __init__(
        self, reservation_id: str = "", current_status: str = "", attempted_action: str = ""
    ) -> None:
        del reservation_id
        safe_action = _safe_action_code(attempted_action)
        safe_status = current_status.lower().strip() if current_status else "unknown"
        super().__init__(
            "Invalid reservation state for requested operation",
            code="INVALID_RESERVATION_STATE",
            details={
                "current_status": safe_status,
                "action_code": safe_action,
            },
        )
        self.current_status = safe_status
        self.action_code = safe_action


class AIProviderPolicyDeniedError(AIGovernanceError):
    """Raised fail-closed when a provider is not explicitly permitted under policy."""

    def __init__(
        self,
        provider: str = "",
        reason: str = "provider_denied_by_default",
    ) -> None:
        del provider
        safe_reason = _safe_reason_code(reason)
        msg = (
            "Provider denied by default (not authorized in budget policy)"
            if safe_reason == "provider_denied_by_default"
            else "Provider access denied by governance policy"
        )
        super().__init__(
            msg,
            code="PROVIDER_POLICY_DENIED",
            details={"reason_code": safe_reason},
        )
        self.reason_code = safe_reason


class AIStaleFencedAttemptError(AIGovernanceError):
    """Raised when an operation presents a stale or mismatched fencing attempt."""

    def __init__(
        self, reservation_id: str = "", expected_attempt: int = 0, actual_attempt: int = 0
    ) -> None:
        del reservation_id
        super().__init__(
            "Mismatched or stale attempt fencing token",
            code="FENCING_ATTEMPT_MISMATCH",
            details={
                "expected_attempt": expected_attempt,
                "actual_attempt": actual_attempt,
            },
        )
        self.expected_attempt = expected_attempt
        self.actual_attempt = actual_attempt
