"""AI Budget and Governance Exception Hierarchy.

Strict fail-closed exceptions. Never exposes secrets, raw prompts, provider responses,
caller identifiers, or sensitive credentials in exception messages, attributes, or details.
Uses fixed generic allowlisted codes, safe policy metrics, and strictly sanitized constructors.
"""

from collections.abc import Mapping
from typing import Any

# Allowlisted, fixed codes and statuses
ALLOWED_ACTION_CODES = frozenset({"reserve", "settle", "release", "expire", "unknown_action"})
ALLOWED_STATUS_CODES = frozenset({"pending", "settled", "released", "expired", "unknown_status"})
ALLOWED_REASON_CODES = frozenset(
    {
        "provider_denied_by_default",
        "provider_explicitly_denied",
        "provider_not_authorized",
        "policy_denied",
    }
)
ALLOWED_ERROR_CODES = frozenset(
    {
        "GOVERNANCE_ERROR",
        "BUDGET_EXHAUSTED",
        "RESERVATION_NOT_FOUND",
        "DUPLICATE_OR_CONFLICTING_OPERATION",
        "INVALID_RESERVATION_STATE",
        "PROVIDER_POLICY_DENIED",
        "FENCING_ATTEMPT_MISMATCH",
    }
)
ALLOWED_MESSAGES = frozenset(
    {
        "AI governance error",
        "AI budget limit exhausted",
        "Reservation not found",
        "Duplicate or conflicting operation",
        "Duplicate or conflicting settlement",
        "Invalid reservation state for requested operation",
        "Provider access denied by governance policy",
        "Provider denied by default (not authorized in budget policy)",
        "Mismatched or stale attempt fencing token",
        "Settlement resulted in budget overrun",
        "Settlement resulted in budget overrun (tokens_overrun)",
        "Settlement resulted in budget overrun (cost_overrun)",
        "Settlement resulted in budget overrun (time_overrun)",
        "Budget limit for max calls exceeded",
        "Budget limit for total tokens exceeded by estimated request",
        "Budget limit for estimated cost exceeded",
        "version must be a positive integer, not bool",
        "max_duration_seconds must be a non-negative finite float or None",
        "allowed_providers must be a tuple or list of strings, not a bare string",
        "denied_providers must be a tuple or list of strings, not a bare string",
        "actual_prompt_tokens must be non-negative integer",
        "actual_completion_tokens must be non-negative integer",
        "estimated_prompt_tokens must be a non-negative integer",
        "estimated_completion_tokens must be a non-negative integer",
        "estimated_cost_micros must be a non-negative integer",
        "EXACT settlement requires actual_prompt_tokens and actual_completion_tokens",
    }
)


def _safe_action_code(action: str | None) -> str:
    """Returns an allowlisted action code, mapping unknown caller inputs to 'unknown_action'."""
    if not action:
        return "unknown_action"
    cleaned = action.lower().strip()
    return cleaned if cleaned in ALLOWED_ACTION_CODES else "unknown_action"


def _safe_status_code(status: str | None) -> str:
    """Returns an allowlisted status code, mapping unknown caller inputs to 'unknown_status'."""
    if not status:
        return "unknown_status"
    cleaned = status.lower().strip()
    return cleaned if cleaned in ALLOWED_STATUS_CODES else "unknown_status"


def _safe_reason_code(reason: str | None) -> str:
    """Returns an allowlisted reason code, mapping unknown caller inputs to 'policy_denied'."""
    if not reason:
        return "policy_denied"
    cleaned = reason.lower().strip()
    return cleaned if cleaned in ALLOWED_REASON_CODES else "policy_denied"


def _safe_error_code(code: str | None, default: str = "GOVERNANCE_ERROR") -> str:
    """Returns an allowlisted error code, mapping unknown caller inputs to default."""
    if not code:
        return default
    return code if code in ALLOWED_ERROR_CODES else default


def _safe_message(msg: str | None, default: str = "AI governance error") -> str:
    """Returns an allowlisted message, preventing caller strings from leaking through exception constructors."""
    if not msg:
        return default
    # If exactly in allowed messages, permit it
    if msg in ALLOWED_MESSAGES:
        return msg
    return default


def _sanitize_exception_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """Sanitizes exception details to only permit allowlisted numeric metrics and safe codes."""
    if not details:
        return {}
    clean: dict[str, Any] = {}
    allowed_numeric_keys = frozenset(
        {
            "requested_value",
            "current_value",
            "limit_value",
            "expected_attempt",
            "actual_attempt",
        }
    )
    for k, v in details.items():
        if k in allowed_numeric_keys and isinstance(v, (int, float)) and not isinstance(v, bool):
            clean[k] = v
        elif k == "limit_name" and str(v) in (
            "max_calls",
            "max_total_tokens",
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_estimated_cost_micros",
            "budget_limit",
        ):
            clean[k] = str(v)
        elif k == "action_code":
            clean[k] = _safe_action_code(str(v))
        elif k == "current_status":
            clean[k] = _safe_status_code(str(v))
        elif k == "reason_code":
            clean[k] = _safe_reason_code(str(v))
    return clean


class AIGovernanceError(Exception):
    """Base exception for all AI governance, budget, and audit errors."""

    def __init__(
        self,
        message: str = "AI governance error",
        *,
        code: str = "GOVERNANCE_ERROR",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        safe_msg = _safe_message(message, default="AI governance error")
        safe_code = _safe_error_code(code, default="GOVERNANCE_ERROR")
        safe_details = _sanitize_exception_details(details)
        super().__init__(safe_msg)
        self.message = safe_msg
        self.code = safe_code
        self.details = safe_details

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
        safe_msg = _safe_message(message, default="AI budget limit exhausted")
        safe_limit_name = (
            str(limit_name)
            if str(limit_name)
            in (
                "max_calls",
                "max_total_tokens",
                "max_prompt_tokens",
                "max_completion_tokens",
                "max_estimated_cost_micros",
                "budget_limit",
            )
            else "budget_limit"
        )
        details = {
            "limit_name": safe_limit_name,
            "requested_value": requested_value
            if isinstance(requested_value, (int, float)) and not isinstance(requested_value, bool)
            else 0,
            "current_value": current_value
            if isinstance(current_value, (int, float)) and not isinstance(current_value, bool)
            else 0,
            "limit_value": limit_value
            if isinstance(limit_value, (int, float)) and not isinstance(limit_value, bool)
            else 0,
        }
        super().__init__(safe_msg, code="BUDGET_EXHAUSTED", details=details)
        self.limit_name = safe_limit_name
        self.requested_value = details["requested_value"]
        self.current_value = details["current_value"]
        self.limit_value = details["limit_value"]


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
        safe_msg = _safe_message(message, default="Duplicate or conflicting operation")
        super().__init__(
            safe_msg,
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
        safe_status = _safe_status_code(current_status)
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
        details = {
            "expected_attempt": expected_attempt
            if isinstance(expected_attempt, int) and not isinstance(expected_attempt, bool)
            else 0,
            "actual_attempt": actual_attempt
            if isinstance(actual_attempt, int) and not isinstance(actual_attempt, bool)
            else 0,
        }
        super().__init__(
            "Mismatched or stale attempt fencing token",
            code="FENCING_ATTEMPT_MISMATCH",
            details=details,
        )
        self.expected_attempt = details["expected_attempt"]
        self.actual_attempt = details["actual_attempt"]
