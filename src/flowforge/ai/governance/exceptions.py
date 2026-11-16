"""AI Budget and Governance Exception Hierarchy.

Strict fail-closed exceptions. Never exposes secrets, raw prompts, provider responses,
caller identifiers, or sensitive credentials in exception messages, attributes, or details.
Uses fixed generic allowlisted reason codes and opaque references.
"""

import hashlib
from collections.abc import Mapping
from typing import Any


def _opaque_ref(val: str | None) -> str:
    """Produces a non-reversible opaque reference hash for debugging without leaking raw IDs or secrets."""
    if not val:
        return ""
    return hashlib.sha256(val.encode("utf-8")).hexdigest()[:12]


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
        details = {
            "limit_name": limit_name,
            "requested_value": requested_value,
            "current_value": current_value,
            "limit_value": limit_value,
        }
        if scope_key:
            details["scope_ref"] = _opaque_ref(scope_key)
        super().__init__(message, code="BUDGET_EXHAUSTED", details=details)
        self.limit_name = limit_name
        self.requested_value = requested_value
        self.current_value = current_value
        self.limit_value = limit_value
        self.scope_ref = _opaque_ref(scope_key) if scope_key else ""


class AIReservationNotFoundError(AIGovernanceError):
    """Raised when operating on a reservation ID that does not exist."""

    def __init__(self, reservation_id: str = "") -> None:
        opaque_id = _opaque_ref(reservation_id)
        super().__init__(
            "Reservation not found",
            code="RESERVATION_NOT_FOUND",
            details={"reservation_ref": opaque_id} if opaque_id else {},
        )
        self.reservation_ref = opaque_id


class AIDuplicateSettlementError(AIGovernanceError):
    """Raised when attempting duplicate or conflicting reservation / settlement operations."""

    def __init__(
        self, reservation_id: str = "", message: str = "Duplicate or conflicting operation"
    ) -> None:
        opaque_id = _opaque_ref(reservation_id)
        super().__init__(
            message,
            code="DUPLICATE_OR_CONFLICTING_OPERATION",
            details={"reservation_ref": opaque_id} if opaque_id else {},
        )
        self.reservation_ref = opaque_id


class AIInvalidReservationStateError(AIGovernanceError):
    """Raised when an operation is invalid for the current reservation status."""

    def __init__(
        self, reservation_id: str = "", current_status: str = "", attempted_action: str = ""
    ) -> None:
        opaque_id = _opaque_ref(reservation_id)
        super().__init__(
            f"Invalid reservation state for {attempted_action or 'operation'}",
            code="INVALID_RESERVATION_STATE",
            details={
                "reservation_ref": opaque_id,
                "current_status": current_status,
                "attempted_action": attempted_action,
            },
        )
        self.reservation_ref = opaque_id
        self.current_status = current_status
        self.attempted_action = attempted_action


class AIProviderPolicyDeniedError(AIGovernanceError):
    """Raised fail-closed when a provider is not explicitly permitted under policy."""

    def __init__(
        self,
        provider: str = "",
        reason: str = "denied by default (not authorized in budget policy)",
    ) -> None:
        opaque_p = _opaque_ref(provider)
        super().__init__(
            f"Provider access denied: {reason}",
            code="PROVIDER_POLICY_DENIED",
            details={"provider_ref": opaque_p, "reason": reason},
        )
        self.provider_ref = opaque_p
        self.reason = reason


class AIStaleFencedAttemptError(AIGovernanceError):
    """Raised when an operation presents a stale or mismatched fencing attempt."""

    def __init__(
        self, reservation_id: str = "", expected_attempt: int = 0, actual_attempt: int = 0
    ) -> None:
        opaque_id = _opaque_ref(reservation_id)
        super().__init__(
            "Mismatched or stale attempt fencing token",
            code="FENCING_ATTEMPT_MISMATCH",
            details={
                "reservation_ref": opaque_id,
                "expected_attempt": expected_attempt,
                "actual_attempt": actual_attempt,
            },
        )
        self.reservation_ref = opaque_id
        self.expected_attempt = expected_attempt
        self.actual_attempt = actual_attempt
