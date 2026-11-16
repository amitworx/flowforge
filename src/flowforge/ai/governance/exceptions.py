"""AI Budget and Governance Exception Hierarchy.

Strict fail-closed exceptions. Never exposes secrets, raw prompts, provider responses,
or sensitive API credentials in exception messages or representations.
Uses fixed generic allowlisted reason codes.
"""

from collections.abc import Mapping
from typing import Any


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
            details["scope_key"] = scope_key
        super().__init__(message, code="BUDGET_EXHAUSTED", details=details)
        self.limit_name = limit_name
        self.requested_value = requested_value
        self.current_value = current_value
        self.limit_value = limit_value
        self.scope_key = scope_key


class AIReservationNotFoundError(AIGovernanceError):
    """Raised when operating on a reservation ID that does not exist."""

    def __init__(self, reservation_id: str = "") -> None:
        super().__init__(
            "Reservation not found",
            code="RESERVATION_NOT_FOUND",
            details={"reservation_id": reservation_id} if reservation_id else {},
        )
        self.reservation_id = reservation_id


class AIDuplicateSettlementError(AIGovernanceError):
    """Raised when attempting duplicate or conflicting reservation / settlement operations."""

    def __init__(
        self, reservation_id: str = "", message: str = "Duplicate or conflicting settlement"
    ) -> None:
        super().__init__(
            message,
            code="DUPLICATE_OR_CONFLICTING_OPERATION",
            details={"reservation_id": reservation_id} if reservation_id else {},
        )
        self.reservation_id = reservation_id


class AIInvalidReservationStateError(AIGovernanceError):
    """Raised when an operation is invalid for the current reservation status."""

    def __init__(
        self, reservation_id: str = "", current_status: str = "", attempted_action: str = ""
    ) -> None:
        super().__init__(
            f"Invalid reservation state for {attempted_action or 'operation'}",
            code="INVALID_RESERVATION_STATE",
            details={
                "reservation_id": reservation_id,
                "current_status": current_status,
                "attempted_action": attempted_action,
            },
        )
        self.reservation_id = reservation_id
        self.current_status = current_status
        self.attempted_action = attempted_action


class AIProviderPolicyDeniedError(AIGovernanceError):
    """Raised fail-closed when a provider is not explicitly permitted under policy."""

    def __init__(
        self,
        provider: str = "",
        reason: str = "denied by default (not authorized in budget policy)",
    ) -> None:
        super().__init__(
            f"Provider access denied: {reason}",
            code="PROVIDER_POLICY_DENIED",
            details={"provider": provider, "reason": reason},
        )
        self.provider = provider
        self.reason = reason


class AIStaleFencedAttemptError(AIGovernanceError):
    """Raised when an operation presents a stale or mismatched fencing attempt."""

    def __init__(
        self, reservation_id: str = "", expected_attempt: int = 0, actual_attempt: int = 0
    ) -> None:
        super().__init__(
            "Mismatched or stale attempt fencing token",
            code="FENCING_ATTEMPT_MISMATCH",
            details={
                "reservation_id": reservation_id,
                "expected_attempt": expected_attempt,
                "actual_attempt": actual_attempt,
            },
        )
        self.reservation_id = reservation_id
        self.expected_attempt = expected_attempt
        self.actual_attempt = actual_attempt
