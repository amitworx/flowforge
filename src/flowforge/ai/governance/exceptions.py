"""AI Budget and Governance Exception Hierarchy.

Strict fail-closed exceptions. Never exposes secrets, raw prompts, provider responses,
or sensitive API credentials in exception messages or representations.
"""

from collections.abc import Mapping
from typing import Any


class AIGovernanceError(Exception):
    """Base exception for all AI governance, budget, and audit errors."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details) if details else {}

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


class AIBudgetExhaustedError(AIGovernanceError):
    """Raised when a reservation or recording exceeds a configured budget limit."""

    def __init__(
        self,
        message: str,
        *,
        limit_name: str,
        requested_value: int | float,
        current_value: int | float,
        limit_value: int | float,
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
        super().__init__(message, details=details)
        self.limit_name = limit_name
        self.requested_value = requested_value
        self.current_value = current_value
        self.limit_value = limit_value
        self.scope_key = scope_key


class AIReservationNotFoundError(AIGovernanceError):
    """Raised when operating on a reservation ID that does not exist."""

    def __init__(self, reservation_id: str) -> None:
        super().__init__(
            f"Reservation '{reservation_id}' not found",
            details={"reservation_id": reservation_id},
        )
        self.reservation_id = reservation_id


class AIDuplicateSettlementError(AIGovernanceError):
    """Raised when attempting to settle a reservation that has already been settled

    with conflicting details, or re-settling an already finalized reservation.
    """

    def __init__(self, reservation_id: str, message: str | None = None) -> None:
        msg = message or f"Reservation '{reservation_id}' has already been settled"
        super().__init__(msg, details={"reservation_id": reservation_id})
        self.reservation_id = reservation_id


class AIInvalidReservationStateError(AIGovernanceError):
    """Raised when an operation is invalid for the current reservation status."""

    def __init__(self, reservation_id: str, current_status: str, attempted_action: str) -> None:
        super().__init__(
            f"Cannot {attempted_action} reservation '{reservation_id}' in status '{current_status}'",
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
    """Raised fail-closed when a provider is not explicitly permitted under hosted policy."""

    def __init__(
        self, provider: str, reason: str = "Provider not allowed by governance policy"
    ) -> None:
        super().__init__(
            f"Provider '{provider}' denied: {reason}",
            details={"provider": provider, "reason": reason},
        )
        self.provider = provider
        self.reason = reason


class AIStaleFencedAttemptError(AIGovernanceError):
    """Raised when an operation presents a stale fencing token or attempt number."""

    def __init__(self, reservation_id: str, expected_attempt: int, actual_attempt: int) -> None:
        super().__init__(
            f"Stale attempt {actual_attempt} for reservation '{reservation_id}' (expected {expected_attempt})",
            details={
                "reservation_id": reservation_id,
                "expected_attempt": expected_attempt,
                "actual_attempt": actual_attempt,
            },
        )
        self.reservation_id = reservation_id
        self.expected_attempt = expected_attempt
        self.actual_attempt = actual_attempt
