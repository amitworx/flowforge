"""Abstract Protocol defining the persistence contract for AI Budget and Governance repositories.

Any future database (PostgreSQL/Django, Redis) or in-memory implementation MUST
satisfy this Protocol.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetSettings,
    OverrunMarker,
    ReservationRecord,
    SettlementRecord,
    SettlementType,
    UsageLedgerEntry,
)


@runtime_checkable
class BudgetRepository(Protocol):
    """Persistence-neutral protocol for transactional, isolated budget state operations."""

    def get_budget_settings(self, scope: BudgetScope) -> BudgetSettings | None:
        """Fetch budget settings for a specific scope, or None if none configured."""
        ...

    def set_budget_settings(self, settings: BudgetSettings) -> None:
        """Upsert budget settings for a scope."""
        ...

    def get_aggregate_usage(
        self, scope: BudgetScope
    ) -> tuple[int, int, int]:  # (total_calls, settled_tokens, settled_cost_micros)
        """Return aggregated settled usage for the given scope."""
        ...

    def get_active_reserved_usage(
        self, scope: BudgetScope
    ) -> tuple[int, int, int]:  # (reserved_calls, reserved_tokens, reserved_cost_micros)
        """Return currently active reserved exposure (pending reservations) for the given scope."""
        ...

    def get_reservation_by_id(self, reservation_id: str) -> ReservationRecord | None:
        """Retrieve reservation record by its unique reservation_id."""
        ...

    def get_reservation_by_idempotency_key(
        self, idempotency_key: str, scope: BudgetScope | None = None
    ) -> ReservationRecord | None:
        """Retrieve reservation record by its client-supplied idempotency key, optionally scoped."""
        ...

    def reserve_atomic(
        self,
        *,
        reservation_id: str,
        idempotency_key: str,
        scope: BudgetScope,
        ancestor_scopes: list[BudgetScope],
        provider: str,
        model: str,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
        estimated_total_tokens: int,
        estimated_cost_micros: int,
        attempt_number: int,
        created_at: datetime.datetime,
        expires_at: datetime.datetime | None = None,
        audit_metadata: Mapping[str, Any] | None = None,
        ledger_entry: UsageLedgerEntry | None = None,
    ) -> tuple[ReservationRecord, bool]:  # (record, is_replayed)
        """Atomically checks all budget limits across scope and ancestors, creates reservation,

        and optionally records ledger entry under a single lock/transaction.
        If idempotency_key matches an existing reservation in this scope with identical parameters,
        returns (existing_record, True). If parameters conflict or cross-tenant key reuse is attempted,
        raises AIDuplicateSettlementError.
        If limits exceeded, raises AIBudgetExhaustedError fail-closed without mutating counters.
        """
        ...

    def create_reservation(
        self,
        *,
        reservation_id: str,
        idempotency_key: str,
        scope: BudgetScope,
        provider: str,
        model: str,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
        estimated_total_tokens: int,
        estimated_cost_micros: int,
        attempt_number: int,
        created_at: datetime.datetime,
        expires_at: datetime.datetime | None = None,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> ReservationRecord:
        """Create a reservation record directly."""
        ...

    def get_settlement_by_id(self, settlement_id: str) -> SettlementRecord | None:
        """Retrieve settlement record by settlement_id."""
        ...

    def get_settlement_by_idempotency_key(
        self, idempotency_key: str, scope: BudgetScope | None = None
    ) -> SettlementRecord | None:
        """Retrieve settlement record by idempotency_key."""
        ...

    def get_settlement_by_reservation_id(self, reservation_id: str) -> SettlementRecord | None:
        """Retrieve settlement record for a reservation."""
        ...

    def settle_reservation(
        self,
        *,
        settlement_id: str,
        reservation_id: str,
        idempotency_key: str,
        settlement_type: SettlementType,
        actual_prompt_tokens: int,
        actual_completion_tokens: int,
        actual_total_tokens: int,
        actual_cost_micros: int,
        attempt_number: int,
        settled_at: datetime.datetime,
        overrun_marker: OverrunMarker = OverrunMarker.NONE,
        overrun_tokens: int = 0,
        overrun_cost_micros: int = 0,
        duration_seconds: float = 0.0,
        audit_metadata: Mapping[str, Any] | None = None,
        ledger_entry: UsageLedgerEntry | None = None,
    ) -> tuple[SettlementRecord, bool]:  # (record, is_replayed)
        """Atomically settle a pending reservation and optionally append ledger entry.

        Transitions reservation from PENDING -> SETTLED.
        Deducts reserved allowance and increments settled usage.
        Guarantees exactly-once settlement:
        - If already settled with same idempotency_key and parameters, returns (existing, True).
        - If already settled with conflicting parameters, raises AIDuplicateSettlementError.
        - If attempt_number does not match reservation's attempt_number, raises AIStaleFencedAttemptError.
        """
        ...

    def release_reservation(
        self,
        *,
        reservation_id: str,
        idempotency_key: str,
        released_at: datetime.datetime,
        reason: str = "Released without usage",
        ledger_entry: UsageLedgerEntry | None = None,
    ) -> tuple[ReservationRecord, bool]:  # (record, is_replayed)
        """Release a pending reservation without charging actual usage.

        Transitions reservation PENDING -> RELEASED.
        Returns reserved allowance to the budget.
        """
        ...

    def append_ledger_entry(self, entry: UsageLedgerEntry) -> None:
        """Append an audit entry to the immutable usage ledger."""
        ...

    def get_ledger_entries(self, scope: BudgetScope, limit: int = 100) -> list[UsageLedgerEntry]:
        """Fetch audit log ledger entries for the given scope."""
        ...
