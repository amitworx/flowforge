"""Reference thread-safe, in-memory repository implementing BudgetRepository.

Designed strictly for unit and concurrent testing without external DB dependencies.
Provides deterministic locking, cross-tenant isolation, fencing token validation,
and idempotency checking.
"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Mapping
from typing import Any

from flowforge.ai.governance.exceptions import (
    AIDuplicateSettlementError,
    AIGovernanceError,
    AIInvalidReservationStateError,
    AIReservationNotFoundError,
    AIStaleFencedAttemptError,
)
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetSettings,
    OverrunMarker,
    ReservationRecord,
    ReservationStatus,
    SettlementRecord,
    SettlementType,
    UsageLedgerEntry,
)


class InMemoryBudgetRepository:
    """Thread-safe reference in-memory repository for AI budget and audit governance."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._settings: dict[str, BudgetSettings] = {}
        # reservations keyed by reservation_id
        self._reservations: dict[str, ReservationRecord] = {}
        # map idempotency_key -> reservation_id
        self._reservation_idempotency: dict[str, str] = {}
        # settlements keyed by settlement_id
        self._settlements: dict[str, SettlementRecord] = {}
        # map reservation_id -> settlement_id
        self._settlement_by_reservation: dict[str, str] = {}
        # map idempotency_key -> settlement_id
        self._settlement_idempotency: dict[str, str] = {}
        # ledger entries
        self._ledger: list[UsageLedgerEntry] = []

    def get_budget_settings(self, scope: BudgetScope) -> BudgetSettings | None:
        with self._lock:
            return self._settings.get(scope.key)

    def set_budget_settings(self, settings: BudgetSettings) -> None:
        with self._lock:
            self._settings[settings.scope.key] = settings

    def get_aggregate_usage(self, scope: BudgetScope) -> tuple[int, int, int]:
        """Returns (settled_calls, settled_tokens, settled_cost_micros) for a scope."""
        with self._lock:
            total_calls = 0
            settled_tokens = 0
            settled_cost = 0
            for s in self._settlements.values():
                if (
                    self._matches_scope(s.scope, scope)
                    and s.settlement_type != SettlementType.ZERO_COST_REJECTED
                ):
                    total_calls += 1
                    settled_tokens += s.actual_total_tokens
                    settled_cost += s.actual_cost_micros
            return (total_calls, settled_tokens, settled_cost)

    def get_active_reserved_usage(self, scope: BudgetScope) -> tuple[int, int, int]:
        """Returns (reserved_calls, reserved_tokens, reserved_cost_micros) for pending reservations."""
        with self._lock:
            calls = 0
            tokens = 0
            cost = 0
            for r in self._reservations.values():
                if r.status == ReservationStatus.PENDING and self._matches_scope(r.scope, scope):
                    calls += 1
                    tokens += r.estimated_total_tokens
                    cost += r.estimated_cost_micros
            return (calls, tokens, cost)

    def _matches_scope(self, item_scope: BudgetScope, target_scope: BudgetScope) -> bool:
        """Scope matching: target_scope matches item_scope if all target_scope specified fields match."""
        if target_scope.scope_type != item_scope.scope_type:
            # Check hierarchical inheritance: TENANT covers WORKFLOW, RUN under the same tenant
            if target_scope.tenant_id and item_scope.tenant_id != target_scope.tenant_id:
                return False
            if target_scope.workflow_id and item_scope.workflow_id != target_scope.workflow_id:
                return False
            if target_scope.run_id and item_scope.run_id != target_scope.run_id:
                return False
            return not (target_scope.provider and item_scope.provider != target_scope.provider)

        if target_scope.tenant_id != item_scope.tenant_id:
            return False
        if target_scope.workflow_id != item_scope.workflow_id:
            return False
        if target_scope.run_id != item_scope.run_id:
            return False
        return target_scope.provider == item_scope.provider

    def get_reservation_by_id(self, reservation_id: str) -> ReservationRecord | None:
        with self._lock:
            return self._reservations.get(reservation_id)

    def get_reservation_by_idempotency_key(self, idempotency_key: str) -> ReservationRecord | None:
        with self._lock:
            res_id = self._reservation_idempotency.get(idempotency_key)
            if res_id:
                return self._reservations.get(res_id)
            return None

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
        with self._lock:
            # Check idempotency
            existing_id = self._reservation_idempotency.get(idempotency_key)
            if existing_id:
                existing = self._reservations[existing_id]
                # Verify match
                if (
                    existing.scope.key == scope.key
                    and existing.provider == provider
                    and existing.model == model
                    and existing.estimated_total_tokens == estimated_total_tokens
                ):
                    return existing
                raise AIDuplicateSettlementError(
                    existing_id,
                    f"Idempotency key '{idempotency_key}' reused with conflicting reservation parameters",
                )

            if reservation_id in self._reservations:
                raise AIGovernanceError(f"Reservation ID '{reservation_id}' already exists")

            record = ReservationRecord(
                reservation_id=reservation_id,
                idempotency_key=idempotency_key,
                scope=scope,
                provider=provider,
                model=model,
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_completion_tokens=estimated_completion_tokens,
                estimated_total_tokens=estimated_total_tokens,
                estimated_cost_micros=estimated_cost_micros,
                attempt_number=attempt_number,
                status=ReservationStatus.PENDING,
                created_at=created_at,
                expires_at=expires_at,
                audit_metadata=dict(audit_metadata) if audit_metadata else {},
            )
            self._reservations[reservation_id] = record
            self._reservation_idempotency[idempotency_key] = reservation_id
            return record

    def get_settlement_by_id(self, settlement_id: str) -> SettlementRecord | None:
        with self._lock:
            return self._settlements.get(settlement_id)

    def get_settlement_by_idempotency_key(self, idempotency_key: str) -> SettlementRecord | None:
        with self._lock:
            s_id = self._settlement_idempotency.get(idempotency_key)
            if s_id:
                return self._settlements.get(s_id)
            return None

    def get_settlement_by_reservation_id(self, reservation_id: str) -> SettlementRecord | None:
        with self._lock:
            s_id = self._settlement_by_reservation.get(reservation_id)
            if s_id:
                return self._settlements.get(s_id)
            return None

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
    ) -> SettlementRecord:
        with self._lock:
            # Check idempotency by key
            existing_s_id = self._settlement_idempotency.get(idempotency_key)
            if existing_s_id:
                existing = self._settlements[existing_s_id]
                if (
                    existing.reservation_id == reservation_id
                    and existing.actual_total_tokens == actual_total_tokens
                    and existing.settlement_type == settlement_type
                ):
                    return existing
                raise AIDuplicateSettlementError(
                    reservation_id,
                    f"Conflicting settlement parameters for idempotency key '{idempotency_key}'",
                )

            res = self._reservations.get(reservation_id)
            if not res:
                raise AIReservationNotFoundError(reservation_id)

            # Check if reservation is already settled
            if res.status == ReservationStatus.SETTLED:
                existing_s_id = self._settlement_by_reservation.get(reservation_id)
                if existing_s_id:
                    existing = self._settlements[existing_s_id]
                    # If duplicate redelivery
                    if (
                        existing.idempotency_key == idempotency_key
                        and existing.actual_total_tokens == actual_total_tokens
                    ):
                        return existing
                raise AIDuplicateSettlementError(reservation_id)

            if res.status != ReservationStatus.PENDING:
                raise AIInvalidReservationStateError(reservation_id, res.status.value, "settle")

            # Fencing check: attempt_number must not be stale
            if attempt_number < res.attempt_number:
                raise AIStaleFencedAttemptError(reservation_id, res.attempt_number, attempt_number)

            record = SettlementRecord(
                settlement_id=settlement_id,
                reservation_id=reservation_id,
                idempotency_key=idempotency_key,
                scope=res.scope,
                provider=res.provider,
                model=res.model,
                settlement_type=settlement_type,
                actual_prompt_tokens=actual_prompt_tokens,
                actual_completion_tokens=actual_completion_tokens,
                actual_total_tokens=actual_total_tokens,
                actual_cost_micros=actual_cost_micros,
                attempt_number=attempt_number,
                settled_at=settled_at,
                overrun_marker=overrun_marker,
                overrun_tokens=overrun_tokens,
                overrun_cost_micros=overrun_cost_micros,
                duration_seconds=duration_seconds,
                audit_metadata=dict(audit_metadata) if audit_metadata else {},
            )

            # Update reservation status to SETTLED
            updated_res = ReservationRecord(
                reservation_id=res.reservation_id,
                idempotency_key=res.idempotency_key,
                scope=res.scope,
                provider=res.provider,
                model=res.model,
                estimated_prompt_tokens=res.estimated_prompt_tokens,
                estimated_completion_tokens=res.estimated_completion_tokens,
                estimated_total_tokens=res.estimated_total_tokens,
                estimated_cost_micros=res.estimated_cost_micros,
                attempt_number=max(res.attempt_number, attempt_number),
                status=ReservationStatus.SETTLED,
                created_at=res.created_at,
                expires_at=res.expires_at,
                audit_metadata=res.audit_metadata,
            )
            self._reservations[reservation_id] = updated_res
            self._settlements[settlement_id] = record
            self._settlement_idempotency[idempotency_key] = settlement_id
            self._settlement_by_reservation[reservation_id] = settlement_id
            return record

    def release_reservation(
        self,
        *,
        reservation_id: str,
        idempotency_key: str,
        released_at: datetime.datetime,
        reason: str = "Released without usage",
    ) -> ReservationRecord:
        del idempotency_key, released_at, reason
        with self._lock:
            res = self._reservations.get(reservation_id)
            if not res:
                raise AIReservationNotFoundError(reservation_id)

            if res.status == ReservationStatus.RELEASED:
                return res

            if res.status != ReservationStatus.PENDING:
                raise AIInvalidReservationStateError(reservation_id, res.status.value, "release")

            updated_res = ReservationRecord(
                reservation_id=res.reservation_id,
                idempotency_key=res.idempotency_key,
                scope=res.scope,
                provider=res.provider,
                model=res.model,
                estimated_prompt_tokens=res.estimated_prompt_tokens,
                estimated_completion_tokens=res.estimated_completion_tokens,
                estimated_total_tokens=res.estimated_total_tokens,
                estimated_cost_micros=res.estimated_cost_micros,
                attempt_number=res.attempt_number,
                status=ReservationStatus.RELEASED,
                created_at=res.created_at,
                expires_at=res.expires_at,
                audit_metadata=res.audit_metadata,
            )
            self._reservations[reservation_id] = updated_res
            return updated_res

    def append_ledger_entry(self, entry: UsageLedgerEntry) -> None:
        with self._lock:
            self._ledger.append(entry)

    def get_ledger_entries(self, scope: BudgetScope, limit: int = 100) -> list[UsageLedgerEntry]:
        with self._lock:
            entries = [e for e in self._ledger if self._matches_scope(e.scope, scope)]
            return entries[-limit:]
