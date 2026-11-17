"""Reference thread-safe, in-memory repository implementing BudgetRepository.

Designed strictly for unit and concurrent testing without external DB dependencies.
Provides deterministic locking, cross-tenant isolation, fencing token validation,
and idempotency checking.
"""

from __future__ import annotations

import datetime
import math
import threading
from collections.abc import Mapping
from typing import Any

from flowforge.ai.governance.exceptions import (
    AIBudgetExhaustedError,
    AIDuplicateSettlementError,
    AIGovernanceError,
    AIInvalidReservationStateError,
    AIReservationNotFoundError,
    AIStaleFencedAttemptError,
)
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
    DispatchStatus,
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
        # map (tenant_id, idempotency_key) -> reservation_id (scoped idempotency)
        self._reservation_idempotency: dict[tuple[str | None, str], str] = {}
        # global map to catch cross-tenant idempotency key hijacking
        self._global_reservation_idempotency: dict[str, str] = {}
        # settlements keyed by settlement_id
        self._settlements: dict[str, SettlementRecord] = {}
        # map reservation_id -> settlement_id
        self._settlement_by_reservation: dict[str, str] = {}
        # map (tenant_id, idempotency_key) -> settlement_id
        self._settlement_idempotency: dict[tuple[str | None, str], str] = {}
        self._global_settlement_idempotency: dict[str, str] = {}
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
            # Check hierarchical inheritance: GLOBAL covers everything; TENANT covers WORKFLOW, RUN under the same tenant
            if target_scope.scope_type == BudgetScopeType.GLOBAL:
                return True
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

    def _check_scope_match(self, item_scope: BudgetScope, scope: BudgetScope | None) -> bool:
        """Enforces full-scope isolation for lookups:

        - If scope is provided: must match tenant_id, workflow_id (if present in scope), and run_id (if present in scope).
        - If scope is None: only global/unscoped items are returned.
        """
        if scope is not None:
            if item_scope.tenant_id != scope.tenant_id:
                return False
            if scope.workflow_id is not None and item_scope.workflow_id != scope.workflow_id:
                return False
            return not (scope.run_id is not None and item_scope.run_id != scope.run_id)
        return item_scope.tenant_id is None

    def get_reservation_by_id(
        self,
        reservation_id: str,
        scope: BudgetScope | None = None,
        *,
        internal_privileged: bool = False,
    ) -> ReservationRecord | None:
        with self._lock:
            res = self._reservations.get(reservation_id)
            if not res:
                return None
            if not internal_privileged and not self._check_scope_match(res.scope, scope):
                return None
            return res

    def get_reservation_by_idempotency_key(
        self, idempotency_key: str, scope: BudgetScope | None = None
    ) -> ReservationRecord | None:
        with self._lock:
            tenant_id = scope.tenant_id if scope else None
            res_id = self._reservation_idempotency.get((tenant_id, idempotency_key))
            if not res_id and not scope:
                res_id = self._global_reservation_idempotency.get(idempotency_key)
            if res_id:
                return self._reservations.get(res_id)
            return None

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
    ) -> tuple[ReservationRecord, bool]:
        """Atomically checks all budget limits across scope and ancestors, creates reservation,

        and optionally records ledger entry under a single lock/transaction.
        """
        with self._lock:
            # Check cross-tenant idempotency key collision / hijacking
            global_res_id = self._global_reservation_idempotency.get(idempotency_key)
            scoped_key = (scope.tenant_id, idempotency_key)
            existing_id = self._reservation_idempotency.get(scoped_key)

            if global_res_id and not existing_id:
                raise AIDuplicateSettlementError(
                    global_res_id,
                    "Idempotency key reused across different tenant or scope",
                )

            if existing_id:
                existing = self._reservations[existing_id]
                # Strict parameter comparison: scope, provider, model, prompt tokens, completion tokens, cost, attempt
                if (
                    existing.scope.key == scope.key
                    and existing.provider == provider
                    and existing.model == model
                    and existing.estimated_prompt_tokens == estimated_prompt_tokens
                    and existing.estimated_completion_tokens == estimated_completion_tokens
                    and existing.estimated_total_tokens == estimated_total_tokens
                    and existing.estimated_cost_micros == estimated_cost_micros
                    and existing.attempt_number == attempt_number
                ):
                    return (existing, True)
                raise AIDuplicateSettlementError(
                    existing_id,
                    "Idempotency key reused with conflicting reservation parameters",
                )

            if reservation_id in self._reservations:
                raise AIGovernanceError("Reservation identifier already exists")

            # Check limits across scope and all ancestor scopes under this exact lock!
            scopes_to_check = [scope, *ancestor_scopes]
            for s in scopes_to_check:
                settings = self._settings.get(s.key)
                if not settings:
                    continue

                settled_calls, settled_tokens, settled_cost = self._get_usage_unlocked(s)
                res_calls, res_tokens, res_cost = self._get_active_reserved_unlocked(s)

                if (
                    settings.max_calls is not None
                    and settled_calls + res_calls + 1 > settings.max_calls
                ):
                    raise AIBudgetExhaustedError(
                        "Budget limit for max calls exceeded",
                        limit_name="max_calls",
                        requested_value=1,
                        current_value=settled_calls + res_calls,
                        limit_value=settings.max_calls,
                        scope_key=s.key,
                    )

                if (
                    settings.max_total_tokens is not None
                    and settled_tokens + res_tokens + estimated_total_tokens
                    > settings.max_total_tokens
                ):
                    raise AIBudgetExhaustedError(
                        "Budget limit for total tokens exceeded by estimated request",
                        limit_name="max_total_tokens",
                        requested_value=estimated_total_tokens,
                        current_value=settled_tokens + res_tokens,
                        limit_value=settings.max_total_tokens,
                        scope_key=s.key,
                    )

                if (
                    settings.max_estimated_cost_micros is not None
                    and settled_cost + res_cost + estimated_cost_micros
                    > settings.max_estimated_cost_micros
                ):
                    raise AIBudgetExhaustedError(
                        "Budget limit for estimated cost exceeded",
                        limit_name="max_estimated_cost_micros",
                        requested_value=estimated_cost_micros,
                        current_value=settled_cost + res_cost,
                        limit_value=settings.max_estimated_cost_micros,
                        scope_key=s.key,
                    )

            # Create record
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
                audit_metadata=audit_metadata or {},
            )
            self._reservations[reservation_id] = record
            self._reservation_idempotency[scoped_key] = reservation_id
            self._global_reservation_idempotency[idempotency_key] = reservation_id

            if ledger_entry:
                self._ledger.append(ledger_entry)

            return (record, False)

    def _get_usage_unlocked(self, scope: BudgetScope) -> tuple[int, int, int]:
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

    def _get_active_reserved_unlocked(self, scope: BudgetScope) -> tuple[int, int, int]:
        calls = 0
        tokens = 0
        cost = 0
        for r in self._reservations.values():
            if r.status == ReservationStatus.PENDING and self._matches_scope(r.scope, scope):
                calls += 1
                tokens += r.estimated_total_tokens
                cost += r.estimated_cost_micros
        return (calls, tokens, cost)

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
        record, _ = self.reserve_atomic(
            reservation_id=reservation_id,
            idempotency_key=idempotency_key,
            scope=scope,
            ancestor_scopes=[],
            provider=provider,
            model=model,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
            estimated_total_tokens=estimated_total_tokens,
            estimated_cost_micros=estimated_cost_micros,
            attempt_number=attempt_number,
            created_at=created_at,
            expires_at=expires_at,
            audit_metadata=audit_metadata,
        )
        return record

    def get_settlement_by_id(
        self,
        settlement_id: str,
        scope: BudgetScope | None = None,
        *,
        internal_privileged: bool = False,
    ) -> SettlementRecord | None:
        with self._lock:
            stl = self._settlements.get(settlement_id)
            if not stl:
                return None
            if not internal_privileged and not self._check_scope_match(stl.scope, scope):
                return None
            return stl

    def get_settlement_by_idempotency_key(
        self, idempotency_key: str, scope: BudgetScope | None = None
    ) -> SettlementRecord | None:
        with self._lock:
            tenant_id = scope.tenant_id if scope else None
            s_id = self._settlement_idempotency.get((tenant_id, idempotency_key))
            if not s_id and not scope:
                s_id = self._global_settlement_idempotency.get(idempotency_key)
            if s_id:
                stl = self._settlements.get(s_id)
                if stl and not self._check_scope_match(stl.scope, scope):
                    return None
                return stl
            return None

    def get_settlement_by_reservation_id(
        self,
        reservation_id: str,
        scope: BudgetScope | None = None,
        *,
        internal_privileged: bool = False,
    ) -> SettlementRecord | None:
        with self._lock:
            s_id = self._settlement_by_reservation.get(reservation_id)
            if s_id:
                stl = self._settlements.get(s_id)
                if (
                    stl
                    and not internal_privileged
                    and not self._check_scope_match(stl.scope, scope)
                ):
                    return None
                return stl
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
        ledger_entry: UsageLedgerEntry | None = None,
    ) -> tuple[SettlementRecord, bool]:
        with self._lock:
            res = self._reservations.get(reservation_id)
            if not res:
                raise AIReservationNotFoundError(reservation_id)

            tenant_id = res.scope.tenant_id
            scoped_key = (tenant_id, idempotency_key)

            # Check cross-tenant hijacking
            global_s_id = self._global_settlement_idempotency.get(idempotency_key)
            existing_s_id = self._settlement_idempotency.get(scoped_key)

            if global_s_id and not existing_s_id:
                raise AIDuplicateSettlementError(
                    reservation_id,
                    "Idempotency key reused across different tenant",
                )

            if existing_s_id:
                existing = self._settlements[existing_s_id]
                # Strict comparison across ALL settlement fields including duration
                if (
                    existing.reservation_id == reservation_id
                    and existing.actual_prompt_tokens == actual_prompt_tokens
                    and existing.actual_completion_tokens == actual_completion_tokens
                    and existing.actual_total_tokens == actual_total_tokens
                    and existing.actual_cost_micros == actual_cost_micros
                    and existing.settlement_type == settlement_type
                    and existing.attempt_number == attempt_number
                    and existing.overrun_marker == overrun_marker
                    and existing.overrun_tokens == overrun_tokens
                    and existing.overrun_cost_micros == overrun_cost_micros
                    and math.isclose(existing.duration_seconds, duration_seconds, rel_tol=1e-5)
                ):
                    return (existing, True)
                raise AIDuplicateSettlementError(
                    reservation_id,
                    "Conflicting settlement parameters for idempotency key",
                )

            # Check if reservation is already settled
            if res.status == ReservationStatus.SETTLED:
                existing_s_id = self._settlement_by_reservation.get(reservation_id)
                if existing_s_id:
                    existing = self._settlements[existing_s_id]
                    if (
                        existing.idempotency_key == idempotency_key
                        and existing.actual_prompt_tokens == actual_prompt_tokens
                        and existing.actual_completion_tokens == actual_completion_tokens
                        and existing.actual_total_tokens == actual_total_tokens
                        and existing.actual_cost_micros == actual_cost_micros
                        and existing.settlement_type == settlement_type
                        and existing.attempt_number == attempt_number
                    ):
                        return (existing, True)
                raise AIDuplicateSettlementError(reservation_id)

            if res.status != ReservationStatus.PENDING:
                raise AIInvalidReservationStateError(reservation_id, res.status.value, "settle")

            # Fencing check: attempt_number must exactly match reservation attempt_number
            if attempt_number != res.attempt_number:
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
                audit_metadata=audit_metadata or {},
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
                attempt_number=res.attempt_number,
                status=ReservationStatus.SETTLED,
                created_at=res.created_at,
                expires_at=res.expires_at,
                audit_metadata=res.audit_metadata,
                dispatch_status=res.dispatch_status,
            )
            self._reservations[reservation_id] = updated_res
            self._settlements[settlement_id] = record
            self._settlement_idempotency[scoped_key] = settlement_id
            self._global_settlement_idempotency[idempotency_key] = settlement_id
            self._settlement_by_reservation[reservation_id] = settlement_id

            if ledger_entry:
                self._ledger.append(ledger_entry)

            return (record, False)

    def release_reservation(
        self,
        *,
        reservation_id: str,
        idempotency_key: str,
        scope: BudgetScope | None = None,
        released_at: datetime.datetime,
        reason: str = "Released without usage",
        ledger_entry: UsageLedgerEntry | None = None,
    ) -> tuple[ReservationRecord, bool]:
        del idempotency_key, released_at, reason
        with self._lock:
            res = self._reservations.get(reservation_id)
            if not res:
                raise AIReservationNotFoundError(reservation_id)

            # Tenant authorization check:
            # If reservation has tenant_id, caller must supply matching scope
            if res.scope.tenant_id:
                if scope is None or scope.tenant_id != res.scope.tenant_id:
                    raise AIReservationNotFoundError(reservation_id)
                if scope.workflow_id is not None and scope.workflow_id != res.scope.workflow_id:
                    raise AIReservationNotFoundError(reservation_id)
                if scope.run_id is not None and scope.run_id != res.scope.run_id:
                    raise AIReservationNotFoundError(reservation_id)
            else:
                if scope is not None and scope.tenant_id is not None:
                    raise AIReservationNotFoundError(reservation_id)

            if res.status == ReservationStatus.RELEASED:
                return (res, True)

            # Prevent release if already settled or expired
            if res.status != ReservationStatus.PENDING:
                raise AIInvalidReservationStateError(reservation_id, res.status.value, "release")

            # Prevent release if request was already dispatched to a provider!
            if res.dispatch_status != DispatchStatus.NOT_DISPATCHED:
                raise AIInvalidReservationStateError(
                    reservation_id, res.dispatch_status.value, "release"
                )

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
                dispatch_status=res.dispatch_status,
            )
            self._reservations[reservation_id] = updated_res
            if ledger_entry:
                self._ledger.append(ledger_entry)
            return (updated_res, False)

    def transition_dispatch_status(
        self,
        *,
        reservation_id: str,
        scope: BudgetScope | None = None,
        from_status: DispatchStatus,
        to_status: DispatchStatus,
        attempt_number: int,
    ) -> ReservationRecord:
        del from_status
        with self._lock:
            res = self._reservations.get(reservation_id)
            if not res:
                raise AIReservationNotFoundError(reservation_id)

            # Scope validation
            if res.scope.tenant_id:
                if scope is None or scope.tenant_id != res.scope.tenant_id:
                    raise AIReservationNotFoundError(reservation_id)
                if scope.workflow_id is not None and scope.workflow_id != res.scope.workflow_id:
                    raise AIReservationNotFoundError(reservation_id)
                if scope.run_id is not None and scope.run_id != res.scope.run_id:
                    raise AIReservationNotFoundError(reservation_id)

            # Fencing attempt check
            if attempt_number != res.attempt_number:
                raise AIStaleFencedAttemptError(
                    reservation_id,
                    expected_attempt=res.attempt_number,
                    actual_attempt=attempt_number,
                )

            # Validate legal transitions
            legal_transitions = {
                DispatchStatus.NOT_DISPATCHED: {DispatchStatus.DISPATCH_STARTED},
                DispatchStatus.DISPATCH_STARTED: {
                    DispatchStatus.RESPONSE_RECEIVED,
                    DispatchStatus.RESPONSE_LOST,
                    DispatchStatus.BILLING_UNCERTAIN,
                },
                DispatchStatus.RESPONSE_LOST: {DispatchStatus.BILLING_UNCERTAIN},
                DispatchStatus.BILLING_UNCERTAIN: set(),
                DispatchStatus.RESPONSE_RECEIVED: set(),
            }
            if to_status not in legal_transitions.get(res.dispatch_status, set()):
                raise AIInvalidReservationStateError(
                    reservation_id, res.dispatch_status.value, "dispatch"
                )

            updated = ReservationRecord(
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
                status=res.status,
                created_at=res.created_at,
                expires_at=res.expires_at,
                audit_metadata=res.audit_metadata,
                dispatch_status=to_status,
            )
            self._reservations[reservation_id] = updated
            return updated

    def append_ledger_entry(self, entry: UsageLedgerEntry) -> None:
        with self._lock:
            self._ledger.append(entry)

    def get_ledger_entries(self, scope: BudgetScope, limit: int = 100) -> list[UsageLedgerEntry]:
        with self._lock:
            entries = [e for e in self._ledger if self._matches_scope(e.scope, scope)]
            return entries[-limit:]
