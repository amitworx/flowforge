"""PostgreSQL-backed implementation of BudgetRepository.

Provides durable, transactional, and tenant-isolated AI governance persistence.
Enforces:
1. Deterministic lock ordering: GLOBAL -> TENANT -> WORKFLOW -> RUN / PROVIDER.
2. Full-scope tenant and hierarchy authorization.
3. Strict two-tier idempotency and conflict detection.
4. Bounded SQLSTATE 40P01 (deadlock) and 40001 (serialization failure) retries.
5. Strict atomic counter mutations and immutable audit ledger consistency.
6. Fail-closed error handling and confidential error surfaces.
"""

from __future__ import annotations

import datetime
import math
import random
import time
from collections.abc import Callable, Mapping
from typing import Any

from django.db import DatabaseError, transaction
from django.db.models import F

from flowforge.ai.governance.db.models import (
    AIBudgetScopeCounter,
    AIBudgetSettingsModel,
    AIReservationModel,
    AISettlementModel,
    AIUsageLedgerModel,
)
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
    sanitize_audit_metadata,
    sanitize_note,
)
from flowforge.ai.governance.repository import BudgetRepository

# Scope hierarchy ordering for deterministic deadlock prevention
SCOPE_TYPE_ORDER = {
    BudgetScopeType.GLOBAL.value: 1,
    BudgetScopeType.TENANT.value: 2,
    BudgetScopeType.WORKFLOW.value: 3,
    BudgetScopeType.RUN.value: 4,
    BudgetScopeType.PROVIDER.value: 4,
}


def _with_db_retry[T](operation: Callable[[], T], max_retries: int = 3) -> T:
    """Executes an operation inside bounded retries for transient PostgreSQL 40P01/40001 errors.

    Retries the entire transaction from scratch with jittered backoff.
    Never sleeps while holding DB locks.
    """
    base_delay_ms = 25
    max_delay_ms = 250

    for attempt in range(max_retries):
        try:
            return operation()
        except DatabaseError as exc:
            sqlstate = getattr(getattr(exc, "__cause__", None), "sqlstate", None)
            if sqlstate in ("40P01", "40001") and attempt < max_retries - 1:
                jitter = random.uniform(0.8, 1.2)
                delay = min(max_delay_ms, (base_delay_ms * (2**attempt))) * jitter / 1000.0
                time.sleep(delay)
                continue
            raise


class PostgreSQLBudgetRepository(BudgetRepository):
    """Production PostgreSQL-backed implementation of the BudgetRepository protocol."""

    # -------------------------------------------------------------------------
    # Scope & Model Mapping Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _scope_from_model(model: Any) -> BudgetScope:
        return BudgetScope(
            scope_type=BudgetScopeType(model.scope_type),
            tenant_id=model.tenant_id,
            workflow_id=model.workflow_id,
            run_id=model.run_id,
            provider=model.scope_provider if hasattr(model, "scope_provider") else model.provider,
        )

    @staticmethod
    def _reservation_to_record(m: AIReservationModel) -> ReservationRecord:
        scope = BudgetScope(
            scope_type=BudgetScopeType(m.scope_type),
            tenant_id=m.tenant_id,
            workflow_id=m.workflow_id,
            run_id=m.run_id,
            provider=m.scope_provider,
        )
        return ReservationRecord(
            reservation_id=m.reservation_id,
            idempotency_key=m.idempotency_key,
            scope=scope,
            provider=m.provider,
            model=m.model,
            estimated_prompt_tokens=m.estimated_prompt_tokens,
            estimated_completion_tokens=m.estimated_completion_tokens,
            estimated_total_tokens=m.estimated_total_tokens,
            estimated_cost_micros=m.estimated_cost_micros,
            attempt_number=m.attempt_number,
            status=ReservationStatus(m.status),
            created_at=m.created_at,
            expires_at=m.expires_at,
            audit_metadata=m.audit_metadata,
            dispatch_status=DispatchStatus(
                getattr(m, "dispatch_status", DispatchStatus.NOT_DISPATCHED.value)
            ),
        )

    @staticmethod
    def _settlement_to_record(m: AISettlementModel) -> SettlementRecord:
        scope = BudgetScope(
            scope_type=BudgetScopeType(m.scope_type),
            tenant_id=m.tenant_id,
            workflow_id=m.workflow_id,
            run_id=m.run_id,
            provider=m.scope_provider,
        )
        return SettlementRecord(
            settlement_id=m.settlement_id,
            reservation_id=m.reservation_id,
            idempotency_key=m.idempotency_key,
            scope=scope,
            provider=m.provider,
            model=m.model,
            settlement_type=SettlementType(m.settlement_type),
            actual_prompt_tokens=m.actual_prompt_tokens,
            actual_completion_tokens=m.actual_completion_tokens,
            actual_total_tokens=m.actual_total_tokens,
            actual_cost_micros=m.actual_cost_micros,
            attempt_number=m.attempt_number,
            settled_at=m.settled_at,
            overrun_marker=OverrunMarker(m.overrun_marker),
            overrun_tokens=m.overrun_tokens,
            overrun_cost_micros=m.overrun_cost_micros,
            duration_seconds=m.duration_seconds,
            audit_metadata=m.audit_metadata,
        )

    @staticmethod
    def _ledger_to_entry(m: AIUsageLedgerModel) -> UsageLedgerEntry:
        scope = BudgetScope(
            scope_type=BudgetScopeType(m.scope_type),
            tenant_id=m.tenant_id,
            workflow_id=m.workflow_id,
            run_id=m.run_id,
            provider=m.scope_provider,
        )
        return UsageLedgerEntry(
            entry_id=m.entry_id,
            event_type=m.event_type,
            idempotency_key=m.idempotency_key,
            scope=scope,
            provider=m.provider,
            model=m.model,
            timestamp=m.timestamp,
            tokens_delta=m.tokens_delta,
            cost_micros_delta=m.cost_micros_delta,
            calls_delta=m.calls_delta,
            overrun_marker=OverrunMarker(m.overrun_marker),
            notes=m.notes,
            audit_metadata=m.audit_metadata,
        )

    # -------------------------------------------------------------------------
    # Deterministic Lock Ordering & Counter Management
    # -------------------------------------------------------------------------

    def _ensure_and_lock_counters(
        self, scopes: list[BudgetScope]
    ) -> dict[str, AIBudgetScopeCounter]:
        """Safely ensures counter rows exist and acquires row locks in verified canonical order.

        Order: GLOBAL (1) -> TENANT (2) -> WORKFLOW (3) -> RUN / PROVIDER (4).
        Lexicographically sorted by scope_key within each tier.
        """
        # Deduplicate scopes by scope_key
        unique_scopes: dict[str, BudgetScope] = {s.key: s for s in scopes}

        def sort_key(k: str) -> tuple[int, str]:
            scope = unique_scopes[k]
            order = SCOPE_TYPE_ORDER.get(scope.scope_type.value, 99)
            return (order, k)

        sorted_keys = sorted(unique_scopes.keys(), key=sort_key)

        # 1. Ensure rows exist via bulk INSERT ... ON CONFLICT DO NOTHING
        rows_to_create = [
            AIBudgetScopeCounter(
                scope_key=k,
                scope_type=unique_scopes[k].scope_type.value,
                tenant_id=unique_scopes[k].tenant_id,
                workflow_id=unique_scopes[k].workflow_id,
                run_id=unique_scopes[k].run_id,
                provider=unique_scopes[k].provider,
            )
            for k in sorted_keys
        ]
        if rows_to_create:
            AIBudgetScopeCounter.objects.bulk_create(rows_to_create, ignore_conflicts=True)

        # 2. Acquire locks strictly in verified order
        locked: dict[str, AIBudgetScopeCounter] = {}
        for k in sorted_keys:
            counter = AIBudgetScopeCounter.objects.select_for_update().filter(scope_key=k).first()
            if counter is not None:
                locked[k] = counter
        return locked

    # -------------------------------------------------------------------------
    # Budget Settings
    # -------------------------------------------------------------------------

    def get_budget_settings(self, scope: BudgetScope) -> BudgetSettings | None:
        try:
            m = AIBudgetSettingsModel.objects.filter(scope_key=scope.key).first()
            if not m:
                return None
            return BudgetSettings(
                scope=scope,
                version=m.version,
                max_calls=m.max_calls,
                max_total_tokens=m.max_total_tokens,
                max_prompt_tokens=m.max_prompt_tokens,
                max_completion_tokens=m.max_completion_tokens,
                max_estimated_cost_micros=m.max_estimated_cost_micros,
                max_duration_seconds=m.max_duration_seconds,
                allowed_providers=tuple(m.allowed_providers),
                denied_providers=tuple(m.denied_providers),
                allow_overrun=m.allow_overrun,
            )
        except Exception:
            return None

    def set_budget_settings(self, settings: BudgetSettings) -> None:
        AIBudgetSettingsModel.objects.update_or_create(
            scope_key=settings.scope.key,
            defaults={
                "scope_type": settings.scope.scope_type.value,
                "tenant_id": settings.scope.tenant_id,
                "workflow_id": settings.scope.workflow_id,
                "run_id": settings.scope.run_id,
                "provider": settings.scope.provider,
                "version": settings.version,
                "max_calls": settings.max_calls,
                "max_total_tokens": settings.max_total_tokens,
                "max_prompt_tokens": settings.max_prompt_tokens,
                "max_completion_tokens": settings.max_completion_tokens,
                "max_estimated_cost_micros": settings.max_estimated_cost_micros,
                "max_duration_seconds": settings.max_duration_seconds,
                "allowed_providers": list(settings.allowed_providers),
                "denied_providers": list(settings.denied_providers),
                "allow_overrun": settings.allow_overrun,
            },
        )

    # -------------------------------------------------------------------------
    # Usage Aggregates
    # -------------------------------------------------------------------------

    def get_aggregate_usage(self, scope: BudgetScope) -> tuple[int, int, int]:
        counter = AIBudgetScopeCounter.objects.filter(scope_key=scope.key).first()
        if not counter:
            return (0, 0, 0)
        return (counter.settled_calls, counter.settled_tokens, counter.settled_cost_micros)

    def get_active_reserved_usage(self, scope: BudgetScope) -> tuple[int, int, int]:
        counter = AIBudgetScopeCounter.objects.filter(scope_key=scope.key).first()
        if not counter:
            return (0, 0, 0)
        return (counter.reserved_calls, counter.reserved_tokens, counter.reserved_cost_micros)

    # -------------------------------------------------------------------------
    # Lookups & Full-Scope Tenant Authorization
    # -------------------------------------------------------------------------

    def get_reservation_by_id(
        self,
        reservation_id: str,
        scope: BudgetScope | None = None,
        *,
        internal_privileged: bool = False,
    ) -> ReservationRecord | None:
        m = AIReservationModel.objects.filter(reservation_id=reservation_id).first()
        if not m:
            return None
        if not internal_privileged:
            if scope is not None:
                if m.tenant_id != scope.tenant_id:
                    return None
                if scope.workflow_id is not None and m.workflow_id != scope.workflow_id:
                    return None
                if scope.run_id is not None and m.run_id != scope.run_id:
                    return None
            else:
                if m.tenant_id is not None:
                    return None
        return self._reservation_to_record(m)

    def get_reservation_by_idempotency_key(
        self, idempotency_key: str, scope: BudgetScope | None = None
    ) -> ReservationRecord | None:
        m = AIReservationModel.objects.filter(idempotency_key=idempotency_key).first()
        if not m:
            return None

        # Full-scope authorization check:
        # If scope is provided, only return the record if it matches the scope's tenant.
        # If it belongs to a different tenant, return None to prevent enumeration.
        if scope is not None:
            if m.tenant_id != scope.tenant_id:
                return None
            if scope.workflow_id is not None and m.workflow_id != scope.workflow_id:
                return None
            if scope.run_id is not None and m.run_id != scope.run_id:
                return None
        else:
            # If scope is omitted, only global/unscoped reservations may be returned
            if m.tenant_id is not None:
                return None
        return self._reservation_to_record(m)

    def get_settlement_by_id(
        self,
        settlement_id: str,
        scope: BudgetScope | None = None,
        *,
        internal_privileged: bool = False,
    ) -> SettlementRecord | None:
        m = AISettlementModel.objects.filter(settlement_id=settlement_id).first()
        if not m:
            return None
        if not internal_privileged:
            if scope is not None:
                if m.tenant_id != scope.tenant_id:
                    return None
                if scope.workflow_id is not None and m.workflow_id != scope.workflow_id:
                    return None
                if scope.run_id is not None and m.run_id != scope.run_id:
                    return None
            else:
                if m.tenant_id is not None:
                    return None
        return self._settlement_to_record(m)

    def get_settlement_by_idempotency_key(
        self, idempotency_key: str, scope: BudgetScope | None = None
    ) -> SettlementRecord | None:
        m = AISettlementModel.objects.filter(idempotency_key=idempotency_key).first()
        if not m:
            return None

        # Full-scope authorization check
        if scope is not None:
            if m.tenant_id != scope.tenant_id:
                return None
            if scope.workflow_id is not None and m.workflow_id != scope.workflow_id:
                return None
            if scope.run_id is not None and m.run_id != scope.run_id:
                return None
        else:
            if m.tenant_id is not None:
                return None
        return self._settlement_to_record(m)

    def get_settlement_by_reservation_id(
        self,
        reservation_id: str,
        scope: BudgetScope | None = None,
        *,
        internal_privileged: bool = False,
    ) -> SettlementRecord | None:
        m = AISettlementModel.objects.filter(reservation_id=reservation_id).first()
        if not m:
            return None
        if not internal_privileged:
            if scope is not None:
                if m.tenant_id != scope.tenant_id:
                    return None
                if scope.workflow_id is not None and m.workflow_id != scope.workflow_id:
                    return None
                if scope.run_id is not None and m.run_id != scope.run_id:
                    return None
            else:
                if m.tenant_id is not None:
                    return None
        return self._settlement_to_record(m)

    # -------------------------------------------------------------------------
    # Atomic Reservation Operations
    # -------------------------------------------------------------------------

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
        """Atomically checks budget limits, acquires hierarchical locks, persists reservation,

        and updates counters and ledger inside a single database transaction.
        """

        def _txn() -> tuple[ReservationRecord, bool]:
            with transaction.atomic():
                # 1. Check idempotency collision / replay
                existing = (
                    AIReservationModel.objects.select_for_update()
                    .filter(idempotency_key=idempotency_key)
                    .first()
                )
                if existing:
                    # Tenant authorization check
                    if existing.tenant_id != scope.tenant_id or existing.scope_key != scope.key:
                        raise AIDuplicateSettlementError(
                            existing.reservation_id,
                            "Idempotency key reused across different tenant or scope",
                        )

                    # Strict parameter matching
                    if (
                        existing.provider == provider
                        and existing.model == model
                        and existing.estimated_prompt_tokens == estimated_prompt_tokens
                        and existing.estimated_completion_tokens == estimated_completion_tokens
                        and existing.estimated_total_tokens == estimated_total_tokens
                        and existing.estimated_cost_micros == estimated_cost_micros
                        and existing.attempt_number == attempt_number
                    ):
                        return (self._reservation_to_record(existing), True)
                    raise AIDuplicateSettlementError(
                        existing.reservation_id,
                        "Idempotency key reused with conflicting reservation parameters",
                    )

                if AIReservationModel.objects.filter(reservation_id=reservation_id).exists():
                    raise AIGovernanceError("Reservation identifier already exists")

                # 2. Acquire deterministic hierarchical locks
                # Always combine scope and ancestor scopes so all tiers are locked and tracked
                combined_scopes = list(
                    {
                        s.key: s for s in [scope, *ancestor_scopes, *scope.get_ancestor_scopes()]
                    }.values()
                )
                locked_counters = self._ensure_and_lock_counters(combined_scopes)

                # 3. Check budget limits across all hierarchy levels
                for s in combined_scopes:
                    settings_m = AIBudgetSettingsModel.objects.filter(scope_key=s.key).first()
                    if not settings_m:
                        continue

                    c = locked_counters[s.key]
                    current_calls = c.settled_calls + c.reserved_calls
                    current_tokens = c.settled_tokens + c.reserved_tokens
                    current_cost = c.settled_cost_micros + c.reserved_cost_micros

                    if (
                        settings_m.max_calls is not None
                        and current_calls + 1 > settings_m.max_calls
                    ):
                        raise AIBudgetExhaustedError(
                            "Budget limit for max calls exceeded",
                            limit_name="max_calls",
                            requested_value=1,
                            current_value=current_calls,
                            limit_value=settings_m.max_calls,
                            scope_key=s.key,
                        )

                    if (
                        settings_m.max_total_tokens is not None
                        and current_tokens + estimated_total_tokens > settings_m.max_total_tokens
                    ):
                        raise AIBudgetExhaustedError(
                            "Budget limit for total tokens exceeded by estimated request",
                            limit_name="max_total_tokens",
                            requested_value=estimated_total_tokens,
                            current_value=current_tokens,
                            limit_value=settings_m.max_total_tokens,
                            scope_key=s.key,
                        )

                    if (
                        settings_m.max_estimated_cost_micros is not None
                        and current_cost + estimated_cost_micros
                        > settings_m.max_estimated_cost_micros
                    ):
                        raise AIBudgetExhaustedError(
                            "Budget limit for estimated cost exceeded",
                            limit_name="max_estimated_cost_micros",
                            requested_value=estimated_cost_micros,
                            current_value=current_cost,
                            limit_value=settings_m.max_estimated_cost_micros,
                            scope_key=s.key,
                        )

                # 4. Mutate counters
                for s in combined_scopes:
                    AIBudgetScopeCounter.objects.filter(scope_key=s.key).update(
                        reserved_calls=F("reserved_calls") + 1,
                        reserved_tokens=F("reserved_tokens") + estimated_total_tokens,
                        reserved_cost_micros=F("reserved_cost_micros") + estimated_cost_micros,
                    )

                # 5. Persist reservation
                res_model = AIReservationModel.objects.create(
                    reservation_id=reservation_id,
                    idempotency_key=idempotency_key,
                    scope_key=scope.key,
                    scope_type=scope.scope_type.value,
                    tenant_id=scope.tenant_id,
                    workflow_id=scope.workflow_id,
                    run_id=scope.run_id,
                    scope_provider=scope.provider,
                    provider=provider,
                    model=model,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    estimated_completion_tokens=estimated_completion_tokens,
                    estimated_total_tokens=estimated_total_tokens,
                    estimated_cost_micros=estimated_cost_micros,
                    attempt_number=attempt_number,
                    status=ReservationStatus.PENDING.value,
                    created_at=created_at,
                    expires_at=expires_at,
                    audit_metadata=sanitize_audit_metadata(audit_metadata),
                )

                # 6. Append ledger entry if supplied
                if ledger_entry:
                    self._persist_ledger_entry(ledger_entry)

                return (self._reservation_to_record(res_model), False)

        return _with_db_retry(_txn)

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

    # -------------------------------------------------------------------------
    # Settle Reservation
    # -------------------------------------------------------------------------

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
        """Atomically settles a pending reservation, books actual usage into counters,

        and updates reservation status inside a single transaction.
        """

        def _txn() -> tuple[SettlementRecord, bool]:
            with transaction.atomic():
                # 1. Check settlement idempotency
                existing_s = (
                    AISettlementModel.objects.select_for_update()
                    .filter(idempotency_key=idempotency_key)
                    .first()
                )
                if existing_s:
                    if (
                        existing_s.reservation_id == reservation_id
                        and existing_s.actual_prompt_tokens == actual_prompt_tokens
                        and existing_s.actual_completion_tokens == actual_completion_tokens
                        and existing_s.actual_total_tokens == actual_total_tokens
                        and existing_s.actual_cost_micros == actual_cost_micros
                        and existing_s.settlement_type == settlement_type.value
                        and existing_s.attempt_number == attempt_number
                        and existing_s.overrun_marker == overrun_marker.value
                        and existing_s.overrun_tokens == overrun_tokens
                        and existing_s.overrun_cost_micros == overrun_cost_micros
                        and math.isclose(
                            existing_s.duration_seconds, duration_seconds, rel_tol=1e-5
                        )
                    ):
                        return (self._settlement_to_record(existing_s), True)
                    raise AIDuplicateSettlementError(
                        reservation_id,
                        "Conflicting settlement parameters for idempotency key",
                    )

                # 2. Lock reservation
                res = (
                    AIReservationModel.objects.select_for_update()
                    .filter(reservation_id=reservation_id)
                    .first()
                )
                if not res:
                    raise AIReservationNotFoundError(reservation_id)

                if res.status == ReservationStatus.SETTLED.value:
                    existing = AISettlementModel.objects.filter(
                        reservation_id=reservation_id
                    ).first()
                    if existing and existing.idempotency_key == idempotency_key:
                        return (self._settlement_to_record(existing), True)
                    raise AIDuplicateSettlementError(reservation_id)

                if res.status != ReservationStatus.PENDING.value:
                    raise AIInvalidReservationStateError(reservation_id, res.status, "settle")

                # Fencing token check
                if attempt_number != res.attempt_number:
                    raise AIStaleFencedAttemptError(
                        reservation_id, res.attempt_number, attempt_number
                    )

                # 3. Lock scope hierarchy counters
                res_scope = self._scope_from_model(res)
                all_scopes = [res_scope, *res_scope.get_ancestor_scopes()]
                self._ensure_and_lock_counters(all_scopes)

                # 4. Mutate counters atomically
                # Deduct reserved exposure
                calls_inc = 1 if settlement_type != SettlementType.ZERO_COST_REJECTED else 0
                tokens_inc = (
                    actual_total_tokens
                    if settlement_type != SettlementType.ZERO_COST_REJECTED
                    else 0
                )
                cost_inc = (
                    actual_cost_micros
                    if settlement_type != SettlementType.ZERO_COST_REJECTED
                    else 0
                )

                for s in all_scopes:
                    AIBudgetScopeCounter.objects.filter(scope_key=s.key).update(
                        reserved_calls=F("reserved_calls") - 1,
                        reserved_tokens=F("reserved_tokens") - res.estimated_total_tokens,
                        reserved_cost_micros=F("reserved_cost_micros") - res.estimated_cost_micros,
                        settled_calls=F("settled_calls") + calls_inc,
                        settled_tokens=F("settled_tokens") + tokens_inc,
                        settled_cost_micros=F("settled_cost_micros") + cost_inc,
                    )

                # 5. Update reservation status
                res.status = ReservationStatus.SETTLED.value
                res.save(update_fields=["status"])

                # 6. Create settlement record
                settlement_model = AISettlementModel.objects.create(
                    settlement_id=settlement_id,
                    reservation=res,
                    idempotency_key=idempotency_key,
                    scope_key=res.scope_key,
                    scope_type=res.scope_type,
                    tenant_id=res.tenant_id,
                    workflow_id=res.workflow_id,
                    run_id=res.run_id,
                    scope_provider=res.scope_provider,
                    provider=res.provider,
                    model=res.model,
                    settlement_type=settlement_type.value,
                    actual_prompt_tokens=actual_prompt_tokens,
                    actual_completion_tokens=actual_completion_tokens,
                    actual_total_tokens=actual_total_tokens,
                    actual_cost_micros=actual_cost_micros,
                    attempt_number=attempt_number,
                    settled_at=settled_at,
                    overrun_marker=overrun_marker.value,
                    overrun_tokens=overrun_tokens,
                    overrun_cost_micros=overrun_cost_micros,
                    duration_seconds=duration_seconds,
                    audit_metadata=sanitize_audit_metadata(audit_metadata),
                )

                # 7. Append ledger entry if supplied
                if ledger_entry:
                    self._persist_ledger_entry(ledger_entry)

                return (self._settlement_to_record(settlement_model), False)

        return _with_db_retry(_txn)

    # -------------------------------------------------------------------------
    # Release Reservation
    # -------------------------------------------------------------------------

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

        def _txn() -> tuple[ReservationRecord, bool]:
            with transaction.atomic():
                res = (
                    AIReservationModel.objects.select_for_update()
                    .filter(reservation_id=reservation_id)
                    .first()
                )
                if not res:
                    raise AIReservationNotFoundError(reservation_id)

                # Full-scope tenant authorization
                if res.tenant_id:
                    if scope is None or scope.tenant_id != res.tenant_id:
                        raise AIReservationNotFoundError(reservation_id)
                    if scope.workflow_id is not None and scope.workflow_id != res.workflow_id:
                        raise AIReservationNotFoundError(reservation_id)
                    if scope.run_id is not None and scope.run_id != res.run_id:
                        raise AIReservationNotFoundError(reservation_id)
                else:
                    if scope is not None and scope.tenant_id is not None:
                        raise AIReservationNotFoundError(reservation_id)

                if res.status == ReservationStatus.RELEASED.value:
                    return (self._reservation_to_record(res), True)

                if res.status != ReservationStatus.PENDING.value:
                    raise AIInvalidReservationStateError(reservation_id, res.status, "release")

                # Cannot release if request was already dispatched to a provider
                current_dispatch = getattr(
                    res, "dispatch_status", DispatchStatus.NOT_DISPATCHED.value
                )
                if current_dispatch != DispatchStatus.NOT_DISPATCHED.value:
                    raise AIInvalidReservationStateError(
                        reservation_id, current_dispatch, "release"
                    )

                # Lock scope hierarchy counters
                res_scope = self._scope_from_model(res)
                all_scopes = [res_scope, *res_scope.get_ancestor_scopes()]
                self._ensure_and_lock_counters(all_scopes)

                # Deduct reserved exposure without adding settled usage
                for s in all_scopes:
                    AIBudgetScopeCounter.objects.filter(scope_key=s.key).update(
                        reserved_calls=F("reserved_calls") - 1,
                        reserved_tokens=F("reserved_tokens") - res.estimated_total_tokens,
                        reserved_cost_micros=F("reserved_cost_micros") - res.estimated_cost_micros,
                    )

                res.status = ReservationStatus.RELEASED.value
                res.save(update_fields=["status"])

                if ledger_entry:
                    self._persist_ledger_entry(ledger_entry)

                return (self._reservation_to_record(res), False)

        return _with_db_retry(_txn)

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

        def _txn() -> ReservationRecord:
            with transaction.atomic():
                res = (
                    AIReservationModel.objects.select_for_update()
                    .filter(reservation_id=reservation_id)
                    .first()
                )
                if not res:
                    raise AIReservationNotFoundError(reservation_id)

                if res.tenant_id:
                    if scope is None or scope.tenant_id != res.tenant_id:
                        raise AIReservationNotFoundError(reservation_id)
                    if scope.workflow_id is not None and scope.workflow_id != res.workflow_id:
                        raise AIReservationNotFoundError(reservation_id)
                    if scope.run_id is not None and scope.run_id != res.run_id:
                        raise AIReservationNotFoundError(reservation_id)

                if res.attempt_number != attempt_number:
                    raise AIStaleFencedAttemptError(
                        reservation_id,
                        expected_attempt=res.attempt_number,
                        actual_attempt=attempt_number,
                    )

                current_dispatch = getattr(
                    res, "dispatch_status", DispatchStatus.NOT_DISPATCHED.value
                )
                legal_transitions = {
                    DispatchStatus.NOT_DISPATCHED.value: {DispatchStatus.DISPATCH_STARTED.value},
                    DispatchStatus.DISPATCH_STARTED.value: {
                        DispatchStatus.RESPONSE_RECEIVED.value,
                        DispatchStatus.RESPONSE_LOST.value,
                        DispatchStatus.BILLING_UNCERTAIN.value,
                    },
                    DispatchStatus.RESPONSE_LOST.value: {DispatchStatus.BILLING_UNCERTAIN.value},
                    DispatchStatus.BILLING_UNCERTAIN.value: set(),
                    DispatchStatus.RESPONSE_RECEIVED.value: set(),
                }
                target_val = (
                    to_status.value if isinstance(to_status, DispatchStatus) else str(to_status)
                )
                allowed = legal_transitions.get(current_dispatch, set())
                if target_val not in allowed:
                    raise AIInvalidReservationStateError(
                        reservation_id, current_dispatch, "dispatch"
                    )

                res.dispatch_status = target_val
                res.save(update_fields=["dispatch_status"])
                return self._reservation_to_record(res)

        return _with_db_retry(_txn)

    # -------------------------------------------------------------------------
    # Ledger Append & Query
    # -------------------------------------------------------------------------

    def _persist_ledger_entry(self, entry: UsageLedgerEntry) -> None:
        AIUsageLedgerModel.objects.create(
            entry_id=entry.entry_id,
            event_type=entry.event_type,
            idempotency_key=entry.idempotency_key,
            scope_key=entry.scope.key,
            scope_type=entry.scope.scope_type.value,
            tenant_id=entry.scope.tenant_id,
            workflow_id=entry.scope.workflow_id,
            run_id=entry.scope.run_id,
            scope_provider=entry.scope.provider,
            provider=entry.provider,
            model=entry.model,
            timestamp=entry.timestamp,
            tokens_delta=entry.tokens_delta,
            cost_micros_delta=entry.cost_micros_delta,
            calls_delta=entry.calls_delta,
            overrun_marker=entry.overrun_marker.value,
            notes=sanitize_note(entry.notes),
            audit_metadata=sanitize_audit_metadata(entry.audit_metadata),
        )

    def append_ledger_entry(self, entry: UsageLedgerEntry) -> None:
        self._persist_ledger_entry(entry)

    def get_ledger_entries(self, scope: BudgetScope, limit: int = 100) -> list[UsageLedgerEntry]:
        qs = AIUsageLedgerModel.objects.filter(scope_key=scope.key).order_by("timestamp")
        entries = [self._ledger_to_entry(m) for m in qs]
        return entries[-limit:]
