"""Budget Policy Service and Orchestrator.

Enforces:
1. Deny-by-default hosted provider access across scope hierarchy.
2. Cross-scope tenant/workflow/run budget bounds and ancestor limits.
3. Atomic reservation under lock (eliminating check-then-write race).
4. Distinguishes known actual usage, estimated exposure, and zero-cost pre-dispatch rejections.
5. Truthful overrun recording when actual token consumption exceeds bounds.
6. Pluggable clock for deterministic testing (no implicit wall-clock time).
7. Cross-tenant isolation and scoped idempotency.
8. Privacy: allowlisted bounded metadata fields, sanitized ledger notes, no raw prompts or secret keys stored.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from flowforge.ai.governance.exceptions import (
    AIBudgetExhaustedError,
    AIGovernanceError,
    AIProviderPolicyDeniedError,
)
from flowforge.ai.governance.models import (
    BudgetScope,
    OverrunMarker,
    ReservationRecord,
    SettlementRecord,
    SettlementType,
    UsageLedgerEntry,
)
from flowforge.ai.governance.repository import BudgetRepository


class AIBudgetPolicyService:
    """Core domain service orchestrating AI budget enforcement, reservations, and settlements."""

    def __init__(
        self,
        repository: BudgetRepository,
        clock: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._repo = repository
        self._clock = clock or (lambda: datetime.datetime.now(datetime.UTC))

    def _now(self) -> datetime.datetime:
        ts = self._clock()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.UTC)
        return ts

    def evaluate_provider_policy(self, scope: BudgetScope, provider: str) -> None:
        """Check provider authorization against budget settings across all hierarchy levels.

        Deny-by-default hosted policy:
        1. If explicitly denied at ANY hierarchy level -> denied immediately.
        2. Safe local/mock providers ('mock', 'ollama', 'test') are allowed unless explicitly denied.
        3. External hosted providers must be explicitly allowed by at least one scope's allowed_providers.
        """
        scopes_to_check = [scope, *scope.get_ancestor_scopes()]
        explicitly_allowed = False
        default_safe = ("mock", "ollama", "test")

        for s in scopes_to_check:
            settings = self._repo.get_budget_settings(s)
            if settings is not None:
                if provider in settings.denied_providers:
                    raise AIProviderPolicyDeniedError(
                        provider,
                        reason="provider_explicitly_denied",
                    )
                if provider in settings.allowed_providers:
                    explicitly_allowed = True

        if provider in default_safe:
            return

        if not explicitly_allowed:
            raise AIProviderPolicyDeniedError(
                provider,
                reason="provider_denied_by_default",
            )

    def reserve(
        self,
        *,
        scope: BudgetScope,
        provider: str,
        model: str,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
        estimated_cost_micros: int = 0,
        idempotency_key: str | None = None,
        attempt_number: int = 1,
        ttl_seconds: float | None = None,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> ReservationRecord:
        """Evaluate provider policy and atomically reserve AI capacity under repository lock."""
        if (
            isinstance(estimated_prompt_tokens, bool)
            or not isinstance(estimated_prompt_tokens, int)
            or estimated_prompt_tokens < 0
        ):
            raise AIGovernanceError("estimated_prompt_tokens must be a non-negative integer")
        if (
            isinstance(estimated_completion_tokens, bool)
            or not isinstance(estimated_completion_tokens, int)
            or estimated_completion_tokens < 0
        ):
            raise AIGovernanceError("estimated_completion_tokens must be a non-negative integer")
        if (
            isinstance(estimated_cost_micros, bool)
            or not isinstance(estimated_cost_micros, int)
            or estimated_cost_micros < 0
        ):
            raise AIGovernanceError("estimated_cost_micros must be a non-negative integer")

        estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens

        # 1. Deny-by-default provider policy check across hierarchy
        self.evaluate_provider_policy(scope, provider)

        now = self._now()
        expires_at = now + datetime.timedelta(seconds=ttl_seconds) if ttl_seconds else None
        res_id = str(uuid.uuid4())
        idem_key = idempotency_key or str(uuid.uuid4())

        # Construct ledger entry to append atomically with reservation
        ledger_entry = UsageLedgerEntry(
            entry_id=str(uuid.uuid4()),
            event_type="reservation",
            idempotency_key=idem_key,
            scope=scope,
            provider=provider,
            model=model,
            timestamp=now,
            tokens_delta=estimated_total_tokens,
            cost_micros_delta=estimated_cost_micros,
            calls_delta=1,
            notes=f"Reserved {estimated_total_tokens} estimated tokens",
            audit_metadata=audit_metadata or {},
        )

        ancestors = scope.get_ancestor_scopes()
        record, _ = self._repo.reserve_atomic(
            reservation_id=res_id,
            idempotency_key=idem_key,
            scope=scope,
            ancestor_scopes=ancestors,
            provider=provider,
            model=model,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
            estimated_total_tokens=estimated_total_tokens,
            estimated_cost_micros=estimated_cost_micros,
            attempt_number=attempt_number,
            created_at=now,
            expires_at=expires_at,
            audit_metadata=audit_metadata,
            ledger_entry=ledger_entry,
        )
        return record

    def settle(
        self,
        *,
        reservation_id: str,
        scope: BudgetScope | None = None,
        actual_prompt_tokens: int | None = None,
        actual_completion_tokens: int | None = None,
        actual_cost_micros: int | None = None,
        settlement_type: SettlementType = SettlementType.EXACT,
        attempt_number: int = 1,
        duration_seconds: float = 0.0,
        idempotency_key: str | None = None,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> SettlementRecord:
        """Atomically settle a completed or failed reservation."""
        if scope is not None:
            res = self._repo.get_reservation_by_id(reservation_id, scope=scope)
        else:
            # Internal privileged lookup to find reservation and assert its scope
            res = self._repo.get_reservation_by_id(reservation_id, internal_privileged=True)
        if not res:
            raise AIGovernanceError("Reservation not found")

        now = self._now()
        idem_key = idempotency_key or f"settle-{reservation_id}-{attempt_number}"

        # Reconcile token & cost amounts based on settlement type
        if settlement_type == SettlementType.ZERO_COST_REJECTED:
            final_prompt = 0
            final_completion = 0
            final_total = 0
            final_cost = 0
        elif settlement_type in (SettlementType.ESTIMATED_EXPOSURE, SettlementType.UNKNOWN_USAGE):
            final_prompt = (
                actual_prompt_tokens
                if actual_prompt_tokens is not None
                else res.estimated_prompt_tokens
            )
            final_completion = (
                actual_completion_tokens
                if actual_completion_tokens is not None
                else res.estimated_completion_tokens
            )
            final_total = final_prompt + final_completion
            final_cost = (
                actual_cost_micros if actual_cost_micros is not None else res.estimated_cost_micros
            )
        else:
            # EXACT
            if actual_prompt_tokens is None or actual_completion_tokens is None:
                raise AIGovernanceError(
                    "EXACT settlement requires actual_prompt_tokens and actual_completion_tokens"
                )
            if (
                isinstance(actual_prompt_tokens, bool)
                or not isinstance(actual_prompt_tokens, int)
                or actual_prompt_tokens < 0
            ):
                raise AIGovernanceError("actual_prompt_tokens must be non-negative integer")
            if (
                isinstance(actual_completion_tokens, bool)
                or not isinstance(actual_completion_tokens, int)
                or actual_completion_tokens < 0
            ):
                raise AIGovernanceError("actual_completion_tokens must be non-negative integer")
            final_prompt = actual_prompt_tokens
            final_completion = actual_completion_tokens
            final_total = final_prompt + final_completion
            final_cost = (
                actual_cost_micros
                if actual_cost_micros is not None
                else int(
                    final_total * (res.estimated_cost_micros / max(1, res.estimated_total_tokens))
                )
            )

        # Detect and mark overruns truthfully
        settings = self._repo.get_budget_settings(res.scope)
        overrun_marker = OverrunMarker.NONE
        overrun_tokens = 0
        overrun_cost = 0

        if settings:
            settled_calls, settled_tokens, settled_cost = self._repo.get_aggregate_usage(res.scope)
            projected_tokens = settled_tokens + final_total
            projected_cost = settled_cost + final_cost

            if (
                settings.max_total_tokens is not None
                and projected_tokens > settings.max_total_tokens
            ):
                overrun_marker = OverrunMarker.TOKENS_OVERRUN
                overrun_tokens = projected_tokens - settings.max_total_tokens

            if (
                settings.max_estimated_cost_micros is not None
                and projected_cost > settings.max_estimated_cost_micros
            ):
                overrun_marker = OverrunMarker.COST_OVERRUN
                overrun_cost = projected_cost - settings.max_estimated_cost_micros

            if (
                settings.max_duration_seconds is not None
                and duration_seconds > settings.max_duration_seconds
            ):
                overrun_marker = OverrunMarker.TIME_OVERRUN

            # If overrun occurred and allow_overrun is False, raise AIBudgetExhaustedError after recording truthfully
            if overrun_marker != OverrunMarker.NONE and not settings.allow_overrun:
                settlement_id = str(uuid.uuid4())
                ledger_entry = UsageLedgerEntry(
                    entry_id=str(uuid.uuid4()),
                    event_type="settlement_overrun",
                    idempotency_key=idem_key,
                    scope=res.scope,
                    provider=res.provider,
                    model=res.model,
                    timestamp=now,
                    tokens_delta=final_total,
                    cost_micros_delta=final_cost,
                    calls_delta=1 if settlement_type != SettlementType.ZERO_COST_REJECTED else 0,
                    overrun_marker=overrun_marker,
                    notes=f"Overrun: {overrun_tokens} tokens, {overrun_cost} micros",
                    audit_metadata=audit_metadata or {},
                )
                self._repo.settle_reservation(
                    settlement_id=settlement_id,
                    reservation_id=reservation_id,
                    idempotency_key=idem_key,
                    settlement_type=settlement_type,
                    actual_prompt_tokens=final_prompt,
                    actual_completion_tokens=final_completion,
                    actual_total_tokens=final_total,
                    actual_cost_micros=final_cost,
                    attempt_number=attempt_number,
                    settled_at=now,
                    overrun_marker=overrun_marker,
                    overrun_tokens=overrun_tokens,
                    overrun_cost_micros=overrun_cost,
                    duration_seconds=duration_seconds,
                    audit_metadata=audit_metadata,
                    ledger_entry=ledger_entry,
                )
                raise AIBudgetExhaustedError(
                    f"Settlement resulted in budget overrun ({overrun_marker.value})",
                    limit_name="max_total_tokens",
                    requested_value=final_total,
                    current_value=projected_tokens,
                    limit_value=settings.max_total_tokens or 0,
                    scope_key=res.scope.key,
                )

        settlement_id = str(uuid.uuid4())
        ledger_entry = UsageLedgerEntry(
            entry_id=str(uuid.uuid4()),
            event_type="settlement",
            idempotency_key=idem_key,
            scope=res.scope,
            provider=res.provider,
            model=res.model,
            timestamp=now,
            tokens_delta=final_total,
            cost_micros_delta=final_cost,
            calls_delta=1 if settlement_type != SettlementType.ZERO_COST_REJECTED else 0,
            overrun_marker=overrun_marker,
            notes=f"Settled {final_total} tokens ({settlement_type.value})",
            audit_metadata=audit_metadata or {},
        )
        record, _ = self._repo.settle_reservation(
            settlement_id=settlement_id,
            reservation_id=reservation_id,
            idempotency_key=idem_key,
            settlement_type=settlement_type,
            actual_prompt_tokens=final_prompt,
            actual_completion_tokens=final_completion,
            actual_total_tokens=final_total,
            actual_cost_micros=final_cost,
            attempt_number=attempt_number,
            settled_at=now,
            overrun_marker=overrun_marker,
            overrun_tokens=overrun_tokens,
            overrun_cost_micros=overrun_cost,
            duration_seconds=duration_seconds,
            audit_metadata=audit_metadata,
            ledger_entry=ledger_entry,
        )
        return record

    def release(
        self,
        *,
        reservation_id: str,
        scope: BudgetScope | None = None,
        idempotency_key: str | None = None,
        reason: str = "released_without_usage",
    ) -> ReservationRecord:
        """Release reservation without incurring cost or token usage."""
        now = self._now()
        if scope is not None:
            res = self._repo.get_reservation_by_id(reservation_id, scope=scope)
        else:
            res = self._repo.get_reservation_by_id(reservation_id, internal_privileged=True)
        if not res:
            raise AIGovernanceError("Reservation not found")

        idem_key = idempotency_key or f"release-{reservation_id}"
        ledger_entry = UsageLedgerEntry(
            entry_id=str(uuid.uuid4()),
            event_type="release",
            idempotency_key=idem_key,
            scope=res.scope,
            provider=res.provider,
            model=res.model,
            timestamp=now,
            tokens_delta=0,
            cost_micros_delta=0,
            calls_delta=0,
            notes=f"Released reservation: {reason}",
            audit_metadata={},
        )
        record, _ = self._repo.release_reservation(
            reservation_id=reservation_id,
            idempotency_key=idem_key,
            scope=res.scope,
            released_at=now,
            reason=reason,
            ledger_entry=ledger_entry,
        )
        return record
