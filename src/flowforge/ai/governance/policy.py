"""Budget Policy Service and Orchestrator.

Enforces:
1. Deny-by-default hosted provider access.
2. Tenant/workflow/run budget bounds (calls, tokens, cost, duration).
3. Distinguishes known actual usage, estimated exposure, and zero-cost pre-dispatch rejections.
4. Truthful overrun recording when actual token consumption exceeds bounds.
5. Pluggable clock for deterministic testing (no implicit wall-clock time).
6. Cross-tenant isolation.
7. Privacy: no raw prompts, secret keys, or provider responses stored.
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
    BudgetScopeType,
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
        # Injectable clock defaults to UTC datetime.now
        self._clock = clock or (lambda: datetime.datetime.now(datetime.UTC))

    def _now(self) -> datetime.datetime:
        ts = self._clock()
        if ts.tzinfo is None:
            # Ensure timezone awareness
            ts = ts.replace(tzinfo=datetime.UTC)
        return ts

    def evaluate_provider_policy(self, scope: BudgetScope, provider: str) -> None:
        """Check provider authorization against budget settings.

        Deny-by-default hosted policy:
        If settings are defined, provider must be explicitly in allowed_providers
        and must not be in denied_providers.
        """
        settings = self._repo.get_budget_settings(scope)
        if settings is None:
            # Check global scope fallback
            global_scope = BudgetScope(scope_type=BudgetScopeType.GLOBAL)
            settings = self._repo.get_budget_settings(global_scope)

        if settings is not None:
            if provider in settings.denied_providers:
                raise AIProviderPolicyDeniedError(
                    provider, f"Provider '{provider}' is explicitly denied in budget settings"
                )
            if provider not in settings.allowed_providers:
                raise AIProviderPolicyDeniedError(
                    provider,
                    f"Provider '{provider}' is not in allowed providers {settings.allowed_providers}",
                )
        else:
            # Deny-by-default for hosted providers when no settings defined
            # Default safe allowed: mock, ollama, test
            default_safe = ("mock", "ollama", "test")
            if provider not in default_safe:
                raise AIProviderPolicyDeniedError(
                    provider,
                    f"Hosted provider '{provider}' denied by default (no explicit budget policy authorizes it)",
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
        """Evaluate budget limits and atomically reserve AI capacity."""
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

        # Check existing reservation by idempotency key
        if idempotency_key:
            existing = self._repo.get_reservation_by_idempotency_key(idempotency_key)
            if existing:
                return existing

        # 1. Deny-by-default provider policy check
        self.evaluate_provider_policy(scope, provider)

        # 2. Check budget limits across hierarchy (tenant, workflow, run)
        settings = self._repo.get_budget_settings(scope)
        if settings:
            settled_calls, settled_tokens, settled_cost = self._repo.get_aggregate_usage(scope)
            res_calls, res_tokens, res_cost = self._repo.get_active_reserved_usage(scope)

            # Check calls
            if (
                settings.max_calls is not None
                and settled_calls + res_calls + 1 > settings.max_calls
            ):
                raise AIBudgetExhaustedError(
                    f"Scope '{scope.key}' exceeded max_calls ({settings.max_calls})",
                    limit_name="max_calls",
                    requested_value=1,
                    current_value=settled_calls + res_calls,
                    limit_value=settings.max_calls,
                    scope_key=scope.key,
                )

            # Check total tokens
            if (
                settings.max_total_tokens is not None
                and settled_tokens + res_tokens + estimated_total_tokens > settings.max_total_tokens
            ):
                raise AIBudgetExhaustedError(
                    f"Scope '{scope.key}' exceeded max_total_tokens ({settings.max_total_tokens}) by estimated request ({estimated_total_tokens})",
                    limit_name="max_total_tokens",
                    requested_value=estimated_total_tokens,
                    current_value=settled_tokens + res_tokens,
                    limit_value=settings.max_total_tokens,
                    scope_key=scope.key,
                )

            # Check cost
            if (
                settings.max_estimated_cost_micros is not None
                and settled_cost + res_cost + estimated_cost_micros
                > settings.max_estimated_cost_micros
            ):
                raise AIBudgetExhaustedError(
                    f"Scope '{scope.key}' exceeded max_estimated_cost_micros ({settings.max_estimated_cost_micros}) by estimated cost ({estimated_cost_micros})",
                    limit_name="max_estimated_cost_micros",
                    requested_value=estimated_cost_micros,
                    current_value=settled_cost + res_cost,
                    limit_value=settings.max_estimated_cost_micros,
                    scope_key=scope.key,
                )

        now = self._now()
        expires_at = now + datetime.timedelta(seconds=ttl_seconds) if ttl_seconds else None
        res_id = str(uuid.uuid4())
        idem_key = idempotency_key or str(uuid.uuid4())

        record = self._repo.create_reservation(
            reservation_id=res_id,
            idempotency_key=idem_key,
            scope=scope,
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
        )

        # Audit ledger
        self._repo.append_ledger_entry(
            UsageLedgerEntry(
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
        )
        return record

    def settle(
        self,
        *,
        reservation_id: str,
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
        res = self._repo.get_reservation_by_id(reservation_id)
        if not res:
            raise AIGovernanceError(f"Reservation '{reservation_id}' not found")

        # Check existing settlement by reservation or idempotency
        if idempotency_key:
            existing = self._repo.get_settlement_by_idempotency_key(idempotency_key)
            if existing:
                return existing

        now = self._now()
        idem_key = idempotency_key or f"settle-{reservation_id}-{attempt_number}"

        # Reconcile token & cost amounts based on settlement type
        if settlement_type == SettlementType.ZERO_COST_REJECTED:
            # Pre-dispatch rejection: 0 tokens, 0 cost
            final_prompt = 0
            final_completion = 0
            final_total = 0
            final_cost = 0
        elif settlement_type in (SettlementType.ESTIMATED_EXPOSURE, SettlementType.UNKNOWN_USAGE):
            # Maybe-sent or lost response: retain estimated exposure conservatively
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
                # Still settle in repository to release pending reservation and record actual exposure!
                settlement_id = str(uuid.uuid4())
                record = self._repo.settle_reservation(
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
                )
                self._repo.append_ledger_entry(
                    UsageLedgerEntry(
                        entry_id=str(uuid.uuid4()),
                        event_type="settlement_overrun",
                        idempotency_key=idem_key,
                        scope=res.scope,
                        provider=res.provider,
                        model=res.model,
                        timestamp=now,
                        tokens_delta=final_total,
                        cost_micros_delta=final_cost,
                        calls_delta=1
                        if settlement_type != SettlementType.ZERO_COST_REJECTED
                        else 0,
                        overrun_marker=overrun_marker,
                        notes=f"Overrun: {overrun_tokens} tokens, {overrun_cost} micros",
                        audit_metadata=audit_metadata or {},
                    )
                )
                raise AIBudgetExhaustedError(
                    f"Settlement resulted in budget overrun ({overrun_marker.value}): "
                    f"{overrun_tokens} tokens over limit",
                    limit_name="max_total_tokens",
                    requested_value=final_total,
                    current_value=projected_tokens,
                    limit_value=settings.max_total_tokens or 0,
                    scope_key=res.scope.key,
                )

        settlement_id = str(uuid.uuid4())
        record = self._repo.settle_reservation(
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
        )

        self._repo.append_ledger_entry(
            UsageLedgerEntry(
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
        )
        return record

    def release(
        self,
        *,
        reservation_id: str,
        idempotency_key: str | None = None,
        reason: str = "Released without usage",
    ) -> ReservationRecord:
        """Release reservation without incurring cost or token usage."""
        now = self._now()
        res = self._repo.get_reservation_by_id(reservation_id)
        if not res:
            raise AIGovernanceError(f"Reservation '{reservation_id}' not found")

        idem_key = idempotency_key or f"release-{reservation_id}"
        record = self._repo.release_reservation(
            reservation_id=reservation_id,
            idempotency_key=idem_key,
            released_at=now,
            reason=reason,
        )

        self._repo.append_ledger_entry(
            UsageLedgerEntry(
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
        )
        return record
