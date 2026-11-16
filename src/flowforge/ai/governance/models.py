"""Immutable data models and value objects for AI governance and budget management.

Strict privacy: no raw prompts, secret API keys, provider tokens, or raw LLM responses
are held in these records. All timestamps are UTC aware.
"""

from __future__ import annotations

import datetime
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from flowforge.ai.governance.exceptions import AIGovernanceError

SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

ALLOWED_AUDIT_METADATA_KEYS = frozenset(
    {
        "job_id",
        "step_name",
        "workflow_id",
        "run_id",
        "tenant_id",
        "user_id",
        "correlation_id",
        "attempt",
        "provider",
        "model",
        "status",
        "reason",
        "client_version",
    }
)

VALID_STATUS_STRINGS = frozenset({"pending", "settled", "released", "expired", "failed", "success"})


def validate_identifier(val: str, field_name: str) -> None:
    """Validates that an identifier (provider, model, idempotency key) is bounded and contains only safe characters."""
    if not isinstance(val, str) or not val:
        raise AIGovernanceError(f"{field_name} must be a non-empty string")
    if len(val) > 128:
        raise AIGovernanceError(f"{field_name} length must not exceed 128 characters")
    # Disallow control characters, path traversal, null bytes
    if any(c in val for c in ("\x00", "/", "\\", "\n", "\r", "\t", " ", ";")):
        raise AIGovernanceError(f"{field_name} contains invalid characters")


def sanitize_audit_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Sanitizes audit metadata by allowlisting permitted bounded keys and casting values to strictly typed strings.

    Drops arbitrary unknown strings to prevent storing unknown-format printable secrets.
    """
    if not metadata:
        return {}
    clean: dict[str, str] = {}
    for k, v in metadata.items():
        key_str = str(k).lower().strip()
        if key_str in ALLOWED_AUDIT_METADATA_KEYS and v is not None:
            val_str = str(v).strip()
            # Enforce strict typing / format validation per field:
            if key_str in ("job_id", "workflow_id", "run_id", "tenant_id", "user_id"):
                if SAFE_IDENTIFIER_PATTERN.match(val_str) or UUID_PATTERN.match(val_str):
                    clean[key_str] = val_str[:64]
            elif key_str == "correlation_id":
                # Must be valid UUID or alphanumeric hex/dash identifier
                if (
                    UUID_PATTERN.match(val_str) or SAFE_IDENTIFIER_PATTERN.match(val_str)
                ) and not any(val_str.startswith(p) for p in ("ghp_", "sk-", "eyJ", "gho_")):
                    clean[key_str] = val_str[:64]
            elif key_str in ("provider", "model", "step_name", "client_version"):
                if SAFE_IDENTIFIER_PATTERN.match(val_str):
                    clean[key_str] = val_str[:64]
            elif key_str == "status":
                if val_str.lower() in VALID_STATUS_STRINGS:
                    clean[key_str] = val_str.lower()
            elif key_str == "reason":
                # Fixed reason strings only
                if val_str in (
                    "user_cancelled",
                    "released_without_usage",
                    "timeout",
                    "policy_denied",
                    "completed",
                ):
                    clean[key_str] = val_str
            elif key_str == "attempt" and isinstance(v, (int, str)) and str(v).isdigit():
                clean[key_str] = str(v)[:8]
    return clean


def sanitize_note(note: str) -> str:
    """Sanitizes ledger notes to guarantee no raw secrets, keys, or arbitrary caller data are retained."""
    if not note:
        return ""
    # Truncate and strip sensitive prefixes/substrings
    lower_note = note.lower()
    for sensitive_sub in ("bearer ", "sk-", "secret", "token", "password", "prompt", "ghp_"):
        if sensitive_sub in lower_note:
            return "Sanitized note (sensitive content removed)"
    return note[:256]


class BudgetScopeType(StrEnum):
    """Scope boundaries for budget allocation and accounting."""

    GLOBAL = "global"
    TENANT = "tenant"
    WORKFLOW = "workflow"
    RUN = "run"
    PROVIDER = "provider"


@dataclass(frozen=True)
class BudgetScope:
    """Explicit, bounded hierarchical budget scope key."""

    scope_type: BudgetScopeType
    tenant_id: str | None = None
    workflow_id: str | None = None
    run_id: str | None = None
    provider: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope_type, BudgetScopeType):
            try:
                object.__setattr__(self, "scope_type", BudgetScopeType(self.scope_type))
            except ValueError as err:
                raise AIGovernanceError("Invalid scope_type specified") from err

        # Validation rules:
        if self.scope_type == BudgetScopeType.TENANT and not self.tenant_id:
            raise AIGovernanceError("tenant_id is required for TENANT budget scope")
        if self.scope_type == BudgetScopeType.WORKFLOW and (
            not self.tenant_id or not self.workflow_id
        ):
            raise AIGovernanceError(
                "tenant_id and workflow_id are required for WORKFLOW budget scope"
            )
        if self.scope_type == BudgetScopeType.RUN and (not self.tenant_id or not self.run_id):
            raise AIGovernanceError("tenant_id and run_id are required for RUN budget scope")
        if self.scope_type == BudgetScopeType.PROVIDER and not self.provider:
            raise AIGovernanceError("provider is required for PROVIDER budget scope")

        for name, ident in [
            ("tenant_id", self.tenant_id),
            ("workflow_id", self.workflow_id),
            ("run_id", self.run_id),
            ("provider", self.provider),
        ]:
            if ident is not None:
                validate_identifier(ident, name)

    @property
    def key(self) -> str:
        """Deterministic canonical string representation for isolation and keying."""
        parts = [f"type={self.scope_type.value}"]
        if self.tenant_id is not None:
            parts.append(f"tenant={self.tenant_id}")
        if self.workflow_id is not None:
            parts.append(f"workflow={self.workflow_id}")
        if self.run_id is not None:
            parts.append(f"run={self.run_id}")
        if self.provider is not None:
            parts.append(f"provider={self.provider}")
        return ":".join(parts)

    def get_ancestor_scopes(self) -> list[BudgetScope]:
        """Returns the list of ancestor scopes in top-down order (GLOBAL -> TENANT -> WORKFLOW -> RUN/PROVIDER)."""
        ancestors: list[BudgetScope] = [BudgetScope(scope_type=BudgetScopeType.GLOBAL)]
        if self.tenant_id:
            tenant_s = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id=self.tenant_id)
            if self.scope_type != BudgetScopeType.TENANT:
                ancestors.append(tenant_s)

            if self.workflow_id and self.scope_type not in (
                BudgetScopeType.TENANT,
                BudgetScopeType.WORKFLOW,
            ):
                ancestors.append(
                    BudgetScope(
                        scope_type=BudgetScopeType.WORKFLOW,
                        tenant_id=self.tenant_id,
                        workflow_id=self.workflow_id,
                    )
                )
        return ancestors


@dataclass(frozen=True)
class BudgetSettings:
    """Versionable, bounded budget configuration limits."""

    scope: BudgetScope
    version: int = 1
    max_calls: int | None = None
    max_total_tokens: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_estimated_cost_micros: int | None = None  # in micro-USD (1 USD = 1,000,000 micros)
    max_duration_seconds: float | None = None
    allowed_providers: tuple[str, ...] = ("mock", "ollama")  # Deny-by-default for hosted external
    denied_providers: tuple[str, ...] = ()
    allow_overrun: bool = (
        False  # If True, records overrun rather than raising immediately on completion
    )

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise AIGovernanceError("version must be a positive integer, not bool")

        for name, val in [
            ("max_calls", self.max_calls),
            ("max_total_tokens", self.max_total_tokens),
            ("max_prompt_tokens", self.max_prompt_tokens),
            ("max_completion_tokens", self.max_completion_tokens),
            ("max_estimated_cost_micros", self.max_estimated_cost_micros),
        ]:
            if val is not None and (isinstance(val, bool) or not isinstance(val, int) or val < 0):
                raise AIGovernanceError(f"{name} must be a non-negative integer or None")

        if self.max_duration_seconds is not None and (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or self.max_duration_seconds < 0
            or math.isnan(self.max_duration_seconds)
            or math.isinf(self.max_duration_seconds)
        ):
            raise AIGovernanceError(
                "max_duration_seconds must be a non-negative finite float or None"
            )

        # Validate allowed_providers & denied_providers must be tuples/lists of strings, NOT a bare string
        if isinstance(self.allowed_providers, str):
            raise AIGovernanceError(
                "allowed_providers must be a tuple or list of strings, not a bare string"
            )
        if isinstance(self.denied_providers, str):
            raise AIGovernanceError(
                "denied_providers must be a tuple or list of strings, not a bare string"
            )

        allowed_tuple = tuple(self.allowed_providers)
        denied_tuple = tuple(self.denied_providers)

        for p in allowed_tuple:
            validate_identifier(p, "allowed_provider")
        for p in denied_tuple:
            validate_identifier(p, "denied_provider")

        object.__setattr__(self, "allowed_providers", allowed_tuple)
        object.__setattr__(self, "denied_providers", denied_tuple)


class ReservationStatus(StrEnum):
    """Lifecycle state of an AI budget reservation."""

    PENDING = "pending"
    SETTLED = "settled"
    RELEASED = "released"
    EXPIRED = "expired"


class SettlementType(StrEnum):
    """The outcome of a completed or reconciled AI operation."""

    EXACT = "exact"  # Provider returned verified actual tokens
    ESTIMATED_EXPOSURE = (
        "estimated_exposure"  # Dispatched / maybe-sent; provider response lost/failed
    )
    ZERO_COST_REJECTED = "zero_cost_rejected"  # Rejected locally before network dispatch (cost 0)
    UNKNOWN_USAGE = (
        "unknown_usage"  # Unknown actual usage; preserved estimated exposure conservatively
    )


class OverrunMarker(StrEnum):
    """Truthful marker documenting whether and how an execution exceeded limits."""

    NONE = "none"
    CALLS_OVERRUN = "calls_overrun"
    TOKENS_OVERRUN = "tokens_overrun"
    COST_OVERRUN = "cost_overrun"
    TIME_OVERRUN = "time_overrun"


@dataclass(frozen=True)
class ReservationRecord:
    """Immutable record of an authorized and reserved AI call allowance."""

    reservation_id: str
    idempotency_key: str
    scope: BudgetScope
    provider: str
    model: str
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    estimated_total_tokens: int
    estimated_cost_micros: int
    attempt_number: int
    status: ReservationStatus
    created_at: datetime.datetime
    expires_at: datetime.datetime | None = None
    audit_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reservation_id:
            raise AIGovernanceError("reservation_id cannot be empty")
        if not self.idempotency_key:
            raise AIGovernanceError("idempotency_key cannot be empty")
        validate_identifier(self.provider, "provider")
        validate_identifier(self.model, "model")
        if self.attempt_number < 1:
            raise AIGovernanceError("attempt_number must be >= 1")
        if self.created_at.tzinfo is None:
            raise AIGovernanceError("created_at must be timezone-aware (UTC)")
        object.__setattr__(self, "audit_metadata", sanitize_audit_metadata(self.audit_metadata))


@dataclass(frozen=True)
class SettlementRecord:
    """Immutable record of a finalized reservation settlement."""

    settlement_id: str
    reservation_id: str
    idempotency_key: str
    scope: BudgetScope
    provider: str
    model: str
    settlement_type: SettlementType
    actual_prompt_tokens: int
    actual_completion_tokens: int
    actual_total_tokens: int
    actual_cost_micros: int
    attempt_number: int
    settled_at: datetime.datetime
    overrun_marker: OverrunMarker = OverrunMarker.NONE
    overrun_tokens: int = 0
    overrun_cost_micros: int = 0
    duration_seconds: float = 0.0
    audit_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.settlement_id:
            raise AIGovernanceError("settlement_id cannot be empty")
        if not self.reservation_id:
            raise AIGovernanceError("reservation_id cannot be empty")
        validate_identifier(self.provider, "provider")
        validate_identifier(self.model, "model")
        if self.settled_at.tzinfo is None:
            raise AIGovernanceError("settled_at must be timezone-aware (UTC)")
        for name, val in [
            ("actual_prompt_tokens", self.actual_prompt_tokens),
            ("actual_completion_tokens", self.actual_completion_tokens),
            ("actual_total_tokens", self.actual_total_tokens),
            ("actual_cost_micros", self.actual_cost_micros),
            ("overrun_tokens", self.overrun_tokens),
            ("overrun_cost_micros", self.overrun_cost_micros),
        ]:
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise AIGovernanceError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or self.duration_seconds < 0
            or math.isnan(self.duration_seconds)
            or math.isinf(self.duration_seconds)
        ):
            raise AIGovernanceError("duration_seconds must be a non-negative finite float")
        object.__setattr__(self, "audit_metadata", sanitize_audit_metadata(self.audit_metadata))


@dataclass(frozen=True)
class UsageLedgerEntry:
    """Single immutable audit log entry in the AI usage ledger."""

    entry_id: str
    event_type: str  # "reservation", "settlement", "release", "denial"
    idempotency_key: str
    scope: BudgetScope
    provider: str
    model: str
    timestamp: datetime.datetime
    tokens_delta: int
    cost_micros_delta: int
    calls_delta: int
    overrun_marker: OverrunMarker = OverrunMarker.NONE
    notes: str = ""
    audit_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise AIGovernanceError("timestamp must be timezone-aware (UTC)")
        object.__setattr__(self, "audit_metadata", sanitize_audit_metadata(self.audit_metadata))
        object.__setattr__(self, "notes", sanitize_note(self.notes))
