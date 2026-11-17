"""PostgreSQL Django models for AI Budget Governance persistence.

Provides durable storage for:
1. Materialized scope usage counters and lock target (`AIBudgetScopeCounter`).
2. Scope budget limits and provider policies (`AIBudgetSettingsModel`).
3. Authorized reservations with attempt fencing (`AIReservationModel`).
4. Reconciled settlements with actual metrics (`AISettlementModel`).
5. Append-only immutable audit ledger (`AIUsageLedgerModel`).

Strict privacy: No raw prompts, secrets, completions, or unallowlisted metadata stored.
"""

from __future__ import annotations

from django.db import models


class AIBudgetScopeCounter(models.Model):
    """Authoritative materialized usage counters and lock coordination target per scope."""

    scope_key = models.CharField(max_length=255, primary_key=True)
    scope_type = models.CharField(max_length=32, db_index=True)
    tenant_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    workflow_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    run_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    provider = models.CharField(max_length=128, null=True, blank=True, db_index=True)

    # Materialized settled usage
    settled_calls = models.BigIntegerField(default=0)
    settled_tokens = models.BigIntegerField(default=0)
    settled_cost_micros = models.BigIntegerField(default=0)

    # Materialized active reserved exposure
    reserved_calls = models.BigIntegerField(default=0)
    reserved_tokens = models.BigIntegerField(default=0)
    reserved_cost_micros = models.BigIntegerField(default=0)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ai_governance_scope_counter"
        indexes = [
            models.Index(fields=["scope_type", "scope_key"], name="idx_counter_type_key"),
            models.Index(fields=["tenant_id", "scope_type"], name="idx_counter_tenant_type"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(settled_calls__gte=0)
                & models.Q(settled_tokens__gte=0)
                & models.Q(settled_cost_micros__gte=0)
                & models.Q(reserved_calls__gte=0)
                & models.Q(reserved_tokens__gte=0)
                & models.Q(reserved_cost_micros__gte=0),
                name="check_scope_counter_non_negative",
            )
        ]

    def __str__(self) -> str:
        return f"AIBudgetScopeCounter({self.scope_key})"


class AIBudgetSettingsModel(models.Model):
    """Durable storage for scope budget configurations."""

    scope_key = models.CharField(max_length=255, primary_key=True)
    scope_type = models.CharField(max_length=32, db_index=True)
    tenant_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    workflow_id = models.CharField(max_length=128, null=True, blank=True)
    run_id = models.CharField(max_length=128, null=True, blank=True)
    provider = models.CharField(max_length=128, null=True, blank=True)

    version = models.PositiveIntegerField(default=1)
    max_calls = models.BigIntegerField(null=True, blank=True)
    max_total_tokens = models.BigIntegerField(null=True, blank=True)
    max_prompt_tokens = models.BigIntegerField(null=True, blank=True)
    max_completion_tokens = models.BigIntegerField(null=True, blank=True)
    max_estimated_cost_micros = models.BigIntegerField(null=True, blank=True)
    max_duration_seconds = models.FloatField(null=True, blank=True)

    allowed_providers = models.JSONField(default=list)
    denied_providers = models.JSONField(default=list)
    allow_overrun = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ai_governance_budget_settings"
        constraints = [
            models.CheckConstraint(
                condition=models.Q(version__gt=0),
                name="check_budget_settings_version_positive",
            ),
            models.CheckConstraint(
                condition=(models.Q(max_calls__gte=0) | models.Q(max_calls__isnull=True))
                & (models.Q(max_total_tokens__gte=0) | models.Q(max_total_tokens__isnull=True))
                & (models.Q(max_prompt_tokens__gte=0) | models.Q(max_prompt_tokens__isnull=True))
                & (
                    models.Q(max_completion_tokens__gte=0)
                    | models.Q(max_completion_tokens__isnull=True)
                )
                & (
                    models.Q(max_estimated_cost_micros__gte=0)
                    | models.Q(max_estimated_cost_micros__isnull=True)
                )
                & (
                    models.Q(max_duration_seconds__gte=0)
                    | models.Q(max_duration_seconds__isnull=True)
                ),
                name="check_budget_settings_limits_non_negative",
            ),
        ]

    def __str__(self) -> str:
        return f"AIBudgetSettingsModel({self.scope_key}, v{self.version})"


class AIReservationModel(models.Model):
    """Durable record of reserved AI call capacity."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SETTLED = "settled", "Settled"
        RELEASED = "released", "Released"
        EXPIRED = "expired", "Expired"

    class DispatchStatusChoice(models.TextChoices):
        NOT_DISPATCHED = "not_dispatched", "Not Dispatched"
        DISPATCH_STARTED = "dispatch_started", "Dispatch Started"
        RESPONSE_RECEIVED = "response_received", "Response Received"
        RESPONSE_LOST = "response_lost", "Response Lost"
        BILLING_UNCERTAIN = "billing_uncertain", "Billing Uncertain"

    reservation_id = models.CharField(max_length=128, primary_key=True)
    idempotency_key = models.CharField(max_length=128, db_index=True)

    scope_key = models.CharField(max_length=255, db_index=True)
    scope_type = models.CharField(max_length=32)
    tenant_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    workflow_id = models.CharField(max_length=128, null=True, blank=True)
    run_id = models.CharField(max_length=128, null=True, blank=True)
    scope_provider = models.CharField(max_length=128, null=True, blank=True)

    provider = models.CharField(max_length=128)
    model = models.CharField(max_length=128)

    estimated_prompt_tokens = models.BigIntegerField()
    estimated_completion_tokens = models.BigIntegerField()
    estimated_total_tokens = models.BigIntegerField()
    estimated_cost_micros = models.BigIntegerField()
    attempt_number = models.PositiveIntegerField(default=1)

    status = models.CharField(
        max_length=32, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    dispatch_status = models.CharField(
        max_length=32,
        choices=DispatchStatusChoice.choices,
        default=DispatchStatusChoice.NOT_DISPATCHED,
        db_index=True,
    )
    created_at = models.DateTimeField(db_index=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    audit_metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "ai_governance_reservation"
        constraints = [
            models.UniqueConstraint(
                fields=["idempotency_key"],
                name="unique_reservation_global_idempotency_key",
            ),
            models.CheckConstraint(
                condition=models.Q(estimated_prompt_tokens__gte=0)
                & models.Q(estimated_completion_tokens__gte=0)
                & models.Q(estimated_total_tokens__gte=0)
                & models.Q(estimated_cost_micros__gte=0)
                & models.Q(attempt_number__gte=1),
                name="check_reservation_estimates_non_negative",
            ),
        ]
        indexes = [
            models.Index(fields=["tenant_id", "idempotency_key"], name="idx_res_tenant_idemp"),
            models.Index(fields=["status", "expires_at"], name="idx_res_status_expires"),
            models.Index(fields=["scope_key", "status"], name="idx_res_scope_status"),
        ]

    def __str__(self) -> str:
        return f"AIReservationModel({self.reservation_id}, {self.status})"


class AISettlementModel(models.Model):
    """Durable record of a finalized AI settlement."""

    class SettlementTypeChoice(models.TextChoices):
        EXACT = "exact", "Exact"
        ESTIMATED_EXPOSURE = "estimated_exposure", "Estimated Exposure"
        ZERO_COST_REJECTED = "zero_cost_rejected", "Zero Cost Rejected"
        UNKNOWN_USAGE = "unknown_usage", "Unknown Usage"

    class OverrunMarkerChoice(models.TextChoices):
        NONE = "none", "None"
        CALLS_OVERRUN = "calls_overrun", "Calls Overrun"
        TOKENS_OVERRUN = "tokens_overrun", "Tokens Overrun"
        COST_OVERRUN = "cost_overrun", "Cost Overrun"
        TIME_OVERRUN = "time_overrun", "Time Overrun"

    settlement_id = models.CharField(max_length=128, primary_key=True)
    reservation = models.OneToOneField(
        AIReservationModel,
        on_delete=models.PROTECT,
        related_name="settlement",
        to_field="reservation_id",
        db_column="reservation_id",
    )
    idempotency_key = models.CharField(max_length=128, db_index=True)

    scope_key = models.CharField(max_length=255, db_index=True)
    scope_type = models.CharField(max_length=32)
    tenant_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    workflow_id = models.CharField(max_length=128, null=True, blank=True)
    run_id = models.CharField(max_length=128, null=True, blank=True)
    scope_provider = models.CharField(max_length=128, null=True, blank=True)

    provider = models.CharField(max_length=128)
    model = models.CharField(max_length=128)
    settlement_type = models.CharField(max_length=32, choices=SettlementTypeChoice.choices)

    actual_prompt_tokens = models.BigIntegerField()
    actual_completion_tokens = models.BigIntegerField()
    actual_total_tokens = models.BigIntegerField()
    actual_cost_micros = models.BigIntegerField()
    attempt_number = models.PositiveIntegerField()

    settled_at = models.DateTimeField(db_index=True)
    overrun_marker = models.CharField(
        max_length=32, choices=OverrunMarkerChoice.choices, default=OverrunMarkerChoice.NONE
    )
    overrun_tokens = models.BigIntegerField(default=0)
    overrun_cost_micros = models.BigIntegerField(default=0)
    duration_seconds = models.FloatField(default=0.0)
    audit_metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "ai_governance_settlement"
        constraints = [
            models.UniqueConstraint(
                fields=["idempotency_key"],
                name="unique_settlement_global_idempotency_key",
            ),
            models.CheckConstraint(
                condition=models.Q(actual_prompt_tokens__gte=0)
                & models.Q(actual_completion_tokens__gte=0)
                & models.Q(actual_total_tokens__gte=0)
                & models.Q(actual_cost_micros__gte=0)
                & models.Q(overrun_tokens__gte=0)
                & models.Q(overrun_cost_micros__gte=0)
                & models.Q(duration_seconds__gte=0.0)
                & models.Q(attempt_number__gte=1),
                name="check_settlement_values_non_negative",
            ),
        ]
        indexes = [
            models.Index(fields=["tenant_id", "idempotency_key"], name="idx_stl_tenant_idemp"),
            models.Index(fields=["scope_key", "settled_at"], name="idx_stl_scope_time"),
        ]

    def __str__(self) -> str:
        return f"AISettlementModel({self.settlement_id}, {self.settlement_type})"


class AIUsageLedgerModel(models.Model):
    """Immutable audit trail of AI governance ledger events."""

    entry_id = models.CharField(max_length=128, primary_key=True)
    event_type = models.CharField(max_length=32, db_index=True)
    idempotency_key = models.CharField(max_length=128, db_index=True)

    scope_key = models.CharField(max_length=255, db_index=True)
    scope_type = models.CharField(max_length=32)
    tenant_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    workflow_id = models.CharField(max_length=128, null=True, blank=True)
    run_id = models.CharField(max_length=128, null=True, blank=True)
    scope_provider = models.CharField(max_length=128, null=True, blank=True)

    provider = models.CharField(max_length=128)
    model = models.CharField(max_length=128)
    timestamp = models.DateTimeField(db_index=True)

    tokens_delta = models.BigIntegerField()
    cost_micros_delta = models.BigIntegerField()
    calls_delta = models.IntegerField()

    overrun_marker = models.CharField(max_length=32, default="none")
    notes = models.CharField(max_length=256, blank=True, default="")
    audit_metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "ai_governance_ledger"
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["scope_key", "timestamp"], name="idx_ledger_scope_time"),
            models.Index(fields=["tenant_id", "timestamp"], name="idx_ledger_tenant_time"),
        ]

    def __str__(self) -> str:
        return f"AIUsageLedgerModel({self.entry_id}, {self.event_type})"
