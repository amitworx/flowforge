import uuid

from django.conf import settings
from django.db import models

from flowforge.processing.models import ProcessingJob


class WorkflowDefinition(models.Model):
    """
    Representation of an ordered series of validated step definitions.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workflow_definitions",
    )
    name = models.CharField(max_length=128)
    steps = models.JSONField(help_text="Ordered list of validated transformation steps.")
    is_archived = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Whether this workflow definition is archived.",
    )
    archived_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when this workflow was archived.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"


class WorkflowRun(models.Model):
    """
    Synchronous execution audit record of a WorkflowDefinition against a ProcessingJob.
    """

    class State(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    class DispatchStatus(models.TextChoices):
        DISPATCHED = "dispatched", "Dispatched"
        PUBLISHING = "publishing", "Publishing"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workflow_runs",
    )
    workflow_definition = models.ForeignKey(
        WorkflowDefinition,
        on_delete=models.CASCADE,
        related_name="runs",
    )
    source_job = models.ForeignKey(
        ProcessingJob,
        on_delete=models.CASCADE,
        related_name="workflow_runs",
    )
    state = models.CharField(
        max_length=20,
        choices=State.choices,
    )
    dispatch_status = models.CharField(
        max_length=20,
        choices=DispatchStatus.choices,
        default=DispatchStatus.DISPATCHED,
        db_index=True,
    )
    dispatch_attempts = models.PositiveIntegerField(default=1)
    claim_token = models.UUIDField(null=True, blank=True, db_index=True)
    step_snapshot = models.JSONField(
        help_text="Immutable snapshot of the steps executed in this run."
    )
    output_data = models.JSONField(
        default=list,
        blank=True,
        help_text="Bounded normalized output JSON resulting from successful pipeline execution.",
    )
    output_columns = models.JSONField(
        default=list,
        blank=True,
        help_text="Ordered list of final schema column names for completed workflow output.",
    )
    input_row_count = models.PositiveIntegerField(default=0)
    output_row_count = models.PositiveIntegerField(default=0)
    error_summary = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Run {self.id} - {self.state}"

    def get_output_schema(self) -> list[str]:
        if self.output_columns and isinstance(self.output_columns, list):
            return list(self.output_columns)
        if self.output_data and len(self.output_data) > 0 and isinstance(self.output_data[0], dict):
            return list(self.output_data[0].keys())

        from flowforge.workflows.export import derive_workflow_schema

        initial_schema = []
        if self.source_job and self.source_job.column_names:
            initial_schema = list(self.source_job.column_names)
        elif self.source_job and self.source_job.output_data:
            initial_schema = list(self.source_job.output_data[0].keys())

        return derive_workflow_schema(initial_schema, self.step_snapshot or [])


class WorkflowStepRun(models.Model):
    """
    Per-step audit record of execution within a WorkflowRun.
    """

    class Status(models.TextChoices):
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow_run = models.ForeignKey(
        WorkflowRun,
        on_delete=models.CASCADE,
        related_name="step_runs",
    )
    step_index = models.PositiveIntegerField()
    operation = models.CharField(max_length=64)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
    )
    input_row_count = models.PositiveIntegerField(default=0)
    output_row_count = models.PositiveIntegerField(default=0)
    error_summary = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["step_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["workflow_run", "step_index"],
                name="unique_step_index_per_run",
            )
        ]

    def __str__(self) -> str:
        return f"Step {self.step_index} ({self.operation}) - {self.status}"
