import uuid

from django.conf import settings
from django.db import models


class ProcessingJob(models.Model):
    """
    Persisted record of a deterministic document ingestion/processing job.
    Synchronously executed in M0-002; owner-isolated via foreign key.
    """

    class State(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    class FileType(models.TextChoices):
        CSV = "csv", "CSV"
        JSON = "json", "JSON"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="processing_jobs",
    )
    file_type = models.CharField(max_length=10, choices=FileType.choices)
    state = models.CharField(
        max_length=20,
        choices=State.choices,
        default=State.PENDING,
        db_index=True,
    )
    original_filename = models.CharField(max_length=255)
    input_row_count = models.PositiveIntegerField(default=0)
    output_row_count = models.PositiveIntegerField(default=0)
    validation_errors = models.TextField(null=True, blank=True)
    column_names = models.JSONField(default=list, blank=True)
    output_data = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Processing Job"
        verbose_name_plural = "Processing Jobs"

    def __str__(self) -> str:
        return f"ProcessingJob {self.id} ({self.file_type} - {self.state})"
