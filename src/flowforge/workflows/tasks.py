import logging
import uuid

from celery import shared_task
from django.db import models, transaction
from django.utils import timezone

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_workflow_pipeline,
    validate_step_definitions,
)
from flowforge.workflows.models import WorkflowRun, WorkflowStepRun

logger = logging.getLogger(__name__)

GENERIC_INTERNAL_ERROR_MESSAGE = "Internal workflow execution error occurred."


@shared_task(max_retries=0)
def execute_workflow_run_task(run_id: str, claim_token: str | None = None):
    """
    Celery background worker task for executing a WorkflowRun.
    - Atomically claims the run from QUEUED -> RUNNING with a unique fencing token.
    - If claim_token is provided (e.g. from recovery), matches the current claim token or claims anew.
    - If already claimed or processed, immediately exits (at-least-once idempotency).
    - Validates source ownership, completed state, and snapshot integrity before executing.
    - Uses run.step_snapshot to ensure execution remains immutable.
    - Persists step audits and updates final run status.
    - Catches unexpected worker crashes, logs full traceback server-side, and sanitizes API error_summary.
    """
    new_claim_token = uuid.uuid4()
    now = timezone.now()

    # Atomic claim:
    # 1. State must be QUEUED (not already claimed by another worker)
    # 2. Or, if claim_token is explicitly provided (via recovery command rotation),
    #    state can be QUEUED with that specific claim_token
    claim_q = models.Q(id=run_id, state=WorkflowRun.State.QUEUED)
    if claim_token:
        claim_q &= models.Q(claim_token__isnull=True) | models.Q(claim_token=claim_token)
    else:
        # Standard first-time dispatch: claim_token must be NULL or fresh
        claim_q &= models.Q(claim_token__isnull=True)

    updated = WorkflowRun.objects.filter(claim_q).update(
        state=WorkflowRun.State.RUNNING,
        started_at=now,
        claim_token=new_claim_token,
    )

    if updated == 0:
        logger.warning(
            "WorkflowRun %s was not eligible for worker claim (already running, completed, or fenced). Skipping.",
            run_id,
        )
        return

    worker_token = new_claim_token

    try:
        run = WorkflowRun.objects.select_related("source_job", "workflow_definition", "owner").get(
            id=run_id
        )
    except WorkflowRun.DoesNotExist:
        logger.error("WorkflowRun %s not found in database.", run_id)
        return

    # Security & Integrity Validations before execution
    source_job = run.source_job
    workflow_def = run.workflow_definition
    user = run.owner

    # 1. Ownership consistency verification
    if source_job.owner_id != user.id or (workflow_def and workflow_def.owner_id != user.id):
        logger.warning(
            "Security invariant violation: WorkflowRun %s has mismatched ownership. Aborting execution.",
            run_id,
        )
        WorkflowRun.objects.filter(
            id=run_id,
            state=WorkflowRun.State.RUNNING,
            claim_token=worker_token,
        ).update(
            state=WorkflowRun.State.FAILED,
            output_data=[],
            input_row_count=0,
            output_row_count=0,
            error_summary="Source data ownership validation failed.",
            finished_at=timezone.now(),
        )
        return

    # 2. Source job completion status verification
    if source_job.state != ProcessingJob.State.COMPLETED:
        logger.warning(
            "WorkflowRun %s source job %s is in state '%s', not 'completed'. Aborting execution.",
            run_id,
            source_job.id,
            source_job.state,
        )
        WorkflowRun.objects.filter(
            id=run_id,
            state=WorkflowRun.State.RUNNING,
            claim_token=worker_token,
        ).update(
            state=WorkflowRun.State.FAILED,
            output_data=[],
            input_row_count=0,
            output_row_count=0,
            error_summary=f"Source processing job is not in completed state (current: '{source_job.state}').",
            finished_at=timezone.now(),
        )
        return

    # 3. Snapshot structure and integrity verification
    step_snapshot = run.step_snapshot
    if not isinstance(step_snapshot, list):
        logger.warning("WorkflowRun %s has invalid step_snapshot format.", run_id)
        WorkflowRun.objects.filter(
            id=run_id,
            state=WorkflowRun.State.RUNNING,
            claim_token=worker_token,
        ).update(
            state=WorkflowRun.State.FAILED,
            output_data=[],
            input_row_count=0,
            output_row_count=0,
            error_summary="Invalid step snapshot format.",
            finished_at=timezone.now(),
        )
        return

    try:
        validate_step_definitions(step_snapshot)
    except WorkflowExecutionError as exc:
        WorkflowRun.objects.filter(
            id=run_id,
            state=WorkflowRun.State.RUNNING,
            claim_token=worker_token,
        ).update(
            state=WorkflowRun.State.FAILED,
            output_data=[],
            input_row_count=0,
            output_row_count=0,
            error_summary=exc.message,
            finished_at=timezone.now(),
        )
        return

    input_data = source_job.output_data or []
    initial_input_row_count = len(input_data)
    initial_schema = source_job.column_names or (list(input_data[0].keys()) if input_data else [])

    try:
        output_data, step_audits = execute_workflow_pipeline(
            input_data, step_snapshot, schema=initial_schema
        )

        with transaction.atomic():
            # Check fencing token AND state=RUNNING before persisting
            rows_updated = WorkflowRun.objects.filter(
                id=run_id,
                state=WorkflowRun.State.RUNNING,
                claim_token=worker_token,
            ).update(
                state=WorkflowRun.State.COMPLETED,
                output_data=output_data,
                input_row_count=initial_input_row_count,
                output_row_count=len(output_data),
                finished_at=timezone.now(),
            )
            if rows_updated == 0:
                logger.warning(
                    "Fencing conflict: WorkflowRun %s was reclaimed or terminally failed. Dropping results.",
                    run_id,
                )
                return

            step_runs = [
                WorkflowStepRun(
                    workflow_run=run,
                    step_index=audit["step_index"],
                    operation=audit["operation"],
                    status=audit["status"],
                    input_row_count=audit["input_row_count"],
                    output_row_count=audit["output_row_count"],
                    error_summary=audit["error_summary"],
                )
                for audit in step_audits
            ]
            WorkflowStepRun.objects.bulk_create(step_runs)

    except WorkflowExecutionError as exc:
        with transaction.atomic():
            rows_updated = WorkflowRun.objects.filter(
                id=run_id,
                state=WorkflowRun.State.RUNNING,
                claim_token=worker_token,
            ).update(
                state=WorkflowRun.State.FAILED,
                output_data=[],
                input_row_count=initial_input_row_count,
                output_row_count=0,
                error_summary=exc.message,
                finished_at=timezone.now(),
            )
            if rows_updated == 0:
                logger.warning(
                    "Fencing conflict on failure: WorkflowRun %s was reclaimed or terminally failed.",
                    run_id,
                )
                return

            step_objs = []
            for audit in exc.completed_step_audit:
                step_objs.append(
                    WorkflowStepRun(
                        workflow_run=run,
                        step_index=audit["step_index"],
                        operation=audit["operation"],
                        status=audit["status"],
                        input_row_count=audit["input_row_count"],
                        output_row_count=audit["output_row_count"],
                        error_summary=audit["error_summary"],
                    )
                )
            if exc.failed_step_audit:
                step_objs.append(
                    WorkflowStepRun(
                        workflow_run=run,
                        step_index=exc.failed_step_audit["step_index"],
                        operation=exc.failed_step_audit["operation"],
                        status=exc.failed_step_audit["status"],
                        input_row_count=exc.failed_step_audit["input_row_count"],
                        output_row_count=exc.failed_step_audit["output_row_count"],
                        error_summary=exc.failed_step_audit["error_summary"],
                    )
                )
            WorkflowStepRun.objects.bulk_create(step_objs)

    except Exception:
        logger.exception("Unexpected exception executing WorkflowRun %s", run_id)
        with transaction.atomic():
            WorkflowRun.objects.filter(
                id=run_id,
                state=WorkflowRun.State.RUNNING,
                claim_token=worker_token,
            ).update(
                state=WorkflowRun.State.FAILED,
                output_data=[],
                input_row_count=initial_input_row_count,
                output_row_count=0,
                error_summary=GENERIC_INTERNAL_ERROR_MESSAGE,
                finished_at=timezone.now(),
            )
