import logging

from celery import shared_task
from django.db import transaction
from django.utils import timezone

from flowforge.workflows.engine import WorkflowExecutionError, execute_workflow_pipeline
from flowforge.workflows.models import WorkflowRun, WorkflowStepRun

logger = logging.getLogger(__name__)


@shared_task(max_retries=0)
def execute_workflow_run_task(run_id: str):
    """
    Celery background worker task for executing a WorkflowRun.
    - Atomically claims the run from QUEUED -> RUNNING.
    - If already claimed or processed, immediately exits (at-least-once idempotency).
    - Uses run.step_snapshot to ensure execution remains immutable.
    - Persists step audits and updates final run status.
    """
    now = timezone.now()
    # Atomic claim: conditional update ensures only one worker runs this task
    updated = WorkflowRun.objects.filter(id=run_id, state=WorkflowRun.State.QUEUED).update(
        state=WorkflowRun.State.RUNNING,
        started_at=now,
    )

    if updated == 0:
        logger.warning(
            "WorkflowRun %s was not in queued state or already claimed. Skipping.", run_id
        )
        return

    try:
        run = WorkflowRun.objects.select_related("source_job", "owner").get(id=run_id)
    except WorkflowRun.DoesNotExist:
        logger.error("WorkflowRun %s not found in database.", run_id)
        return

    source_job = run.source_job
    input_data = source_job.output_data or []
    initial_input_row_count = len(input_data)
    step_snapshot = run.step_snapshot or []
    initial_schema = source_job.column_names or (list(input_data[0].keys()) if input_data else [])

    try:
        output_data, step_audits = execute_workflow_pipeline(
            input_data, step_snapshot, schema=initial_schema
        )

        with transaction.atomic():
            run.state = WorkflowRun.State.COMPLETED
            run.output_data = output_data
            run.input_row_count = initial_input_row_count
            run.output_row_count = len(output_data)
            run.finished_at = timezone.now()
            run.save(
                update_fields=[
                    "state",
                    "output_data",
                    "input_row_count",
                    "output_row_count",
                    "finished_at",
                ]
            )

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
            run.state = WorkflowRun.State.FAILED
            run.output_data = []
            run.input_row_count = initial_input_row_count
            run.output_row_count = 0
            run.error_summary = exc.message
            run.finished_at = timezone.now()
            run.save(
                update_fields=[
                    "state",
                    "output_data",
                    "input_row_count",
                    "output_row_count",
                    "error_summary",
                    "finished_at",
                ]
            )

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

    except Exception as exc:
        logger.exception("Unexpected exception executing WorkflowRun %s", run_id)
        with transaction.atomic():
            run.state = WorkflowRun.State.FAILED
            run.output_data = []
            run.input_row_count = initial_input_row_count
            run.output_row_count = 0
            run.error_summary = f"Internal execution error: {str(exc)}"
            run.finished_at = timezone.now()
            run.save(
                update_fields=[
                    "state",
                    "output_data",
                    "input_row_count",
                    "output_row_count",
                    "error_summary",
                    "finished_at",
                ]
            )
