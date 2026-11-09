"""Service layer coordinating execution between WorkflowDefinition and ProcessingJob."""

from django.db import transaction
from django.utils import timezone

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.engine import WorkflowExecutionError, execute_workflow_pipeline
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStepRun


class WorkflowServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def execute_workflow_run(
    workflow_def: WorkflowDefinition, source_job: ProcessingJob, user
) -> WorkflowRun:
    """
    Synchronously execute a WorkflowDefinition on a source ProcessingJob.
    Enforces ownership consistency and status requirements.
    Persists audit records into WorkflowRun and WorkflowStepRun.
    """
    if workflow_def.owner != user:
        raise WorkflowServiceError("Workflow definition not found.", status_code=404)
    if source_job.owner != user:
        raise WorkflowServiceError("Source processing job not found.", status_code=404)
    if source_job.state != ProcessingJob.State.COMPLETED:
        raise WorkflowServiceError(
            f"Cannot execute workflow against source job in '{source_job.state}' state. Must be 'completed'.",
            status_code=400,
        )

    input_data = source_job.output_data or []
    initial_input_row_count = len(input_data)
    step_snapshot = list(workflow_def.steps)

    try:
        output_data, step_audits = execute_workflow_pipeline(input_data, step_snapshot)

        with transaction.atomic():
            run = WorkflowRun.objects.create(
                owner=user,
                workflow_definition=workflow_def,
                source_job=source_job,
                state=WorkflowRun.State.COMPLETED,
                step_snapshot=step_snapshot,
                output_data=output_data,
                input_row_count=initial_input_row_count,
                output_row_count=len(output_data),
                finished_at=timezone.now(),
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

        return run

    except WorkflowExecutionError as exc:
        with transaction.atomic():
            run = WorkflowRun.objects.create(
                owner=user,
                workflow_definition=workflow_def,
                source_job=source_job,
                state=WorkflowRun.State.FAILED,
                step_snapshot=step_snapshot,
                output_data=[],
                input_row_count=initial_input_row_count,
                output_row_count=0,
                error_summary=exc.message,
                finished_at=timezone.now(),
            )
            # Create completed step runs and failed step run
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

        return run
