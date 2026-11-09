import logging
import uuid
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from flowforge.workflows.models import WorkflowRun
from flowforge.workflows.tasks import execute_workflow_run_task

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Recovers stranded async workflow runs (dispatch failures or timed-out running workers)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--stale-minutes",
            type=int,
            default=15,
            help="Minutes after which a running workflow run without completion is considered stale.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Maximum number of runs to recover in a single batch.",
        )
        parser.add_argument(
            "--action",
            type=str,
            choices=["requeue", "fail"],
            default="requeue",
            help="Action to take on stale running runs: 'requeue' (rotate claim token and redispatch) or 'fail'.",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        stale_minutes = options["stale_minutes"]
        limit = options["limit"]
        action = options["action"]
        now = timezone.now()
        stale_threshold = now - timedelta(minutes=stale_minutes)

        self.stdout.write(
            f"Starting workflow run recovery (stale threshold: {stale_minutes}m, limit: {limit})..."
        )

        # 1. Recover DISPATCH_FAILED runs
        dispatch_failed_runs = list(
            WorkflowRun.objects.filter(
                state=WorkflowRun.State.QUEUED,
                dispatch_status=WorkflowRun.DispatchStatus.FAILED,
            ).order_by("created_at")[:limit]
        )

        recovered_dispatch_count = 0
        for run in dispatch_failed_runs:
            with transaction.atomic():
                run.dispatch_attempts += 1
                run.dispatch_status = WorkflowRun.DispatchStatus.DISPATCHED
                run.error_summary = None
                run.save(update_fields=["dispatch_attempts", "dispatch_status", "error_summary"])

            try:
                execute_workflow_run_task.delay(
                    str(run.id), str(run.claim_token) if run.claim_token else None
                )
                recovered_dispatch_count += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Re-dispatched failed run {run.id} (attempt {run.dispatch_attempts})."
                    )
                )
            except Exception as exc:
                run.dispatch_status = WorkflowRun.DispatchStatus.FAILED
                run.error_summary = f"Broker unavailable during recovery dispatch: {str(exc)}"
                run.save(update_fields=["dispatch_status", "error_summary"])
                self.stdout.write(self.style.ERROR(f"Failed to re-dispatch run {run.id}: {exc}"))

        # 2. Recover stale RUNNING runs
        stale_running_runs = list(
            WorkflowRun.objects.filter(
                state=WorkflowRun.State.RUNNING,
                started_at__lt=stale_threshold,
            ).order_by("started_at")[:limit]
        )

        recovered_running_count = 0
        for run in stale_running_runs:
            new_fencing_token = uuid.uuid4()
            if action == "requeue":
                # Reclaim by rotating fencing token and setting state back to QUEUED
                updated = WorkflowRun.objects.filter(
                    id=run.id,
                    claim_token=run.claim_token,
                    state=WorkflowRun.State.RUNNING,
                ).update(
                    state=WorkflowRun.State.QUEUED,
                    claim_token=new_fencing_token,
                    error_summary="Recovered from stale running state.",
                )
                if updated > 0:
                    execute_workflow_run_task.delay(str(run.id), str(new_fencing_token))
                    recovered_running_count += 1
                    self.stdout.write(
                        self.style.SUCCESS(f"Reclaimed and requeued stale run {run.id}.")
                    )
            else:
                updated = WorkflowRun.objects.filter(
                    id=run.id,
                    claim_token=run.claim_token,
                    state=WorkflowRun.State.RUNNING,
                ).update(
                    state=WorkflowRun.State.FAILED,
                    claim_token=new_fencing_token,
                    error_summary="Execution timed out and was aborted by recovery policy.",
                    finished_at=now,
                )
                if updated > 0:
                    recovered_running_count += 1
                    self.stdout.write(
                        self.style.WARNING(f"Marked timed-out run {run.id} as failed.")
                    )

        self.stdout.write(
            self.style.SUCCESS(
                f"Recovery complete. Recovered {recovered_dispatch_count} dispatch-failed runs, "
                f"{recovered_running_count} stale running runs."
            )
        )
