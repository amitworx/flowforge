import logging
import uuid
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import models, transaction
from django.utils import timezone

from flowforge.workflows.models import WorkflowRun
from flowforge.workflows.tasks import execute_workflow_run_task

logger = logging.getLogger(__name__)


def calculate_backoff_seconds(attempt: int, base: int = 15, max_seconds: int = 900) -> int:
    """
    Exponential backoff in seconds based on attempt number: base * 2^(attempt - 2), capped at max_seconds.
    Attempt 1 (initial run failure) has 0s backoff (immediate retry eligible).
    Attempt 2 has 15s backoff, Attempt 3 has 30s, Attempt 4 has 60s, etc.
    """
    if attempt <= 1:
        return 0
    delay = base * (2 ** (attempt - 2))
    return min(delay, max_seconds)


class Command(BaseCommand):
    help = "Recovers stranded async workflow runs (dispatch failures or timed-out running workers)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--stale-minutes",
            type=int,
            default=15,
            help="Minutes after which a running or unacknowledged queued run is considered stale.",
        )
        parser.add_argument(
            "--max-attempts",
            type=int,
            default=3,
            help="Maximum dispatch/recovery attempts before marking a run as terminally FAILED.",
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
        parser.add_argument(
            "--ignore-backoff",
            action="store_true",
            default=False,
            help="Ignore exponential backoff delay (primarily for testing and urgent administrative overrides).",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        stale_minutes = options["stale_minutes"]
        max_attempts = options["max_attempts"]
        limit = options["limit"]
        action = options["action"]
        ignore_backoff = options["ignore_backoff"]

        if stale_minutes <= 0:
            raise CommandError("--stale-minutes must be positive.")
        if max_attempts <= 0:
            raise CommandError("--max-attempts must be positive.")
        if limit <= 0:
            raise CommandError("--limit must be positive.")

        now = timezone.now()
        stale_threshold = now - timedelta(minutes=stale_minutes)

        self.stdout.write(
            f"Starting workflow run recovery (stale threshold: {stale_minutes}m, max attempts: {max_attempts}, limit: {limit})..."
        )

        recovered_dispatch_count = 0
        terminal_failed_count = 0
        recovered_running_count = 0

        # 1. Recover stranded QUEUED runs:
        # a) dispatch_status == FAILED
        # b) dispatch_status == DISPATCHED but (started_at or created_at) < stale_threshold (crashed after dispatch)
        # c) dispatch_status == PUBLISHING but (started_at or created_at) < stale_threshold (crashed while publishing)
        candidate_queued_runs = list(
            WorkflowRun.objects.filter(
                models.Q(state=WorkflowRun.State.QUEUED)
                & (
                    models.Q(dispatch_status=WorkflowRun.DispatchStatus.FAILED)
                    | (
                        models.Q(dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED)
                        & (
                            (
                                models.Q(started_at__isnull=False)
                                & models.Q(started_at__lt=stale_threshold)
                            )
                            | (
                                models.Q(started_at__isnull=True)
                                & models.Q(created_at__lt=stale_threshold)
                            )
                        )
                    )
                    | (
                        models.Q(dispatch_status=WorkflowRun.DispatchStatus.PUBLISHING)
                        & (
                            (
                                models.Q(started_at__isnull=False)
                                & models.Q(started_at__lt=stale_threshold)
                            )
                            | (
                                models.Q(started_at__isnull=True)
                                & models.Q(created_at__lt=stale_threshold)
                            )
                        )
                    )
                )
            ).order_by("created_at")[:limit]
        )

        for candidate in candidate_queued_runs:
            # Check attempt budget first
            if candidate.dispatch_attempts >= max_attempts:
                new_fencing_token = uuid.uuid4()
                updated = WorkflowRun.objects.filter(
                    id=candidate.id,
                    state=WorkflowRun.State.QUEUED,
                ).update(
                    state=WorkflowRun.State.FAILED,
                    claim_token=new_fencing_token,
                    error_summary=f"Recovery attempts exhausted (reached max attempts: {max_attempts}).",
                    finished_at=now,
                )
                if updated > 0:
                    terminal_failed_count += 1
                    self.stdout.write(
                        self.style.WARNING(
                            f"Run {candidate.id} exhausted {max_attempts} attempts. Marked terminal FAILED."
                        )
                    )
                continue

            # Exponential backoff check:
            last_activity = candidate.started_at or candidate.created_at
            if not ignore_backoff and last_activity:
                backoff_secs = calculate_backoff_seconds(candidate.dispatch_attempts)
                if (now - last_activity).total_seconds() < backoff_secs:
                    continue

            # Concurrency-safe atomic reservation using row locking inside atomic transaction
            reserved_run = None
            with transaction.atomic():
                locked_runs = list(
                    WorkflowRun.objects.select_for_update(skip_locked=True)
                    .filter(
                        id=candidate.id,
                        state=WorkflowRun.State.QUEUED,
                    )
                    .filter(
                        models.Q(dispatch_status=WorkflowRun.DispatchStatus.FAILED)
                        | (
                            models.Q(dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED)
                            & (
                                (
                                    models.Q(started_at__isnull=False)
                                    & models.Q(started_at__lt=stale_threshold)
                                )
                                | (
                                    models.Q(started_at__isnull=True)
                                    & models.Q(created_at__lt=stale_threshold)
                                )
                            )
                        )
                        | (
                            models.Q(dispatch_status=WorkflowRun.DispatchStatus.PUBLISHING)
                            & (
                                (
                                    models.Q(started_at__isnull=False)
                                    & models.Q(started_at__lt=stale_threshold)
                                )
                                | (
                                    models.Q(started_at__isnull=True)
                                    & models.Q(created_at__lt=stale_threshold)
                                )
                            )
                        )
                    )
                )
                if not locked_runs:
                    continue

                run = locked_runs[0]
                if run.dispatch_attempts >= max_attempts:
                    run.state = WorkflowRun.State.FAILED
                    run.claim_token = uuid.uuid4()
                    run.error_summary = (
                        f"Recovery attempts exhausted (reached max attempts: {max_attempts})."
                    )
                    run.finished_at = now
                    run.save(update_fields=["state", "claim_token", "error_summary", "finished_at"])
                    terminal_failed_count += 1
                    continue

                run.dispatch_attempts += 1
                run.dispatch_status = WorkflowRun.DispatchStatus.PUBLISHING
                run.error_summary = None
                run.started_at = now
                run.save(
                    update_fields=[
                        "dispatch_attempts",
                        "dispatch_status",
                        "error_summary",
                        "started_at",
                    ]
                )
                reserved_run = run

            # Post-commit dispatch to Celery broker
            if reserved_run:
                try:
                    execute_workflow_run_task.delay(
                        str(reserved_run.id),
                        str(reserved_run.claim_token) if reserved_run.claim_token else None,
                    )
                    # On publication success: finalize reservation to DISPATCHED
                    WorkflowRun.objects.filter(
                        id=reserved_run.id,
                        state=WorkflowRun.State.QUEUED,
                        dispatch_status=WorkflowRun.DispatchStatus.PUBLISHING,
                    ).update(
                        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
                    )
                    recovered_dispatch_count += 1
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"Re-dispatched failed run {reserved_run.id} (attempt {reserved_run.dispatch_attempts})."
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to re-dispatch run %s to broker during recovery", reserved_run.id
                    )
                    WorkflowRun.objects.filter(
                        id=reserved_run.id,
                        state=WorkflowRun.State.QUEUED,
                    ).update(
                        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
                        error_summary="Broker unavailable during recovery dispatch. Run is recoverable.",
                    )
                    self.stdout.write(
                        self.style.ERROR(
                            f"Failed to re-dispatch run {reserved_run.id}: broker unavailable."
                        )
                    )

        # 2. Recover stale RUNNING runs
        candidate_running_runs = list(
            WorkflowRun.objects.filter(
                state=WorkflowRun.State.RUNNING,
                started_at__lt=stale_threshold,
            ).order_by("started_at")[:limit]
        )

        for candidate in candidate_running_runs:
            new_fencing_token = uuid.uuid4()
            # Check total recovery/attempt budget including stale reclaims
            if candidate.dispatch_attempts >= max_attempts:
                # Terminal failure: must rotate claim token to fence old worker
                updated = WorkflowRun.objects.filter(
                    id=candidate.id,
                    claim_token=candidate.claim_token,
                    state=WorkflowRun.State.RUNNING,
                ).update(
                    state=WorkflowRun.State.FAILED,
                    claim_token=new_fencing_token,
                    error_summary=f"Recovery attempts exhausted (reached max attempts: {max_attempts}).",
                    finished_at=now,
                )
                if updated > 0:
                    terminal_failed_count += 1
                    self.stdout.write(
                        self.style.WARNING(
                            f"Stale run {candidate.id} exhausted {max_attempts} attempts. Marked terminal FAILED."
                        )
                    )
                continue

            if action == "requeue":
                # Concurrency-safe atomic claim reservation
                reserved_stale = None
                with transaction.atomic():
                    locked_stale = list(
                        WorkflowRun.objects.select_for_update(skip_locked=True).filter(
                            id=candidate.id,
                            claim_token=candidate.claim_token,
                            state=WorkflowRun.State.RUNNING,
                        )
                    )
                    if not locked_stale:
                        continue

                    run = locked_stale[0]
                    if run.dispatch_attempts >= max_attempts:
                        run.state = WorkflowRun.State.FAILED
                        run.claim_token = new_fencing_token
                        run.error_summary = (
                            f"Recovery attempts exhausted (reached max attempts: {max_attempts})."
                        )
                        run.finished_at = now
                        run.save(
                            update_fields=["state", "claim_token", "error_summary", "finished_at"]
                        )
                        terminal_failed_count += 1
                        continue

                    run.dispatch_attempts += 1
                    run.state = WorkflowRun.State.QUEUED
                    run.dispatch_status = WorkflowRun.DispatchStatus.PUBLISHING
                    run.claim_token = new_fencing_token
                    run.error_summary = "Recovered from stale running state."
                    run.started_at = now
                    run.save(
                        update_fields=[
                            "dispatch_attempts",
                            "state",
                            "dispatch_status",
                            "claim_token",
                            "error_summary",
                            "started_at",
                        ]
                    )
                    reserved_stale = run

                if reserved_stale:
                    try:
                        execute_workflow_run_task.delay(
                            str(reserved_stale.id), str(new_fencing_token)
                        )
                        WorkflowRun.objects.filter(
                            id=reserved_stale.id,
                            state=WorkflowRun.State.QUEUED,
                            dispatch_status=WorkflowRun.DispatchStatus.PUBLISHING,
                        ).update(
                            dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
                        )
                        recovered_running_count += 1
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"Reclaimed and requeued stale run {reserved_stale.id} (attempt {reserved_stale.dispatch_attempts})."
                            )
                        )
                    except Exception:
                        logger.exception(
                            "Failed to dispatch reclaimed run %s to broker", reserved_stale.id
                        )
                        WorkflowRun.objects.filter(
                            id=reserved_stale.id,
                            state=WorkflowRun.State.QUEUED,
                        ).update(
                            dispatch_status=WorkflowRun.DispatchStatus.FAILED,
                            error_summary="Broker unavailable during stale run recovery.",
                        )
                        self.stdout.write(
                            self.style.ERROR(
                                f"Failed to dispatch reclaimed run {reserved_stale.id}: broker unavailable."
                            )
                        )
            else:
                updated = WorkflowRun.objects.filter(
                    id=candidate.id,
                    claim_token=candidate.claim_token,
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
                        self.style.WARNING(f"Marked timed-out run {candidate.id} as failed.")
                    )

        self.stdout.write(
            self.style.SUCCESS(
                f"Recovery complete. Recovered {recovered_dispatch_count} dispatch-failed runs, "
                f"{terminal_failed_count} marked terminal failed, {recovered_running_count} stale running runs."
            )
        )
