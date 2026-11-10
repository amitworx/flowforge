import threading
from datetime import timedelta
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db import connection
from django.utils import timezone

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.engine import WorkflowExecutionError
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStepRun

User = get_user_model()


@pytest.fixture
def test_user(db):  # noqa: ARG001
    user, _ = User.objects.get_or_create(
        username="m04d_user", defaults={"email": "m04d@example.com"}
    )
    return user


@pytest.fixture
def sample_job(test_user):
    return ProcessingJob.objects.create(
        owner=test_user,
        original_filename="sample_m04d.csv",
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.COMPLETED,
        column_names=["city", "temp"],
        output_data=[{"city": "Berlin", "temp": "15"}],
    )


@pytest.fixture
def sample_wf(test_user):
    return WorkflowDefinition.objects.create(
        owner=test_user,
        name="Pipeline M04D",
        steps=[{"op": "select_columns", "columns": ["city"]}],
    )


# -------------------------------------------------------------------------
# 1. Stale worker cannot overwrite terminal FAILED on normal pipeline completion
# -------------------------------------------------------------------------
@pytest.mark.django_db(transaction=True)
def test_stale_worker_cannot_overwrite_terminal_failed_or_write_audits(sample_job, sample_wf):
    from flowforge.workflows.tasks import execute_workflow_run_task

    now = timezone.now()
    run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        dispatch_attempts=3,
        step_snapshot=sample_wf.steps,
    )

    worker_claimed_event = threading.Event()
    recovery_finished_event = threading.Event()

    def delayed_worker():
        connection.close()
        with patch("flowforge.workflows.tasks.execute_workflow_pipeline") as mock_pipeline:

            def fake_pipeline(*args, **kwargs):  # noqa: ARG001
                worker_claimed_event.set()
                recovery_finished_event.wait(timeout=5)
                return (
                    [{"city": "Berlin"}],
                    [
                        {
                            "step_index": 0,
                            "operation": "select_columns",
                            "status": "completed",
                            "input_row_count": 1,
                            "output_row_count": 1,
                            "error_summary": None,
                        }
                    ],
                )

            mock_pipeline.side_effect = fake_pipeline
            execute_workflow_run_task(str(run.id), None)
        connection.close()

    worker_thread = threading.Thread(target=delayed_worker)
    worker_thread.start()

    assert worker_claimed_event.wait(timeout=5)

    WorkflowRun.objects.filter(id=run.id).update(started_at=now - timedelta(minutes=30))
    call_command(
        "recover_workflow_runs",
        stale_minutes=15,
        max_attempts=3,
        action="requeue",
        ignore_backoff=True,
    )

    recovery_finished_event.set()
    worker_thread.join(timeout=5)

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert "exhausted" in (run.error_summary or "").lower()
    assert WorkflowStepRun.objects.filter(workflow_run=run).count() == 0


# -------------------------------------------------------------------------
# 2. Stale worker on pipeline failure branch cannot overwrite terminal FAILED
# -------------------------------------------------------------------------
@pytest.mark.django_db(transaction=True)
def test_stale_worker_failure_branch_cannot_overwrite_terminal_state(sample_job, sample_wf, caplog):
    import logging

    from flowforge.workflows.tasks import execute_workflow_run_task

    now = timezone.now()
    run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        dispatch_attempts=3,
        step_snapshot=sample_wf.steps,
    )

    worker_claimed_event = threading.Event()
    recovery_finished_event = threading.Event()

    simulated_failure = WorkflowExecutionError(
        message="Simulated deterministic step failure",
        step_index=0,
        completed_step_audit=[],
        failed_step_audit={
            "step_index": 0,
            "operation": "select_columns",
            "status": "failed",
            "input_row_count": 1,
            "output_row_count": 0,
            "error_summary": "Simulated deterministic step failure",
        },
    )

    def delayed_failing_worker():
        connection.close()
        with patch("flowforge.workflows.tasks.execute_workflow_pipeline") as mock_pipeline:

            def fake_pipeline(*args, **kwargs):  # noqa: ARG001
                worker_claimed_event.set()
                recovery_finished_event.wait(timeout=5)
                raise simulated_failure

            mock_pipeline.side_effect = fake_pipeline
            execute_workflow_run_task(str(run.id), None)
        connection.close()

    with caplog.at_level(logging.WARNING):
        worker_thread = threading.Thread(target=delayed_failing_worker)
        worker_thread.start()

        assert worker_claimed_event.wait(timeout=5)

        WorkflowRun.objects.filter(id=run.id).update(started_at=now - timedelta(minutes=30))
        call_command(
            "recover_workflow_runs",
            stale_minutes=15,
            max_attempts=3,
            action="requeue",
            ignore_backoff=True,
        )

        recovery_finished_event.set()
        worker_thread.join(timeout=5)

    run.refresh_from_db()
    # Must preserve the terminal exhaustion failure message, NOT the step failure message
    assert run.state == WorkflowRun.State.FAILED
    assert "exhausted" in (run.error_summary or "").lower()
    assert WorkflowStepRun.objects.filter(workflow_run=run).count() == 0

    # Explicitly verify the WorkflowExecutionError fencing conflict branch was exercised
    assert any(
        "Fencing conflict on failure: WorkflowRun" in record.message
        and str(run.id) in record.message
        for record in caplog.records
    )
    # And verify unexpected generic exception handler was NOT triggered
    assert not any(
        "Unexpected exception executing WorkflowRun" in record.message for record in caplog.records
    )


@pytest.mark.django_db(transaction=True)
def test_unfenced_worker_failure_branch_persists_workflow_execution_error_and_audits(
    sample_job, sample_wf
):
    """Proves that when un-fenced, WorkflowExecutionError persists exact step audits and error message."""
    from flowforge.workflows.tasks import execute_workflow_run_task

    run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        dispatch_attempts=1,
        step_snapshot=sample_wf.steps,
    )

    simulated_failure = WorkflowExecutionError(
        message="Deterministic unfenced step failure",
        step_index=0,
        completed_step_audit=[],
        failed_step_audit={
            "step_index": 0,
            "operation": "select_columns",
            "status": "failed",
            "input_row_count": 1,
            "output_row_count": 0,
            "error_summary": "Deterministic unfenced step failure",
        },
    )

    with patch(
        "flowforge.workflows.tasks.execute_workflow_pipeline", side_effect=simulated_failure
    ):
        execute_workflow_run_task(str(run.id), None)

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert run.error_summary == "Deterministic unfenced step failure"

    step_runs = list(WorkflowStepRun.objects.filter(workflow_run=run))
    assert len(step_runs) == 1
    assert step_runs[0].status == "failed"
    assert step_runs[0].operation == "select_columns"
    assert step_runs[0].error_summary == "Deterministic unfenced step failure"


# -------------------------------------------------------------------------
# 3. Durable recovery reservation prevents double-dispatch publish race
# -------------------------------------------------------------------------
@pytest.mark.django_db(transaction=True)
def test_durable_recovery_reservation_prevents_publish_race(sample_job, sample_wf):
    now = timezone.now()
    run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        dispatch_attempts=1,
        created_at=now - timedelta(minutes=20),
        step_snapshot=sample_wf.steps,
    )
    WorkflowRun.objects.filter(id=run.id).update(created_at=now - timedelta(minutes=20))

    cmd1_reserved_event = threading.Event()
    cmd2_finished_event = threading.Event()
    dispatches = []

    def cmd1_delay(*args, **kwargs):  # noqa: ARG001
        cmd1_reserved_event.set()
        cmd2_finished_event.wait(timeout=5)
        dispatches.append(("cmd1", args))

    def cmd2_delay(*args, **kwargs):  # noqa: ARG001
        dispatches.append(("cmd2", args))

    def run_cmd1():
        connection.close()
        with (
            patch(
                "flowforge.workflows.tasks.execute_workflow_run_task.delay",
                side_effect=cmd1_delay,
            ),
            patch(
                "flowforge.workflows.management.commands.recover_workflow_runs.execute_workflow_run_task.delay",
                side_effect=cmd1_delay,
            ),
        ):
            call_command(
                "recover_workflow_runs", stale_minutes=15, max_attempts=5, ignore_backoff=True
            )
        connection.close()

    def run_cmd2():
        connection.close()
        with (
            patch(
                "flowforge.workflows.tasks.execute_workflow_run_task.delay",
                side_effect=cmd2_delay,
            ),
            patch(
                "flowforge.workflows.management.commands.recover_workflow_runs.execute_workflow_run_task.delay",
                side_effect=cmd2_delay,
            ),
        ):
            call_command(
                "recover_workflow_runs", stale_minutes=15, max_attempts=5, ignore_backoff=True
            )
        connection.close()
        cmd2_finished_event.set()

    t1 = threading.Thread(target=run_cmd1)
    t1.start()

    assert cmd1_reserved_event.wait(timeout=5)

    t2 = threading.Thread(target=run_cmd2)
    t2.start()
    t2.join(timeout=5)
    t1.join(timeout=5)

    assert len(dispatches) == 1
    assert dispatches[0][0] == "cmd1"
    run.refresh_from_db()
    assert run.dispatch_attempts == 2
    assert run.dispatch_status == WorkflowRun.DispatchStatus.DISPATCHED


# -------------------------------------------------------------------------
# 4. Recovery of abandoned PUBLISHING state when recovery command crashed
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_abandoned_publishing_reservation_recovered_after_timeout(sample_job, sample_wf):
    now = timezone.now()
    # Crashed in PUBLISHING state past stale threshold
    abandoned_run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.PUBLISHING,
        dispatch_attempts=1,
        started_at=now - timedelta(minutes=25),
        created_at=now - timedelta(minutes=30),
        step_snapshot=sample_wf.steps,
    )

    with patch(
        "flowforge.workflows.management.commands.recover_workflow_runs.execute_workflow_run_task.delay"
    ) as mock_delay:
        call_command("recover_workflow_runs", stale_minutes=15, max_attempts=5, ignore_backoff=True)

    assert mock_delay.call_count == 1
    abandoned_run.refresh_from_db()
    assert abandoned_run.dispatch_attempts == 2
    assert abandoned_run.dispatch_status == WorkflowRun.DispatchStatus.DISPATCHED
