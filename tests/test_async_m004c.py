import threading
import time
from datetime import timedelta
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.utils import timezone

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun

User = get_user_model()


@pytest.fixture
def test_user(db):  # noqa: ARG001
    user, _ = User.objects.get_or_create(
        username="m04c_user", defaults={"email": "m04c@example.com"}
    )
    return user


@pytest.fixture
def sample_job(test_user):
    return ProcessingJob.objects.create(
        owner=test_user,
        original_filename="sample.csv",
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.COMPLETED,
        column_names=["city", "temp"],
        output_data=[{"city": "Paris", "temp": "20"}],
    )


@pytest.fixture
def sample_wf(test_user):
    return WorkflowDefinition.objects.create(
        owner=test_user,
        name="Sample Pipeline",
        steps=[{"op": "select_columns", "columns": ["city"]}],
    )


# 1. Parallel recovery invocation test using distinct threads/connections
@pytest.mark.django_db(transaction=True)
def test_parallel_recovery_commands_cannot_both_reserve_same_run(sample_job, sample_wf):
    run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
        dispatch_attempts=1,
        step_snapshot=sample_wf.steps,
    )

    dispatches = []

    def fake_delay(*args, **_kwargs):
        dispatches.append(args)
        time.sleep(0.05)

    with (
        patch(
            "flowforge.workflows.tasks.execute_workflow_run_task.delay",
            side_effect=fake_delay,
        ),
        patch(
            "flowforge.workflows.management.commands.recover_workflow_runs.execute_workflow_run_task.delay",
            side_effect=fake_delay,
        ),
    ):

        def run_cmd():
            connection.close()
            call_command("recover_workflow_runs", max_attempts=5, ignore_backoff=True)
            connection.close()

        t1 = threading.Thread(target=run_cmd)
        t2 = threading.Thread(target=run_cmd)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

    # Only one recovery command should have reserved and dispatched the run in this round
    assert len(dispatches) == 1
    run.refresh_from_db()
    assert run.dispatch_attempts == 2


# 2. Total delivery / recovery budget exhausted by repeated stale-running reclaims
@pytest.mark.django_db
def test_stale_running_recovery_exhausts_total_attempt_budget(sample_job, sample_wf):
    now = timezone.now()
    # Run has had 3 dispatch attempts and is RUNNING past stale threshold
    run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.RUNNING,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        dispatch_attempts=3,
        started_at=now - timedelta(minutes=30),
        step_snapshot=sample_wf.steps,
    )

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        call_command(
            "recover_workflow_runs",
            stale_minutes=15,
            max_attempts=3,
            action="requeue",
            ignore_backoff=True,
        )

    # Should NOT be requeued or dispatched again
    mock_delay.assert_not_called()
    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert "exhausted" in (run.error_summary or "").lower()


# 3. Time-based retry eligibility & exponential backoff
@pytest.mark.django_db
def test_exponential_backoff_and_retry_eligibility(sample_job, sample_wf):
    now = timezone.now()
    # Recently failed dispatch attempt 2 (e.g. 5 seconds ago) -> backoff is 15s
    run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
        dispatch_attempts=2,
        started_at=now - timedelta(seconds=5),  # last attempt was 5s ago (< 15s backoff)
        step_snapshot=sample_wf.steps,
    )

    # Without ignore_backoff: backoff for attempt 2 (15s) should skip it
    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        call_command("recover_workflow_runs", max_attempts=5)

    mock_delay.assert_not_called()
    run.refresh_from_db()
    assert run.dispatch_attempts == 2

    # Simulate time passing beyond backoff threshold (e.g. 1 hour ago)
    WorkflowRun.objects.filter(id=run.id).update(started_at=now - timedelta(hours=1))

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        call_command("recover_workflow_runs", max_attempts=5)

    assert mock_delay.call_count == 1
    run.refresh_from_db()
    assert run.dispatch_attempts == 3


# 4. Stuck queued run with dispatch_status=DISPATCHED recovered after age threshold
@pytest.mark.django_db
def test_stuck_queued_dispatched_recovered_after_age_threshold(sample_job, sample_wf):
    now = timezone.now()
    # Crashed right after view created record or during publish
    stuck_run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        dispatch_attempts=1,
        created_at=now - timedelta(minutes=20),
        step_snapshot=sample_wf.steps,
    )
    # Ensure created_at was set back
    WorkflowRun.objects.filter(id=stuck_run.id).update(created_at=now - timedelta(minutes=20))

    # Fresh queued run (e.g. 5 seconds ago) - in-flight, must NOT be stolen
    fresh_run = WorkflowRun.objects.create(
        owner=sample_wf.owner,
        workflow_definition=sample_wf,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        dispatch_attempts=1,
        created_at=now - timedelta(seconds=5),
        step_snapshot=sample_wf.steps,
    )

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        call_command("recover_workflow_runs", stale_minutes=15, max_attempts=5, ignore_backoff=True)

    # Only stuck_run should have been redispatched, fresh_run untouched
    assert mock_delay.call_count == 1
    call_args = mock_delay.call_args[0]
    assert call_args[0] == str(stuck_run.id)
    fresh_run.refresh_from_db()
    assert fresh_run.dispatch_attempts == 1


# 5. CLI argument validation: positive values for max-attempts, stale-minutes, limit
@pytest.mark.django_db
def test_recover_workflow_runs_validates_positive_arguments():
    with pytest.raises(CommandError, match="must be positive"):
        call_command("recover_workflow_runs", max_attempts=0)

    with pytest.raises(CommandError, match="must be positive"):
        call_command("recover_workflow_runs", stale_minutes=-5)

    with pytest.raises(CommandError, match="must be positive"):
        call_command("recover_workflow_runs", limit=0)
