import uuid
from datetime import timedelta
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db import transaction
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun

User = get_user_model()


@pytest.fixture
def auth_client():
    user = User.objects.create_user(username="m04a_user", password="testpassword123")
    client = APIClient()
    client.force_authenticate(user=user)
    return client, user


@pytest.fixture
def other_user():
    return User.objects.create_user(username="m04a_other", password="testpassword123")


@pytest.fixture
def completed_source_job(auth_client):
    _, user = auth_client
    return ProcessingJob.objects.create(
        owner=user,
        original_filename="sample.csv",
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.COMPLETED,
        column_names=["id", "name", "active"],
        output_data=[
            {"id": "1", "name": "Alice", "active": True},
            {"id": "2", "name": "Bob", "active": False},
        ],
    )


@pytest.fixture
def sample_workflow(auth_client):
    _, user = auth_client
    steps = [
        {"op": "filter_equals", "column": "active", "value": True},
        {"op": "select_columns", "columns": ["id", "name"]},
    ]
    return WorkflowDefinition.objects.create(
        owner=user,
        name="Filter and Select Active",
        steps=steps,
    )


# 1. Transaction rollback guarantees no task is published
@pytest.mark.django_db(transaction=True)
def test_transaction_rollback_prevents_task_dispatch(
    auth_client, completed_source_job, sample_workflow
):
    client, _ = auth_client
    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    payload = {"source_job_id": str(completed_source_job.id)}

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        try:
            with transaction.atomic():
                response = client.post(url, payload, format="json")
                assert response.status_code == status.HTTP_202_ACCEPTED
                # Force rollback of the enclosing transaction
                raise RuntimeError("Simulated transaction failure after endpoint response")
        except RuntimeError:
            pass

        # Since transaction rolled back, task must never be published
        mock_delay.assert_not_called()
        assert not WorkflowRun.objects.filter(workflow_definition=sample_workflow).exists()


# 2. Broker publish failure leaves recoverable state in DB
@pytest.mark.django_db(transaction=True)
def test_broker_publish_failure_handling(auth_client, completed_source_job, sample_workflow):
    client, _ = auth_client
    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    payload = {"source_job_id": str(completed_source_job.id)}

    with patch(
        "flowforge.workflows.tasks.execute_workflow_run_task.delay",
        side_effect=Exception("Redis connection refused: [Errno 111]"),
    ):
        response = client.post(url, payload, format="json")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert "id" in data
    assert data["dispatch_status"] == WorkflowRun.DispatchStatus.FAILED
    assert "detail" in data

    # The view caught broker exception and safely persisted FAILED dispatch status
    run = WorkflowRun.objects.get(id=data["id"])
    assert run.state == WorkflowRun.State.QUEUED
    assert run.dispatch_status == WorkflowRun.DispatchStatus.FAILED
    assert "broker unavailable" in (run.error_summary or "").lower()
    # Confirm raw internal exception is sanitized
    assert "Errno 111" not in (run.error_summary or "")


# 3. Worker source ownership validation before execution
@pytest.mark.django_db
def test_worker_rejects_cross_owner_source_job(sample_workflow, other_user):
    from flowforge.workflows.tasks import execute_workflow_run_task

    # Tampered database state: run owned by user, but source_job belongs to other_user
    cross_job = ProcessingJob.objects.create(
        owner=other_user,
        original_filename="secret.csv",
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.COMPLETED,
        column_names=["secret"],
        output_data=[{"secret": "classified"}],
    )
    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=cross_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_workflow.steps,
    )

    execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert "ownership" in (run.error_summary or "").lower()
    assert run.output_data == []
    assert run.step_runs.count() == 0


# 4. Worker non-completed source job validation before execution
@pytest.mark.django_db
def test_worker_rejects_non_completed_source_job(sample_workflow):
    from flowforge.workflows.tasks import execute_workflow_run_task

    pending_job = ProcessingJob.objects.create(
        owner=sample_workflow.owner,
        original_filename="pending.csv",
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.PENDING,
        column_names=["id"],
        output_data=[],
    )
    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=pending_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_workflow.steps,
    )

    execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert "completed" in (run.error_summary or "").lower()
    assert run.output_data == []


# 5. Worker invalid snapshot structure validation before execution
@pytest.mark.django_db
def test_worker_rejects_invalid_step_snapshot_structure(completed_source_job, sample_workflow):
    from flowforge.workflows.tasks import execute_workflow_run_task

    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot="not-a-list",
    )

    execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert "snapshot" in (run.error_summary or "").lower()


# 6. Worker unexpected exception sanitizes error_summary in API
@pytest.mark.django_db
def test_worker_unexpected_exception_sanitizes_api_error_summary(
    completed_source_job, sample_workflow
):
    from flowforge.workflows.tasks import execute_workflow_run_task

    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_workflow.steps,
    )

    with patch(
        "flowforge.workflows.tasks.execute_workflow_pipeline",
        side_effect=RuntimeError(
            "psycopg.OperationalError: server closed the connection unexpectedly password=supersecret"
        ),
    ):
        execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    # Check that raw traceback or secret details are NOT present
    assert "supersecret" not in (run.error_summary or "")
    assert "psycopg" not in (run.error_summary or "")
    # Check that error_summary is a safe generic message
    assert run.error_summary == "Internal workflow execution error occurred."


# 7. Worker fencing token prevents stale worker from overwriting completed run
@pytest.mark.django_db
def test_worker_claim_fencing_prevents_stale_takeover(completed_source_job, sample_workflow):
    from flowforge.workflows.tasks import execute_workflow_run_task

    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.RUNNING,
        claim_token=uuid.uuid4(),
        started_at=timezone.now() - timedelta(minutes=20),
        step_snapshot=sample_workflow.steps,
    )

    # Invoking task with a different / stale claim token cannot overwrite or proceed
    stale_token = uuid.uuid4()
    execute_workflow_run_task(str(run.id), claim_token=str(stale_token))

    run.refresh_from_db()
    # Still running with original token, not completed by the stale task
    assert run.state == WorkflowRun.State.RUNNING
    assert run.step_runs.count() == 0


# 8. Recovery management command recovers stranded dispatch_failed runs
@pytest.mark.django_db
def test_recovery_command_recovers_dispatch_failed_runs(completed_source_job, sample_workflow):
    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
        dispatch_attempts=1,
        step_snapshot=sample_workflow.steps,
    )

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        call_command("recover_workflow_runs", "--limit=10")
        mock_delay.assert_called_once_with(
            str(run.id), str(run.claim_token) if run.claim_token else None
        )

    run.refresh_from_db()
    assert run.dispatch_status == WorkflowRun.DispatchStatus.DISPATCHED
    assert run.dispatch_attempts == 2


# 9. Recovery management command reclaims timed-out running runs
@pytest.mark.django_db
def test_recovery_command_reclaims_stale_running_runs(completed_source_job, sample_workflow):
    old_time = timezone.now() - timedelta(minutes=30)
    original_token = uuid.uuid4()
    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.RUNNING,
        claim_token=original_token,
        started_at=old_time,
        step_snapshot=sample_workflow.steps,
    )

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        call_command("recover_workflow_runs", "--stale-minutes=15", "--action=requeue")
        mock_delay.assert_called_once()

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.QUEUED
    # Must have rotated the claim token to fence out the old worker
    assert run.claim_token != original_token
