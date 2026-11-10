import uuid
from datetime import timedelta
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun

User = get_user_model()


@pytest.fixture
def auth_client():
    user = User.objects.create_user(username="m04b_user", password="testpassword123")
    client = APIClient()
    client.force_authenticate(user=user)
    return client, user


@pytest.fixture
def other_user():
    return User.objects.create_user(username="m04b_other", password="testpassword123")


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


# 1. Direct Broker publish failure returns HTTP 503 with recoverable run metadata
@pytest.mark.django_db(transaction=True)
def test_direct_broker_publish_failure_returns_503(
    auth_client, completed_source_job, sample_workflow
):
    client, _ = auth_client
    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    payload = {"source_job_id": str(completed_source_job.id)}

    with patch(
        "flowforge.workflows.tasks.execute_workflow_run_task.delay",
        side_effect=Exception("Redis connection refused: [Errno 111] password=secret"),
    ):
        response = client.post(url, payload, format="json")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert "id" in data
    assert data["state"] == "queued"
    assert data["dispatch_status"] == "failed"
    assert "broker unavailable" in data["detail"].lower()
    # Ensure raw secret or exception string is not exposed
    assert "secret" not in data["detail"]
    assert "Errno 111" not in data["detail"]

    run = WorkflowRun.objects.get(id=data["id"])
    assert run.state == WorkflowRun.State.QUEUED
    assert run.dispatch_status == WorkflowRun.DispatchStatus.FAILED
    assert "broker unavailable" in (run.error_summary or "").lower()
    assert "secret" not in (run.error_summary or "")


# 2. Bounded recovery attempts: runs exceeding max_attempts transition to terminal FAILED state
@pytest.mark.django_db
def test_recovery_bounded_attempts_marks_terminal_failed(completed_source_job, sample_workflow):
    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
        dispatch_attempts=3,  # Already hit max attempts
        step_snapshot=sample_workflow.steps,
    )

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        call_command("recover_workflow_runs", "--max-attempts=3", "--limit=10")
        mock_delay.assert_not_called()

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert run.dispatch_status == WorkflowRun.DispatchStatus.FAILED
    assert "recovery attempts exhausted" in (run.error_summary or "").lower()
    assert run.finished_at is not None


# 3. Recovery publish failure leaves run cleanly in dispatch_status=FAILED with generic error
@pytest.mark.django_db
def test_recovery_publish_failure_persists_clean_failure(completed_source_job, sample_workflow):
    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
        dispatch_attempts=1,
        step_snapshot=sample_workflow.steps,
    )

    with patch(
        "flowforge.workflows.tasks.execute_workflow_run_task.delay",
        side_effect=Exception("Redis publish error: connection timeout password=supersecret"),
    ):
        call_command("recover_workflow_runs", "--max-attempts=3", "--limit=10")

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.QUEUED
    assert run.dispatch_status == WorkflowRun.DispatchStatus.FAILED
    assert run.dispatch_attempts == 2
    # Check that error_summary is sanitized and does not leak password or internal exception
    assert "supersecret" not in (run.error_summary or "")
    assert "broker unavailable" in (run.error_summary or "").lower()


# 4. Stale running recovery handles Redis publish failure by maintaining atomic reclaim
@pytest.mark.django_db
def test_stale_running_recovery_publish_failure(completed_source_job, sample_workflow):
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

    with patch(
        "flowforge.workflows.tasks.execute_workflow_run_task.delay",
        side_effect=Exception("Redis down during reclaim password=topsecret"),
    ):
        call_command("recover_workflow_runs", "--stale-minutes=15", "--action=requeue")

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.QUEUED
    assert run.dispatch_status == WorkflowRun.DispatchStatus.FAILED
    assert "topsecret" not in (run.error_summary or "")
    assert "broker unavailable" in (run.error_summary or "").lower()


# 5. Concurrent workers: legitimate running worker cannot be stolen by another worker
@pytest.mark.django_db
def test_legitimately_running_worker_cannot_be_claimed_by_second_worker(
    completed_source_job, sample_workflow
):
    from flowforge.workflows.tasks import execute_workflow_run_task

    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.RUNNING,
        claim_token=uuid.uuid4(),
        started_at=timezone.now(),  # Freshly started, NOT stale
        step_snapshot=sample_workflow.steps,
    )
    initial_claim_token = run.claim_token

    # A second worker tries to claim the run without claim_token (or with random token)
    execute_workflow_run_task(str(run.id), claim_token=None)

    run.refresh_from_db()
    # The running run was NOT stolen; claim_token remains unchanged
    assert run.claim_token == initial_claim_token
    assert run.state == WorkflowRun.State.RUNNING
