import uuid
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStepRun

User = get_user_model()


@pytest.fixture
def auth_client():
    user = User.objects.create_user(username="asyncuser", password="testpassword123")
    client = APIClient()
    client.force_authenticate(user=user)
    return client, user


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


@pytest.mark.django_db
def test_async_workflow_run_endpoint_returns_202_and_queued(
    auth_client, completed_source_job, sample_workflow
):
    client, user = auth_client
    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    payload = {"source_job_id": str(completed_source_job.id)}

    response = client.post(url, payload, format="json")

    assert response.status_code == status.HTTP_202_ACCEPTED
    data = response.json()
    assert "id" in data
    assert data["state"] == "queued"
    assert data["workflow_definition"] == str(sample_workflow.id)
    assert data["source_job"] == str(completed_source_job.id)
    assert data["status_url"] == f"/api/v1/workflow-runs/{data['id']}/"

    # Verify run exists in DB with queued state
    run = WorkflowRun.objects.get(id=data["id"])
    assert run.state == WorkflowRun.State.QUEUED
    assert run.step_snapshot == sample_workflow.steps


@pytest.mark.django_db
def test_async_workflow_run_unauthenticated(completed_source_job, sample_workflow):
    client = APIClient()
    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    payload = {"source_job_id": str(completed_source_job.id)}

    response = client.post(url, payload, format="json")
    assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)


@pytest.mark.django_db
def test_async_workflow_run_cross_user_isolation(auth_client, sample_workflow):
    client, _ = auth_client
    other_user = User.objects.create_user(username="otheruser", password="password")
    other_job = ProcessingJob.objects.create(
        owner=other_user,
        original_filename="other.csv",
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.COMPLETED,
        column_names=["id"],
        output_data=[{"id": 1}],
    )

    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    response = client.post(url, {"source_job_id": str(other_job.id)}, format="json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.django_db
def test_async_workflow_run_pending_source_job_rejected(auth_client, sample_workflow):
    client, user = auth_client
    pending_job = ProcessingJob.objects.create(
        owner=user,
        original_filename="pending.csv",
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.PENDING,
        column_names=["id"],
        output_data=[],
    )

    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    response = client.post(url, {"source_job_id": str(pending_job.id)}, format="json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "completed" in response.json().get("detail", "").lower()


@pytest.mark.django_db
def test_async_workflow_run_nonexistent_workflow_returns_404(auth_client, completed_source_job):
    client, _ = auth_client
    url = reverse("workflow-async-run-create", kwargs={"uuid": uuid.uuid4()})
    response = client.post(url, {"source_job_id": str(completed_source_job.id)}, format="json")
    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.django_db
def test_async_workflow_task_execution(completed_source_job, sample_workflow):
    from flowforge.workflows.tasks import execute_workflow_run_task

    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_workflow.steps,
    )

    execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.COMPLETED
    assert run.input_row_count == 2
    assert run.output_row_count == 1
    assert run.output_data == [{"id": "1", "name": "Alice"}]
    assert run.started_at is not None
    assert run.finished_at is not None
    assert run.finished_at >= run.started_at

    # Audit records created
    step_runs = list(run.step_runs.order_by("step_index"))
    assert len(step_runs) == 2
    assert step_runs[0].operation == "filter_equals"
    assert step_runs[0].status == WorkflowStepRun.Status.COMPLETED
    assert step_runs[0].output_row_count == 1
    assert step_runs[1].operation == "select_columns"
    assert step_runs[1].status == WorkflowStepRun.Status.COMPLETED
    assert step_runs[1].output_row_count == 1


@pytest.mark.django_db
def test_async_workflow_duplicate_task_delivery_is_idempotent(
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

    # First delivery
    execute_workflow_run_task(str(run.id))
    run.refresh_from_db()
    assert run.state == WorkflowRun.State.COMPLETED
    initial_step_count = run.step_runs.count()

    # Second delivery (at-least-once Celery delivery)
    execute_workflow_run_task(str(run.id))
    run.refresh_from_db()
    assert run.state == WorkflowRun.State.COMPLETED
    assert run.step_runs.count() == initial_step_count


@pytest.mark.django_db
def test_async_workflow_definition_mutation_does_not_affect_queued_run(
    completed_source_job, sample_workflow
):
    from flowforge.workflows.tasks import execute_workflow_run_task

    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=list(sample_workflow.steps),
    )

    # Mutate the definition before the worker picks up the run
    sample_workflow.steps = [{"op": "drop_columns", "columns": ["name"]}]
    sample_workflow.save()

    execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.COMPLETED
    # Must have executed the snapshot (filter_equals + select_columns)
    assert run.output_data == [{"id": "1", "name": "Alice"}]
    step_runs = list(run.step_runs.order_by("step_index"))
    assert len(step_runs) == 2
    assert step_runs[0].operation == "filter_equals"
    assert step_runs[1].operation == "select_columns"


@pytest.mark.django_db
def test_async_workflow_task_failure_recording(completed_source_job, sample_workflow):
    from flowforge.workflows.tasks import execute_workflow_run_task

    steps = [
        {"op": "filter_equals", "column": "active", "value": True},
        {"op": "select_columns", "columns": ["non_existent_column"]},
    ]
    run = WorkflowRun.objects.create(
        owner=sample_workflow.owner,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=steps,
    )

    execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED
    assert "non_existent_column" in (run.error_summary or "")
    assert run.output_data == []
    assert run.output_row_count == 0

    # Step 0 succeeded, Step 1 failed
    step_runs = list(run.step_runs.order_by("step_index"))
    assert len(step_runs) == 2
    assert step_runs[0].status == WorkflowStepRun.Status.COMPLETED
    assert step_runs[1].status == WorkflowStepRun.Status.FAILED
    assert "non_existent_column" in (step_runs[1].error_summary or "")


@pytest.mark.django_db(transaction=True)
def test_async_workflow_transaction_on_commit_dispatch(
    auth_client, completed_source_job, sample_workflow
):
    client, _ = auth_client
    url = reverse("workflow-async-run-create", kwargs={"uuid": sample_workflow.id})
    payload = {"source_job_id": str(completed_source_job.id)}

    with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
        response = client.post(url, payload, format="json")
        assert response.status_code == status.HTTP_202_ACCEPTED
        mock_delay.assert_called_once()
        call_run_id = mock_delay.call_args[0][0]
        assert str(response.json()["id"]) == str(call_run_id)


@pytest.mark.django_db
def test_async_workflow_polling_status(auth_client, completed_source_job, sample_workflow):
    from flowforge.workflows.tasks import execute_workflow_run_task

    client, user = auth_client
    run = WorkflowRun.objects.create(
        owner=user,
        workflow_definition=sample_workflow,
        source_job=completed_source_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_workflow.steps,
    )

    # Poll status while queued
    poll_url = reverse("workflow-run-detail", kwargs={"uuid": run.id})
    poll_res = client.get(poll_url)
    assert poll_res.status_code == status.HTTP_200_OK
    assert poll_res.json()["state"] == "queued"
    assert poll_res.json()["output_data"] == []

    # Execute task
    execute_workflow_run_task(str(run.id))

    # Poll status after completion
    poll_res_after = client.get(poll_url)
    assert poll_res_after.status_code == status.HTTP_200_OK
    assert poll_res_after.json()["state"] == "completed"
    assert len(poll_res_after.json()["step_runs"]) == 2
    assert len(poll_res_after.json()["output_data"]) == 1
