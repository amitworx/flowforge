"""Integration tests for WorkflowRun result export (download) API endpoint:
GET /api/v1/workflow-runs/{uuid}/download/?format=json|csv
"""

import csv
import io
import json
import uuid
from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
from flowforge.workflows.services import execute_workflow_run
from flowforge.workflows.tasks import execute_workflow_run_task


@pytest.fixture
def test_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="testuser", password="password123")


@pytest.fixture
def other_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="otheruser", password="password123")


@pytest.fixture
def auth_client(test_user: User) -> APIClient:
    client = APIClient()
    client.force_authenticate(user=test_user)
    return client


@pytest.fixture
def other_client(other_user: User) -> APIClient:
    client = APIClient()
    client.force_authenticate(user=other_user)
    return client


@pytest.fixture
def sample_csv_job(test_user: User) -> ProcessingJob:
    return ProcessingJob.objects.create(
        owner=test_user,
        original_filename="customers.csv",
        file_type="csv",
        state=ProcessingJob.State.COMPLETED,
        input_row_count=2,
        output_row_count=2,
        column_names=["customer_id", "name", "zip_code"],
        output_data=[
            {"customer_id": "001", "name": "Alice", "zip_code": "01234"},
            {"customer_id": "002", "name": "Bob", "zip_code": "05678"},
        ],
    )


@pytest.fixture
def sample_export_workflow(test_user: User) -> WorkflowDefinition:
    return WorkflowDefinition.objects.create(
        owner=test_user,
        name="Customer Zip Filter",
        steps=[
            {"op": "select_columns", "columns": ["customer_id", "zip_code"]},
            {"op": "rename_columns", "mapping": {"zip_code": "postal_code"}},
        ],
    )


@pytest.mark.django_db
def test_download_endpoint_requires_authentication(
    test_user, sample_csv_job, sample_export_workflow
):
    run = execute_workflow_run(sample_export_workflow, sample_csv_job, test_user)
    anon_client = APIClient()
    response = anon_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=json")
    assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)


@pytest.mark.django_db
def test_download_endpoint_cross_user_and_nonexistent_returns_404(
    auth_client, other_client, test_user, sample_csv_job, sample_export_workflow
):
    run = execute_workflow_run(sample_export_workflow, sample_csv_job, test_user)

    # Cross user access returns 404
    resp_cross = other_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=json")
    assert resp_cross.status_code == status.HTTP_404_NOT_FOUND
    assert resp_cross.json()["detail"] == "Workflow run not found."

    # Nonexistent UUID returns identical 404
    resp_nonexistent = auth_client.get(
        f"/api/v1/workflow-runs/{uuid.uuid4()}/download/?format=json"
    )
    assert resp_nonexistent.status_code == status.HTTP_404_NOT_FOUND
    assert resp_nonexistent.json()["detail"] == "Workflow run not found."


@pytest.mark.django_db
def test_download_endpoint_non_completed_states_return_409(
    auth_client, test_user, sample_csv_job, sample_export_workflow
):
    # Queued run
    queued_run = WorkflowRun.objects.create(
        owner=test_user,
        workflow_definition=sample_export_workflow,
        source_job=sample_csv_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_export_workflow.steps,
    )
    resp_queued = auth_client.get(f"/api/v1/workflow-runs/{queued_run.id}/download/?format=json")
    assert resp_queued.status_code == status.HTTP_409_CONFLICT
    assert "queued" in resp_queued.json()["detail"].lower()

    # Running run
    running_run = WorkflowRun.objects.create(
        owner=test_user,
        workflow_definition=sample_export_workflow,
        source_job=sample_csv_job,
        state=WorkflowRun.State.RUNNING,
        step_snapshot=sample_export_workflow.steps,
    )
    resp_running = auth_client.get(f"/api/v1/workflow-runs/{running_run.id}/download/?format=csv")
    assert resp_running.status_code == status.HTTP_409_CONFLICT
    assert "running" in resp_running.json()["detail"].lower()

    # Failed run
    failed_run = WorkflowRun.objects.create(
        owner=test_user,
        workflow_definition=sample_export_workflow,
        source_job=sample_csv_job,
        state=WorkflowRun.State.FAILED,
        step_snapshot=sample_export_workflow.steps,
        error_summary="Failed run",
    )
    resp_failed = auth_client.get(f"/api/v1/workflow-runs/{failed_run.id}/download/?format=json")
    assert resp_failed.status_code == status.HTTP_409_CONFLICT
    assert "failed" in resp_failed.json()["detail"].lower()


@pytest.mark.django_db
def test_download_endpoint_invalid_format_returns_400(
    auth_client, test_user, sample_csv_job, sample_export_workflow
):
    run = execute_workflow_run(sample_export_workflow, sample_csv_job, test_user)
    resp = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=xml")
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "format" in resp.json()["detail"].lower()


@pytest.mark.django_db
def test_download_endpoint_json_export_success(
    auth_client, test_user, sample_csv_job, sample_export_workflow
):
    run = execute_workflow_run(sample_export_workflow, sample_csv_job, test_user)
    resp = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=json")

    assert resp.status_code == status.HTTP_200_OK
    assert resp["Content-Type"] == "application/json; charset=utf-8"
    assert resp["Content-Disposition"] == f'attachment; filename="workflow-run-{run.id}.json"'

    data = json.loads(resp.content.decode("utf-8"))
    assert len(data) == 2
    assert data[0] == {"customer_id": "001", "postal_code": "01234"}
    assert data[1] == {"customer_id": "002", "postal_code": "05678"}


@pytest.mark.django_db
def test_download_endpoint_csv_export_success(
    auth_client, test_user, sample_csv_job, sample_export_workflow
):
    run = execute_workflow_run(sample_export_workflow, sample_csv_job, test_user)
    resp = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=csv")

    assert resp.status_code == status.HTTP_200_OK
    assert resp["Content-Type"] == "text/csv; charset=utf-8"
    assert resp["Content-Disposition"] == f'attachment; filename="workflow-run-{run.id}.csv"'

    rows = list(csv.reader(io.StringIO(resp.content.decode("utf-8"))))
    assert len(rows) == 3
    assert rows[0] == ["customer_id", "postal_code"]
    assert rows[1] == ["001", "01234"]
    assert rows[2] == ["002", "05678"]


@pytest.mark.django_db
def test_download_endpoint_empty_results_outputs_header_only_csv(
    auth_client, test_user, sample_csv_job
):
    # Workflow that filters out all rows
    wf = WorkflowDefinition.objects.create(
        owner=test_user,
        name="Filter None",
        steps=[
            {"op": "select_columns", "columns": ["customer_id", "name"]},
            {"op": "filter_equals", "column": "name", "value": "NonexistentPerson"},
        ],
    )
    run = execute_workflow_run(wf, sample_csv_job, test_user)
    assert run.state == WorkflowRun.State.COMPLETED
    assert run.output_row_count == 0

    resp = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=csv")
    assert resp.status_code == status.HTTP_200_OK

    rows = list(csv.reader(io.StringIO(resp.content.decode("utf-8"))))
    assert len(rows) == 1
    assert rows[0] == ["customer_id", "name"]


@pytest.mark.django_db
def test_download_endpoint_async_completed_run_exports_identically(
    auth_client, test_user, sample_csv_job, sample_export_workflow
):
    run = WorkflowRun.objects.create(
        owner=test_user,
        workflow_definition=sample_export_workflow,
        source_job=sample_csv_job,
        state=WorkflowRun.State.QUEUED,
        dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
        step_snapshot=sample_export_workflow.steps,
    )
    execute_workflow_run_task(str(run.id), None)
    run.refresh_from_db()
    assert run.state == WorkflowRun.State.COMPLETED

    resp_csv = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=csv")
    assert resp_csv.status_code == status.HTTP_200_OK
    rows = list(csv.reader(io.StringIO(resp_csv.content.decode("utf-8"))))
    assert len(rows) == 3
    assert rows[0] == ["customer_id", "postal_code"]


@pytest.mark.django_db
def test_download_endpoint_definition_mutation_does_not_affect_export(
    auth_client, test_user, sample_csv_job, sample_export_workflow
):
    run = execute_workflow_run(sample_export_workflow, sample_csv_job, test_user)

    # Mutate the workflow definition after execution
    sample_export_workflow.steps = [{"op": "select_columns", "columns": ["name"]}]
    sample_export_workflow.save()

    resp = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=csv")
    assert resp.status_code == status.HTTP_200_OK
    rows = list(csv.reader(io.StringIO(resp.content.decode("utf-8"))))
    assert rows[0] == ["customer_id", "postal_code"]


@pytest.mark.django_db
def test_download_endpoint_payload_too_large_returns_413(
    auth_client, test_user, sample_csv_job, sample_export_workflow
):
    run = execute_workflow_run(sample_export_workflow, sample_csv_job, test_user)

    # Patch DEFAULT_MAX_EXPORT_SIZE_BYTES to a tiny limit (e.g. 5 bytes) to simulate oversized export
    with patch("flowforge.workflows.views.export_workflow_run_to_json") as mock_json:
        from flowforge.workflows.export import ExportPayloadTooLargeError

        mock_json.side_effect = ExportPayloadTooLargeError(
            "JSON export size exceeds maximum limit of 10485760 bytes."
        )
        resp_json = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=json")
        assert resp_json.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "exceeds" in resp_json.json()["detail"].lower()

    with patch("flowforge.workflows.views.export_workflow_run_to_csv") as mock_csv:
        mock_csv.side_effect = ExportPayloadTooLargeError(
            "CSV export size exceeds maximum limit of 10485760 bytes."
        )
        resp_csv = auth_client.get(f"/api/v1/workflow-runs/{run.id}/download/?format=csv")
        assert resp_csv.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "exceeds" in resp_csv.json()["detail"].lower()
