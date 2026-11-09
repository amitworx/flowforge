import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def test_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="testuser", password="password123")


@pytest.fixture
def other_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="otheruser", password="password123")


@pytest.fixture
def sample_job(test_user: User) -> ProcessingJob:
    return ProcessingJob.objects.create(
        owner=test_user,
        file_type="csv",
        state=ProcessingJob.State.COMPLETED,
        original_filename="users.csv",
        input_row_count=3,
        output_row_count=3,
        column_names=["id", "name", "status", "dept"],
        output_data=[
            {"id": "001", "name": "Alice", "status": "active", "dept": "eng"},
            {"id": "002", "name": "Bob", "status": "inactive", "dept": "hr"},
            {"id": "003", "name": "Charlie", "status": "active", "dept": "eng"},
        ],
    )


@pytest.mark.django_db
def test_workflow_endpoints_require_authentication(api_client):
    assert api_client.get("/api/v1/workflows/").status_code == status.HTTP_403_FORBIDDEN
    assert api_client.post("/api/v1/workflows/").status_code == status.HTTP_403_FORBIDDEN
    assert api_client.get("/api/v1/workflow-runs/").status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.django_db
def test_create_and_list_workflow_definition(api_client, test_user):
    api_client.force_authenticate(user=test_user)

    payload = {
        "name": "User Filter Workflow",
        "steps": [
            {"op": "select_columns", "columns": ["id", "name", "status"]},
            {"op": "rename_columns", "mapping": {"name": "user_name"}},
            {"op": "filter_equals", "column": "status", "value": "active"},
        ],
    }
    response = api_client.post("/api/v1/workflows/", data=payload, format="json")
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == "User Filter Workflow"
    assert len(data["steps"]) == 3
    workflow_id = data["id"]

    # List
    list_res = api_client.get("/api/v1/workflows/")
    assert list_res.status_code == status.HTTP_200_OK
    assert list_res.json()["count"] == 1

    # Retrieve
    get_res = api_client.get(f"/api/v1/workflows/{workflow_id}/")
    assert get_res.status_code == status.HTTP_200_OK
    assert get_res.json()["id"] == workflow_id


@pytest.mark.django_db
def test_invalid_workflow_definition_rejected(api_client, test_user):
    api_client.force_authenticate(user=test_user)

    # Empty steps
    res = api_client.post("/api/v1/workflows/", data={"name": "Empty", "steps": []}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Unsupported operation
    res = api_client.post(
        "/api/v1/workflows/",
        data={"name": "Bad Op", "steps": [{"op": "unsupported"}]},
        format="json",
    )
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Duplicate column in select
    res = api_client.post(
        "/api/v1/workflows/",
        data={"name": "Dup", "steps": [{"op": "select_columns", "columns": ["a", "a"]}]},
        format="json",
    )
    assert res.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.django_db
def test_execute_workflow_run_successful_chain(api_client, test_user, sample_job):
    api_client.force_authenticate(user=test_user)

    # Create definition
    wf = WorkflowDefinition.objects.create(
        owner=test_user,
        name="Filter Active",
        steps=[
            {"op": "select_columns", "columns": ["id", "name", "status"]},
            {"op": "rename_columns", "mapping": {"name": "account_name"}},
            {"op": "filter_equals", "column": "status", "value": "active"},
        ],
    )

    run_res = api_client.post(
        f"/api/v1/workflows/{wf.id}/runs/",
        data={"source_job_id": str(sample_job.id)},
        format="json",
    )
    assert run_res.status_code == status.HTTP_201_CREATED
    run_data = run_res.json()
    assert run_data["state"] == "completed"
    assert run_data["input_row_count"] == 3
    assert run_data["output_row_count"] == 2
    run_id = run_data["id"]

    # Retrieve detailed run
    detail_res = api_client.get(f"/api/v1/workflow-runs/{run_id}/")
    assert detail_res.status_code == status.HTTP_200_OK
    detail = detail_res.json()
    assert detail["state"] == "completed"
    assert len(detail["output_data"]) == 2
    # Preserves leading zeros!
    assert detail["output_data"][0]["id"] == "001"
    assert detail["output_data"][0]["account_name"] == "Alice"
    assert detail["output_data"][1]["id"] == "003"
    assert detail["output_data"][1]["account_name"] == "Charlie"

    # Step audit records
    steps = detail["step_runs"]
    assert len(steps) == 3
    assert steps[0]["operation"] == "select_columns"
    assert steps[0]["status"] == "completed"
    assert steps[1]["operation"] == "rename_columns"
    assert steps[2]["operation"] == "filter_equals"

    # Original job data unmodified
    sample_job.refresh_from_db()
    assert len(sample_job.output_data) == 3
    assert "dept" in sample_job.output_data[0]


@pytest.mark.django_db
def test_execute_workflow_run_failure_audit_durability(api_client, test_user, sample_job):
    api_client.force_authenticate(user=test_user)

    wf = WorkflowDefinition.objects.create(
        owner=test_user,
        name="Faulty Step 2",
        steps=[
            {"op": "select_columns", "columns": ["id", "name"]},  # Step 0 succeeds
            {"op": "rename_columns", "mapping": {"missing_col": "new_col"}},  # Step 1 fails
            {"op": "filter_equals", "column": "id", "value": "001"},  # Step 2 never reached
        ],
    )

    run_res = api_client.post(
        f"/api/v1/workflows/{wf.id}/runs/",
        data={"source_job_id": str(sample_job.id)},
        format="json",
    )
    assert run_res.status_code == status.HTTP_201_CREATED
    run_data = run_res.json()
    assert run_data["state"] == "failed"
    assert "error_summary" in run_data
    run_id = run_data["id"]

    detail_res = api_client.get(f"/api/v1/workflow-runs/{run_id}/")
    detail = detail_res.json()
    assert detail["state"] == "failed"
    assert detail["output_data"] == []  # No partial output marked as completed
    assert len(detail["step_runs"]) == 2  # Step 0 completed, step 1 failed
    assert detail["step_runs"][0]["status"] == "completed"
    assert detail["step_runs"][1]["status"] == "failed"
    assert "missing_col" in detail["step_runs"][1]["error_summary"]


@pytest.mark.django_db
def test_cross_user_isolation(api_client, test_user, other_user, sample_job):
    # test_user owns wf and sample_job
    wf = WorkflowDefinition.objects.create(
        owner=test_user,
        name="Private WF",
        steps=[{"op": "select_columns", "columns": ["id"]}],
    )

    # other_user attempts to retrieve test_user's workflow
    api_client.force_authenticate(user=other_user)
    assert api_client.get(f"/api/v1/workflows/{wf.id}/").status_code == status.HTTP_404_NOT_FOUND

    # other_user attempts to run test_user's workflow
    res = api_client.post(
        f"/api/v1/workflows/{wf.id}/runs/",
        data={"source_job_id": str(sample_job.id)},
        format="json",
    )
    assert res.status_code == status.HTTP_404_NOT_FOUND

    # other_user attempts to run their own workflow against test_user's job
    other_wf = WorkflowDefinition.objects.create(
        owner=other_user,
        name="Other WF",
        steps=[{"op": "select_columns", "columns": ["id"]}],
    )
    res2 = api_client.post(
        f"/api/v1/workflows/{other_wf.id}/runs/",
        data={"source_job_id": str(sample_job.id)},
        format="json",
    )
    assert res2.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.django_db
def test_workflow_definition_immutability_and_execution_snapshot(api_client, test_user, sample_job):
    api_client.force_authenticate(user=test_user)

    wf = WorkflowDefinition.objects.create(
        owner=test_user,
        name="Original Snapshot",
        steps=[{"op": "select_columns", "columns": ["id", "name"]}],
    )

    run_res = api_client.post(
        f"/api/v1/workflows/{wf.id}/runs/",
        data={"source_job_id": str(sample_job.id)},
        format="json",
    )
    run_id = run_res.json()["id"]

    # Directly mutate the workflow definition steps in DB
    wf.steps = [{"op": "select_columns", "columns": ["name"]}]
    wf.save()

    # Verify that historical run retains the exact step snapshot executed
    detail = api_client.get(f"/api/v1/workflow-runs/{run_id}/").json()
    assert detail["step_snapshot"] == [{"op": "select_columns", "columns": ["id", "name"]}]
