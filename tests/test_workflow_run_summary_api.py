import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun


@pytest.fixture
def auth_client(django_user_model):
    user = django_user_model.objects.create_user(username="testuser", password="password123")
    client = APIClient()
    client.force_authenticate(user=user)
    client.user = user
    return client


@pytest.fixture
def other_client(django_user_model):
    user = django_user_model.objects.create_user(username="otheruser", password="password123")
    client = APIClient()
    client.force_authenticate(user=user)
    client.user = user
    return client


@pytest.fixture
def sample_steps():
    return [
        {"op": "select_columns", "columns": ["id", "name", "department"]},
        {"op": "filter_equals", "column": "department", "value": "Engineering"},
    ]


@pytest.fixture
def test_data(auth_client, other_client, sample_steps):
    user = auth_client.user
    other_user = other_client.user

    # Source jobs
    job1 = ProcessingJob.objects.create(
        owner=user,
        file_type="csv",
        state=ProcessingJob.State.COMPLETED,
        original_filename="users1.csv",
        input_row_count=100,
        output_row_count=100,
    )
    job2 = ProcessingJob.objects.create(
        owner=user,
        file_type="csv",
        state=ProcessingJob.State.COMPLETED,
        original_filename="users2.csv",
        input_row_count=50,
        output_row_count=50,
    )
    foreign_job = ProcessingJob.objects.create(
        owner=other_user,
        file_type="csv",
        state=ProcessingJob.State.COMPLETED,
        original_filename="foreign.csv",
        input_row_count=200,
        output_row_count=200,
    )

    # Workflows
    wf1 = WorkflowDefinition.objects.create(
        owner=user,
        name="Pipeline 1",
        steps=sample_steps,
    )
    wf2 = WorkflowDefinition.objects.create(
        owner=user,
        name="Pipeline 2 (Archived)",
        steps=sample_steps,
        is_archived=True,
    )
    foreign_wf = WorkflowDefinition.objects.create(
        owner=other_user,
        name="Foreign Pipeline",
        steps=sample_steps,
    )

    t0 = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)

    # User runs
    # 1. Completed run
    run1 = WorkflowRun.objects.create(
        owner=user,
        workflow_definition=wf1,
        source_job=job1,
        state=WorkflowRun.State.COMPLETED,
        step_snapshot=sample_steps,
        input_row_count=100,
        output_row_count=80,
    )
    WorkflowRun.objects.filter(id=run1.id).update(created_at=t0)

    # 2. Failed run
    run2 = WorkflowRun.objects.create(
        owner=user,
        workflow_definition=wf1,
        source_job=job1,
        state=WorkflowRun.State.FAILED,
        step_snapshot=sample_steps,
        input_row_count=100,
        output_row_count=0,
    )
    WorkflowRun.objects.filter(id=run2.id).update(created_at=t0 + timedelta(hours=1))

    # 3. Queued run
    run3 = WorkflowRun.objects.create(
        owner=user,
        workflow_definition=wf2,
        source_job=job2,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_steps,
        input_row_count=50,
        output_row_count=0,
    )
    WorkflowRun.objects.filter(id=run3.id).update(created_at=t0 + timedelta(hours=2))

    # 4. Running run
    run4 = WorkflowRun.objects.create(
        owner=user,
        workflow_definition=wf2,
        source_job=job2,
        state=WorkflowRun.State.RUNNING,
        step_snapshot=sample_steps,
        input_row_count=50,
        output_row_count=0,
    )
    WorkflowRun.objects.filter(id=run4.id).update(created_at=t0 + timedelta(hours=3))

    # 5. Second completed run (on archived wf2)
    run5 = WorkflowRun.objects.create(
        owner=user,
        workflow_definition=wf2,
        source_job=job2,
        state=WorkflowRun.State.COMPLETED,
        step_snapshot=sample_steps,
        input_row_count=50,
        output_row_count=40,
    )
    WorkflowRun.objects.filter(id=run5.id).update(created_at=t0 + timedelta(hours=4))

    # Foreign user run
    foreign_run = WorkflowRun.objects.create(
        owner=other_user,
        workflow_definition=foreign_wf,
        source_job=foreign_job,
        state=WorkflowRun.State.COMPLETED,
        step_snapshot=sample_steps,
        input_row_count=1000,
        output_row_count=1000,
    )
    WorkflowRun.objects.filter(id=foreign_run.id).update(created_at=t0)

    return {
        "user": user,
        "other_user": other_user,
        "job1": job1,
        "job2": job2,
        "foreign_job": foreign_job,
        "wf1": wf1,
        "wf2": wf2,
        "foreign_wf": foreign_wf,
        "run1": run1,
        "run2": run2,
        "run3": run3,
        "run4": run4,
        "run5": run5,
        "foreign_run": foreign_run,
        "t0": t0,
    }


@pytest.mark.django_db
class TestWorkflowRunSummaryAPI:
    def test_anonymous_request_denied(self):
        client = APIClient()
        response = client.get("/api/v1/workflow-runs/summary/")
        assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_empty_database_returns_bounded_schema_with_zeros_and_null_rate(self, auth_client):
        response = auth_client.get("/api/v1/workflow-runs/summary/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data == {
            "total_runs": 0,
            "by_state": {
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
            },
            "completed_input_rows": 0,
            "completed_output_rows": 0,
            "success_rate": None,
        }

    def test_summary_aggregates_and_success_rate_calculation(self, auth_client, test_data):
        _ = test_data
        response = auth_client.get("/api/v1/workflow-runs/summary/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # user has 5 runs: 2 completed, 1 failed, 1 queued, 1 running
        assert data["total_runs"] == 5
        assert data["by_state"] == {
            "queued": 1,
            "running": 1,
            "completed": 2,
            "failed": 1,
        }
        # completed rows sum only from run1 (100, 80) and run5 (50, 40)
        assert data["completed_input_rows"] == 150
        assert data["completed_output_rows"] == 120

        # success_rate: completed / (completed + failed) = 2 / (2 + 1) = 2/3 = 0.67
        assert data["success_rate"] == 0.67

    def test_owner_isolation(self, other_client, test_data):
        _ = test_data
        # other_user has 1 completed run with 1000 input/output rows
        response = other_client.get("/api/v1/workflow-runs/summary/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_runs"] == 1
        assert data["by_state"] == {
            "queued": 0,
            "running": 0,
            "completed": 1,
            "failed": 0,
        }
        assert data["completed_input_rows"] == 1000
        assert data["completed_output_rows"] == 1000
        assert data["success_rate"] == 1.0

    def test_filter_by_workflow_definition(self, auth_client, test_data):
        wf1_id = test_data["wf1"].id
        response = auth_client.get(f"/api/v1/workflow-runs/summary/?workflow_definition={wf1_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # wf1 has 1 completed, 1 failed
        assert data["total_runs"] == 2
        assert data["by_state"] == {
            "queued": 0,
            "running": 0,
            "completed": 1,
            "failed": 1,
        }
        assert data["completed_input_rows"] == 100
        assert data["completed_output_rows"] == 80
        # 1 completed / (1 completed + 1 failed) = 0.5
        assert data["success_rate"] == 0.5

    def test_filter_by_archived_workflow_definition_preserves_history(self, auth_client, test_data):
        wf2_id = test_data["wf2"].id
        response = auth_client.get(f"/api/v1/workflow-runs/summary/?workflow_definition={wf2_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # wf2 has 1 queued, 1 running, 1 completed
        assert data["total_runs"] == 3
        assert data["by_state"] == {
            "queued": 1,
            "running": 1,
            "completed": 1,
            "failed": 0,
        }
        assert data["completed_input_rows"] == 50
        assert data["completed_output_rows"] == 40
        # 1 completed / (1 completed + 0 failed) = 1.0
        assert data["success_rate"] == 1.0

    def test_filter_by_foreign_or_missing_workflow_definition_returns_empty_summary(
        self, auth_client, test_data
    ):
        # 1. Foreign workflow
        foreign_id = test_data["foreign_wf"].id
        res_foreign = auth_client.get(
            f"/api/v1/workflow-runs/summary/?workflow_definition={foreign_id}"
        )
        assert res_foreign.status_code == status.HTTP_200_OK
        assert res_foreign.json()["total_runs"] == 0
        assert res_foreign.json()["success_rate"] is None

        # 2. Missing UUID
        random_id = uuid.uuid4()
        res_missing = auth_client.get(
            f"/api/v1/workflow-runs/summary/?workflow_definition={random_id}"
        )
        assert res_missing.status_code == status.HTTP_200_OK
        assert res_missing.json()["total_runs"] == 0
        assert res_missing.json()["success_rate"] is None

        # Indistinguishable
        assert res_foreign.json() == res_missing.json()

    def test_filter_by_source_job(self, auth_client, test_data):
        job1_id = test_data["job1"].id
        response = auth_client.get(f"/api/v1/workflow-runs/summary/?source_job={job1_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_runs"] == 2
        assert data["by_state"]["completed"] == 1
        assert data["by_state"]["failed"] == 1

    def test_filter_by_foreign_or_missing_source_job_returns_empty_summary(
        self, auth_client, test_data
    ):
        foreign_id = test_data["foreign_job"].id
        res_foreign = auth_client.get(f"/api/v1/workflow-runs/summary/?source_job={foreign_id}")
        assert res_foreign.status_code == status.HTTP_200_OK
        assert res_foreign.json()["total_runs"] == 0

        res_missing = auth_client.get(f"/api/v1/workflow-runs/summary/?source_job={uuid.uuid4()}")
        assert res_missing.status_code == status.HTTP_200_OK
        assert res_missing.json()["total_runs"] == 0

    def test_filter_by_datetime_inclusive_bounds(self, auth_client, test_data):
        t0 = test_data["t0"]
        # Match only run1 and run2
        t_from = t0.isoformat()
        t_to = (t0 + timedelta(hours=1)).isoformat()
        url = f"/api/v1/workflow-runs/summary/?created_from={t_from}&created_to={t_to}"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_runs"] == 2
        assert data["by_state"]["completed"] == 1
        assert data["by_state"]["failed"] == 1
        assert data["completed_input_rows"] == 100

    def test_filter_by_datetime_with_timezone_offset(self, auth_client, test_data):
        _ = test_data
        # t0 is 2026-11-01T12:00:00Z -> In +05:30: 2026-11-01T17:30:00+05:30
        t_from_offset = "2026-11-01T17:30:00+05:30"
        t_to_offset = "2026-11-01T18:30:00+05:30"
        url = (
            f"/api/v1/workflow-runs/summary/?created_from={t_from_offset}&created_to={t_to_offset}"
        )
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_runs"] == 2

    def test_reject_unsupported_query_params_with_400(self, auth_client):
        # state is allowed on list endpoint, but MUST be rejected on summary
        for bad_param in [
            "state=completed",
            "page=1",
            "page_size=10",
            "ordering=-created_at",
            "unknown_param=foo",
        ]:
            response = auth_client.get(f"/api/v1/workflow-runs/summary/?{bad_param}")
            assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_reject_naive_datetime_with_400(self, auth_client):
        url = "/api/v1/workflow-runs/summary/?created_from=2026-11-01T12:00:00"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "created_from" in response.json()

    def test_reject_inverted_datetime_range_with_400(self, auth_client):
        url = (
            "/api/v1/workflow-runs/summary/"
            "?created_from=2026-11-02T12:00:00Z&created_to=2026-11-01T12:00:00Z"
        )
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "created_from" in response.json()

    def test_reject_malformed_uuid_with_400(self, auth_client):
        url = "/api/v1/workflow-runs/summary/?workflow_definition=not-a-valid-uuid"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_success_rate_rounding_and_terminal_denominator(
        self, auth_client, test_data, sample_steps
    ):
        user = test_data["user"]
        wf = test_data["wf1"]
        job = test_data["job1"]

        # Clear existing runs for user
        WorkflowRun.objects.filter(owner=user).delete()

        # 1 completed, 2 failed -> 1 / (1 + 2) = 1/3 = 0.33
        WorkflowRun.objects.create(
            owner=user,
            workflow_definition=wf,
            source_job=job,
            state=WorkflowRun.State.COMPLETED,
            step_snapshot=sample_steps,
        )
        WorkflowRun.objects.create(
            owner=user,
            workflow_definition=wf,
            source_job=job,
            state=WorkflowRun.State.FAILED,
            step_snapshot=sample_steps,
        )
        WorkflowRun.objects.create(
            owner=user,
            workflow_definition=wf,
            source_job=job,
            state=WorkflowRun.State.FAILED,
            step_snapshot=sample_steps,
        )
        # Add queued and running runs - must not affect success_rate denominator
        WorkflowRun.objects.create(
            owner=user,
            workflow_definition=wf,
            source_job=job,
            state=WorkflowRun.State.QUEUED,
            step_snapshot=sample_steps,
        )
        WorkflowRun.objects.create(
            owner=user,
            workflow_definition=wf,
            source_job=job,
            state=WorkflowRun.State.RUNNING,
            step_snapshot=sample_steps,
        )

        response = auth_client.get("/api/v1/workflow-runs/summary/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_runs"] == 5
        assert data["by_state"]["completed"] == 1
        assert data["by_state"]["failed"] == 2
        assert data["by_state"]["queued"] == 1
        assert data["by_state"]["running"] == 1
        assert data["success_rate"] == 0.33

    def test_database_side_aggregation_query_count(
        self, auth_client, test_data, django_assert_num_queries
    ):
        _ = test_data
        # Summary aggregation should execute in bounded queries (1 aggregate query + auth query)
        with django_assert_num_queries(1):
            response = auth_client.get("/api/v1/workflow-runs/summary/")
            assert response.status_code == status.HTTP_200_OK

    def test_zero_side_effects(self, auth_client, test_data):
        _ = test_data
        runs_count_before = WorkflowRun.objects.count()
        wf_count_before = WorkflowDefinition.objects.count()
        jobs_count_before = ProcessingJob.objects.count()

        with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_celery:
            response = auth_client.get("/api/v1/workflow-runs/summary/")
            assert response.status_code == status.HTTP_200_OK
            mock_celery.assert_not_called()

        assert WorkflowRun.objects.count() == runs_count_before
        assert WorkflowDefinition.objects.count() == wf_count_before
        assert ProcessingJob.objects.count() == jobs_count_before
