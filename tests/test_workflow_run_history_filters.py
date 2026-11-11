import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun


@pytest.fixture
def auth_client(db: None, django_user_model):  # noqa: ARG001
    user = django_user_model.objects.create_user(
        username="history_tester", email="history@example.com", password="password123"
    )
    client = APIClient()
    client.force_authenticate(user=user)
    client.user = user
    return client


@pytest.fixture
def other_user(db: None, django_user_model):  # noqa: ARG001
    return django_user_model.objects.create_user(
        username="history_other", email="other@example.com", password="password123"
    )


@pytest.fixture
def test_data(auth_client, other_user):
    # Jobs
    job1 = ProcessingJob.objects.create(
        owner=auth_client.user,
        file_type="json",
        state=ProcessingJob.State.COMPLETED,
        original_filename="j1.json",
    )
    job2 = ProcessingJob.objects.create(
        owner=auth_client.user,
        file_type="json",
        state=ProcessingJob.State.COMPLETED,
        original_filename="j2.json",
    )
    foreign_job = ProcessingJob.objects.create(
        owner=other_user,
        file_type="json",
        state=ProcessingJob.State.COMPLETED,
        original_filename="foreign.json",
    )

    # Workflows
    wf1 = WorkflowDefinition.objects.create(
        owner=auth_client.user,
        name="WF 1",
        steps=[{"op": "select_columns", "columns": ["id"]}],
    )
    wf2 = WorkflowDefinition.objects.create(
        owner=auth_client.user,
        name="WF 2 (Archived)",
        steps=[{"op": "select_columns", "columns": ["id"]}],
        is_archived=True,
    )
    foreign_wf = WorkflowDefinition.objects.create(
        owner=other_user,
        name="Foreign WF",
        steps=[{"op": "select_columns", "columns": ["id"]}],
    )

    # Base time
    t0 = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)

    # Runs for user
    # run1: wf1, job1, COMPLETED, t0
    run1 = WorkflowRun.objects.create(
        owner=auth_client.user,
        workflow_definition=wf1,
        source_job=job1,
        state=WorkflowRun.State.COMPLETED,
        step_snapshot=wf1.steps,
    )
    WorkflowRun.objects.filter(id=run1.id).update(created_at=t0)
    run1.refresh_from_db()

    # run2: wf1, job2, FAILED, t0 + 1 hour
    run2 = WorkflowRun.objects.create(
        owner=auth_client.user,
        workflow_definition=wf1,
        source_job=job2,
        state=WorkflowRun.State.FAILED,
        step_snapshot=wf1.steps,
    )
    WorkflowRun.objects.filter(id=run2.id).update(created_at=t0 + timedelta(hours=1))
    run2.refresh_from_db()

    # run3: wf2 (archived), job1, QUEUED, t0 + 2 hours
    run3 = WorkflowRun.objects.create(
        owner=auth_client.user,
        workflow_definition=wf2,
        source_job=job1,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=wf2.steps,
    )
    WorkflowRun.objects.filter(id=run3.id).update(created_at=t0 + timedelta(hours=2))
    run3.refresh_from_db()

    # Foreign run
    foreign_run = WorkflowRun.objects.create(
        owner=other_user,
        workflow_definition=foreign_wf,
        source_job=foreign_job,
        state=WorkflowRun.State.COMPLETED,
        step_snapshot=foreign_wf.steps,
    )
    WorkflowRun.objects.filter(id=foreign_run.id).update(created_at=t0 + timedelta(hours=3))
    foreign_run.refresh_from_db()

    return {
        "job1": job1,
        "job2": job2,
        "foreign_job": foreign_job,
        "wf1": wf1,
        "wf2": wf2,
        "foreign_wf": foreign_wf,
        "run1": run1,
        "run2": run2,
        "run3": run3,
        "foreign_run": foreign_run,
        "t0": t0,
    }


@pytest.mark.django_db
class TestWorkflowRunHistoryFilters:
    def test_anonymous_rejected(self):
        client = APIClient()
        response = client.get("/api/v1/workflow-runs/")
        assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_default_list_shows_only_owned_runs_newest_first(self, auth_client, test_data):
        response = auth_client.get("/api/v1/workflow-runs/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 3
        ids = [item["id"] for item in data["results"]]
        # Default ordering: newest first (run3, run2, run1)
        assert ids == [
            str(test_data["run3"].id),
            str(test_data["run2"].id),
            str(test_data["run1"].id),
        ]

    def test_filter_by_workflow_definition(self, auth_client, test_data):
        url = f"/api/v1/workflow-runs/?workflow_definition={test_data['wf1'].id}"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 2
        ids = [item["id"] for item in data["results"]]
        assert ids == [str(test_data["run2"].id), str(test_data["run1"].id)]

    def test_filter_by_archived_workflow_definition_shows_historical_runs(
        self, auth_client, test_data
    ):
        url = f"/api/v1/workflow-runs/?workflow_definition={test_data['wf2'].id}"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["id"] == str(test_data["run3"].id)

    def test_filter_by_foreign_or_missing_workflow_definition_returns_empty_200(
        self, auth_client, test_data
    ):
        # Foreign workflow
        url_foreign = f"/api/v1/workflow-runs/?workflow_definition={test_data['foreign_wf'].id}"
        resp_foreign = auth_client.get(url_foreign)
        assert resp_foreign.status_code == status.HTTP_200_OK
        assert resp_foreign.json()["count"] == 0
        assert resp_foreign.json()["results"] == []

        # Non-existent workflow
        url_nonexistent = f"/api/v1/workflow-runs/?workflow_definition={uuid.uuid4()}"
        resp_nonexistent = auth_client.get(url_nonexistent)
        assert resp_nonexistent.status_code == status.HTTP_200_OK
        assert resp_nonexistent.json()["count"] == 0
        assert resp_nonexistent.json()["results"] == []

    def test_filter_by_state(self, auth_client, test_data):
        url = "/api/v1/workflow-runs/?state=failed"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["id"] == str(test_data["run2"].id)

    def test_filter_by_source_job(self, auth_client, test_data):
        url = f"/api/v1/workflow-runs/?source_job={test_data['job2'].id}"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["id"] == str(test_data["run2"].id)

    def test_filter_by_foreign_or_missing_source_job_returns_empty_200(
        self, auth_client, test_data
    ):
        url_foreign = f"/api/v1/workflow-runs/?source_job={test_data['foreign_job'].id}"
        resp = auth_client.get(url_foreign)
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["count"] == 0

        url_nonexistent = f"/api/v1/workflow-runs/?source_job={uuid.uuid4()}"
        resp = auth_client.get(url_nonexistent)
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["count"] == 0

    def test_filter_by_created_from_and_created_to_inclusive(self, auth_client, test_data):
        t0 = test_data["t0"]
        # Exact range matching run1 and run2
        t_from = t0.isoformat()
        t_to = (t0 + timedelta(hours=1)).isoformat()
        url = f"/api/v1/workflow-runs/?created_from={t_from}&created_to={t_to}"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 2
        ids = [item["id"] for item in data["results"]]
        assert ids == [str(test_data["run2"].id), str(test_data["run1"].id)]

    def test_filter_by_datetime_with_timezone_offset(self, auth_client):
        # t0 is 2026-11-01T12:00:00Z. In +05:30 that is 2026-11-01T17:30:00+05:30
        t_from_offset = "2026-11-01T17:30:00+05:30"
        t_to_offset = "2026-11-01T18:30:00+05:30"
        url = f"/api/v1/workflow-runs/?created_from={t_from_offset}&created_to={t_to_offset}"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 2

    def test_compound_and_composition(self, auth_client, test_data):
        url = (
            f"/api/v1/workflow-runs/?workflow_definition={test_data['wf1'].id}"
            f"&source_job={test_data['job1'].id}&state=completed"
        )
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["id"] == str(test_data["run1"].id)

    def test_ordering_created_at_ascending(self, auth_client, test_data):
        url = "/api/v1/workflow-runs/?ordering=created_at"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        ids = [item["id"] for item in data["results"]]
        # Oldest first: run1, run2, run3
        assert ids == [
            str(test_data["run1"].id),
            str(test_data["run2"].id),
            str(test_data["run3"].id),
        ]

    def test_ordering_created_at_descending_explicit(self, auth_client, test_data):
        url = "/api/v1/workflow-runs/?ordering=-created_at"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        ids = [item["id"] for item in data["results"]]
        assert ids == [
            str(test_data["run3"].id),
            str(test_data["run2"].id),
            str(test_data["run1"].id),
        ]

    def test_stable_tie_breaker_for_identical_timestamps(self, auth_client, test_data):
        # Create 5 runs at the exact same timestamp
        same_time = test_data["t0"] + timedelta(days=1)
        created_runs = []
        for _i in range(5):
            r = WorkflowRun.objects.create(
                owner=auth_client.user,
                workflow_definition=test_data["wf1"],
                source_job=test_data["job1"],
                state=WorkflowRun.State.COMPLETED,
                step_snapshot=test_data["wf1"].steps,
            )
            WorkflowRun.objects.filter(id=r.id).update(created_at=same_time)
            r.refresh_from_db()
            created_runs.append(r)

        # Pagination with page_size=2
        url_p1 = f"/api/v1/workflow-runs/?created_from={same_time.isoformat()}&page=1&page_size=2"
        res_p1 = auth_client.get(url_p1)
        assert res_p1.status_code == status.HTTP_200_OK
        data_p1 = res_p1.json()
        assert len(data_p1["results"]) == 2

        url_p2 = f"/api/v1/workflow-runs/?created_from={same_time.isoformat()}&page=2&page_size=2"
        res_p2 = auth_client.get(url_p2)
        assert res_p2.status_code == status.HTTP_200_OK
        data_p2 = res_p2.json()
        assert len(data_p2["results"]) == 2

        p1_ids = {item["id"] for item in data_p1["results"]}
        p2_ids = {item["id"] for item in data_p2["results"]}
        # No overlap between pages (stable tie-breaker on UUID)
        assert p1_ids.isdisjoint(p2_ids)

    def test_validation_unknown_query_parameter_rejected(self, auth_client):
        url = "/api/v1/workflow-runs/?foo=bar"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "foo" in str(response.data)

    def test_validation_malformed_uuid_rejected(self, auth_client):
        url = "/api/v1/workflow-runs/?workflow_definition=not-a-uuid"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "workflow_definition" in str(response.data)

        url_job = "/api/v1/workflow-runs/?source_job=bad-uuid"
        response = auth_client.get(url_job)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "source_job" in str(response.data)

    def test_validation_invalid_state_rejected(self, auth_client):
        url = "/api/v1/workflow-runs/?state=invalid_status"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "state" in str(response.data)

    def test_validation_invalid_datetime_rejected(self, auth_client):
        url = "/api/v1/workflow-runs/?created_from=not-a-date"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "created_from" in str(response.data)

    def test_validation_naive_datetime_rejected(self, auth_client):
        # Naive datetime missing timezone offset
        url = "/api/v1/workflow-runs/?created_from=2026-11-01T12:00:00"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "timezone" in str(response.data).lower() or "created_from" in str(response.data)

    def test_validation_inverted_time_range_rejected(self, auth_client):
        t1 = "2026-11-01T15:00:00Z"
        t2 = "2026-11-01T12:00:00Z"
        url = f"/api/v1/workflow-runs/?created_from={t1}&created_to={t2}"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "created_from" in str(response.data) or "detail" in str(response.data)

    def test_validation_unsupported_ordering_rejected(self, auth_client):
        url = "/api/v1/workflow-runs/?ordering=name"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "ordering" in str(response.data)

    def test_pagination_parameters_accepted(self, auth_client):
        url = "/api/v1/workflow-runs/?page=1&page_size=10"
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK

    def test_zero_side_effects_on_filter_queries(self, auth_client):
        runs_before = WorkflowRun.objects.count()
        wfs_before = WorkflowDefinition.objects.count()
        jobs_before = ProcessingJob.objects.count()

        with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
            auth_client.get("/api/v1/workflow-runs/?state=completed")
            auth_client.get("/api/v1/workflow-runs/?invalid_param=bad")
            mock_delay.assert_not_called()

        assert WorkflowRun.objects.count() == runs_before
        assert WorkflowDefinition.objects.count() == wfs_before
        assert ProcessingJob.objects.count() == jobs_before
