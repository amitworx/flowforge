import uuid
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
def source_workflow(auth_client, sample_steps):
    return WorkflowDefinition.objects.create(
        owner=auth_client.user,
        name="Source Pipeline",
        steps=sample_steps,
    )


@pytest.mark.django_db
class TestWorkflowCloneAPI:
    def test_anonymous_request_denied(self, source_workflow):
        anon_client = APIClient()
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        response = anon_client.post(url, {}, format="json")
        assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_clone_nonexistent_or_foreign_workflow_returns_404_parity(
        self, auth_client, other_client, sample_steps
    ):
        # 1. Nonexistent UUID
        random_uuid = uuid.uuid4()
        url_nonexistent = f"/api/v1/workflows/{random_uuid}/clone/"
        res_nonexistent = auth_client.post(url_nonexistent, {}, format="json")
        assert res_nonexistent.status_code == status.HTTP_404_NOT_FOUND

        # 2. Foreign UUID (owned by other_client.user)
        foreign_wf = WorkflowDefinition.objects.create(
            owner=other_client.user,
            name="Foreign Pipeline",
            steps=sample_steps,
        )
        url_foreign = f"/api/v1/workflows/{foreign_wf.id}/clone/"
        res_foreign = auth_client.post(url_foreign, {}, format="json")
        assert res_foreign.status_code == status.HTTP_404_NOT_FOUND

        # Error structures match indistinguishably
        assert res_nonexistent.json() == res_foreign.json()

    def test_basic_authenticated_clone_default_generated_name(
        self, auth_client, source_workflow, sample_steps
    ):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        response = auth_client.post(url, {}, format="json")
        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        assert data["id"] != str(source_workflow.id)
        assert data["name"] == "Source Pipeline (copy)"
        assert data["steps"] == sample_steps
        assert data["is_archived"] is False
        assert data["archived_at"] is None
        assert "created_at" in data
        assert "updated_at" in data

        # Location header
        expected_location = f"/api/v1/workflows/{data['id']}/"
        assert response.headers.get("Location") == expected_location

        # Verify DB persistence
        clone_db = WorkflowDefinition.objects.get(id=data["id"])
        assert clone_db.owner == auth_client.user
        assert clone_db.name == "Source Pipeline (copy)"
        assert clone_db.steps == sample_steps
        assert clone_db.is_archived is False
        assert clone_db.archived_at is None

    def test_clone_with_custom_name(self, auth_client, source_workflow, sample_steps):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        payload = {"name": "Custom Cloned Pipeline"}
        response = auth_client.post(url, payload, format="json")
        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        assert data["name"] == "Custom Cloned Pipeline"
        assert data["steps"] == sample_steps

    def test_clone_name_whitespace_and_unicode_handling(self, auth_client, source_workflow):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        # Leading/trailing whitespace stripped, unicode preserved
        payload = {"name": "  Pipeline — Étude 🚀  "}
        response = auth_client.post(url, payload, format="json")
        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        assert data["name"] == "Pipeline \u2014 \u00c9tude \U0001f680"

    def test_default_generated_name_128_char_boundary_truncation(self, auth_client, sample_steps):
        # 128 characters source name
        long_name = "X" * 128
        source_wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name=long_name,
            steps=sample_steps,
        )
        url = f"/api/v1/workflows/{source_wf.id}/clone/"
        response = auth_client.post(url, {}, format="json")
        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        expected_suffix = " (copy)"
        assert len(data["name"]) == 128
        assert data["name"].endswith(expected_suffix)
        assert data["name"] == ("X" * (128 - len(expected_suffix))) + expected_suffix

    def test_validation_rejects_empty_or_whitespace_only_name(self, auth_client, source_workflow):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        # Empty string
        res1 = auth_client.post(url, {"name": ""}, format="json")
        assert res1.status_code == status.HTTP_400_BAD_REQUEST
        assert "name" in res1.json()

        # Whitespace-only string
        res2 = auth_client.post(url, {"name": "    \t\n  "}, format="json")
        assert res2.status_code == status.HTTP_400_BAD_REQUEST
        assert "name" in res2.json()

    def test_validation_rejects_explicit_name_exceeding_128_chars(
        self, auth_client, source_workflow
    ):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        payload = {"name": "A" * 129}
        response = auth_client.post(url, payload, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "name" in response.json()

    def test_validation_rejects_non_string_name(self, auth_client, source_workflow):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        for bad_val in [123, True, ["name"], {"nested": "name"}]:
            res = auth_client.post(url, {"name": bad_val}, format="json")
            assert res.status_code == status.HTTP_400_BAD_REQUEST
            assert "name" in res.json()

    def test_validation_rejects_unknown_fields(self, auth_client, source_workflow):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        payload = {"name": "Valid Name", "extra_field": "disallowed"}
        response = auth_client.post(url, payload, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "extra_field" in response.json()

    def test_validation_rejects_non_object_json_body(self, auth_client, source_workflow):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        response = auth_client.post(url, ["not", "an", "object"], format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_clone_of_archived_workflow_yields_active_clone(self, auth_client, source_workflow):
        # Archive the source
        archive_url = f"/api/v1/workflows/{source_workflow.id}/archive/"
        res_arch = auth_client.post(archive_url)
        assert res_arch.status_code == status.HTTP_200_OK
        source_workflow.refresh_from_db()
        assert source_workflow.is_archived is True
        assert source_workflow.archived_at is not None
        archived_at_orig = source_workflow.archived_at

        # Clone archived source
        clone_url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        response = auth_client.post(clone_url, {}, format="json")
        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        assert data["is_archived"] is False
        assert data["archived_at"] is None

        # Verify source remains completely untouched
        source_workflow.refresh_from_db()
        assert source_workflow.is_archived is True
        assert source_workflow.archived_at == archived_at_orig

    def test_zero_side_effects_on_runs_and_celery(self, auth_client, source_workflow):
        # Setup an ingestion job and historical run on the source
        job = ProcessingJob.objects.create(
            owner=auth_client.user,
            file_type="csv",
            state=ProcessingJob.State.COMPLETED,
            original_filename="test.csv",
            output_data=[{"id": 1, "name": "Alice", "department": "Engineering"}],
        )
        hist_run = WorkflowRun.objects.create(
            owner=auth_client.user,
            workflow_definition=source_workflow,
            source_job=job,
            state=WorkflowRun.State.COMPLETED,
            step_snapshot=source_workflow.steps,
            output_data=[{"id": 1, "name": "Alice", "department": "Engineering"}],
        )

        with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_celery:
            url = f"/api/v1/workflows/{source_workflow.id}/clone/"
            response = auth_client.post(url, {}, format="json")
            assert response.status_code == status.HTTP_201_CREATED
            mock_celery.assert_not_called()

        clone_id = response.json()["id"]

        # Ensure historical run remains attached strictly to source_workflow
        hist_run.refresh_from_db()
        assert hist_run.workflow_definition == source_workflow

        # Ensure clone has zero runs
        assert WorkflowRun.objects.filter(workflow_definition_id=clone_id).count() == 0

    def test_independent_step_mutation_after_clone(self, auth_client, source_workflow):
        clone_url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        response = auth_client.post(clone_url, {}, format="json")
        assert response.status_code == status.HTTP_201_CREATED
        clone_id = response.json()["id"]

        # Patch original
        orig_patch_url = f"/api/v1/workflows/{source_workflow.id}/"
        new_orig_steps = [{"op": "select_columns", "columns": ["id"]}]
        res_patch_orig = auth_client.patch(orig_patch_url, {"steps": new_orig_steps}, format="json")
        assert res_patch_orig.status_code == status.HTTP_200_OK

        # Patch clone
        clone_patch_url = f"/api/v1/workflows/{clone_id}/"
        new_clone_steps = [{"op": "select_columns", "columns": ["name"]}]
        res_patch_clone = auth_client.patch(
            clone_patch_url, {"steps": new_clone_steps}, format="json"
        )
        assert res_patch_clone.status_code == status.HTTP_200_OK

        # Verify DB states are isolated
        source_workflow.refresh_from_db()
        clone_wf = WorkflowDefinition.objects.get(id=clone_id)
        assert source_workflow.steps == new_orig_steps
        assert clone_wf.steps == new_clone_steps

    def test_repeated_clone_requests_produce_distinct_records(self, auth_client, source_workflow):
        url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        res1 = auth_client.post(url, {}, format="json")
        res2 = auth_client.post(url, {}, format="json")
        assert res1.status_code == status.HTTP_201_CREATED
        assert res2.status_code == status.HTTP_201_CREATED

        id1 = res1.json()["id"]
        id2 = res2.json()["id"]
        assert id1 != id2
        assert id1 != str(source_workflow.id)
        assert id2 != str(source_workflow.id)

    def test_cloned_workflow_is_previewable_and_executable(self, auth_client, source_workflow):
        # 1. Clone source
        clone_url = f"/api/v1/workflows/{source_workflow.id}/clone/"
        res_clone = auth_client.post(clone_url, {}, format="json")
        assert res_clone.status_code == status.HTTP_201_CREATED
        clone_id = res_clone.json()["id"]

        # 2. Ingest processing job matching schema
        job = ProcessingJob.objects.create(
            owner=auth_client.user,
            file_type="csv",
            state=ProcessingJob.State.COMPLETED,
            original_filename="sample.csv",
            column_names=["id", "name", "department"],
            output_data=[
                {"id": 1, "name": "Alice", "department": "Engineering"},
                {"id": 2, "name": "Bob", "department": "Marketing"},
                {"id": 3, "name": "Charlie", "department": "Engineering"},
            ],
        )

        # 3. Preview cloned workflow
        preview_url = f"/api/v1/workflows/{clone_id}/preview/"
        res_preview = auth_client.post(preview_url, {"source_job_id": str(job.id)}, format="json")
        assert res_preview.status_code == status.HTTP_200_OK
        preview_data = res_preview.json()
        assert preview_data["output_row_count"] == 2
        assert preview_data["preview_rows"] == [
            {"id": 1, "name": "Alice", "department": "Engineering"},
            {"id": 3, "name": "Charlie", "department": "Engineering"},
        ]

        # 4. Execute synchronous run on cloned workflow
        run_url = f"/api/v1/workflows/{clone_id}/runs/"
        res_run = auth_client.post(run_url, {"source_job_id": str(job.id)}, format="json")
        assert res_run.status_code == status.HTTP_201_CREATED
        run_data = res_run.json()
        assert run_data["state"] == "completed"
        assert run_data["output_row_count"] == 2

    def test_clone_fails_on_corrupt_or_legacy_source_steps_with_400(self, auth_client):
        # Corrupt test cases: unsupported op, missing required field, invalid shape, >10 steps
        corrupt_cases = [
            # 1. Unsupported operation (e.g. legacy 'filter_rows')
            [{"op": "filter_rows", "column": "dept", "operator": "equals", "value": "x"}],
            # 2. Missing required field in select_columns
            [{"op": "select_columns"}],
            # 3. Non-dictionary step
            ["not_a_dict"],
            # 4. More than 10 steps
            [{"op": "select_columns", "columns": ["id"]}] * 11,
            # 5. Invalid filter_equals (non-scalar value)
            [{"op": "filter_equals", "column": "dept", "value": {"nested": "obj"}}],
        ]

        initial_count = WorkflowDefinition.objects.count()

        for bad_steps in corrupt_cases:
            # Create directly in database bypassing DRF serializer validation
            corrupt_wf = WorkflowDefinition.objects.create(
                owner=auth_client.user,
                name="Legacy Corrupt Pipeline",
                steps=bad_steps,
            )

            clone_url = f"/api/v1/workflows/{corrupt_wf.id}/clone/"
            res = auth_client.post(clone_url, {}, format="json")
            assert res.status_code == status.HTTP_400_BAD_REQUEST
            err_data = res.json()
            assert "detail" in err_data
            assert "invalid" in err_data["detail"].lower()

            # Ensure original is unchanged
            corrupt_wf.refresh_from_db()
            assert corrupt_wf.steps == bad_steps

        # Ensure zero new WorkflowDefinitions were created
        assert WorkflowDefinition.objects.count() == initial_count + len(corrupt_cases)
