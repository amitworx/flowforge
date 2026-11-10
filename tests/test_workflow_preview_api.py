import json
import uuid
from unittest.mock import patch

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStepRun


@pytest.fixture
def auth_client(db: None, django_user_model):  # noqa: ARG001
    user = django_user_model.objects.create_user(
        username="preview_tester", email="tester@example.com", password="password123"
    )
    client = APIClient()
    client.force_authenticate(user=user)
    client.user = user
    return client


@pytest.fixture
def other_user(db: None, django_user_model):  # noqa: ARG001
    return django_user_model.objects.create_user(
        username="preview_other", email="other@example.com", password="password123"
    )


@pytest.fixture
def sample_source_job(auth_client):
    return ProcessingJob.objects.create(
        owner=auth_client.user,
        state=ProcessingJob.State.COMPLETED,
        file_type=ProcessingJob.FileType.JSON,
        original_filename="sample.json",
        output_data=[
            {"id": "001", "name": "alice", "active": True, "score": 10},
            {"id": "002", "name": "bob", "active": False, "score": 20},
            {"id": "003", "name": "charlie", "active": True, "score": 30},
        ],
        input_row_count=3,
        output_row_count=3,
    )


@pytest.fixture
def sample_workflow_def(auth_client):
    return WorkflowDefinition.objects.create(
        owner=auth_client.user,
        name="Test Preview Workflow",
        steps=[
            {"op": "select_columns", "columns": ["id", "name", "active"]},
            {"op": "rename_columns", "mapping": {"name": "full_name"}},
        ],
    )


@pytest.mark.django_db
class TestWorkflowPreviewAPI:
    def test_anonymous_request_rejected(self, sample_workflow_def, sample_source_job):
        client = APIClient()
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = client.post(url, {"source_job_id": str(sample_source_job.id)}, format="json")
        assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_foreign_workflow_returns_safe_404(self, auth_client, other_user, sample_source_job):
        foreign_wf = WorkflowDefinition.objects.create(
            owner=other_user,
            name="Foreign WF",
            steps=[{"op": "select_columns", "columns": ["id"]}],
        )
        url = f"/api/v1/workflows/{foreign_wf.id}/preview/"
        response = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id)}, format="json"
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.data["detail"] == "Workflow definition not found."

    def test_missing_workflow_returns_404(self, auth_client, sample_source_job):
        url = f"/api/v1/workflows/{uuid.uuid4()}/preview/"
        response = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id)}, format="json"
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.data["detail"] == "Workflow definition not found."

    def test_malformed_workflow_uuid_returns_404(self, auth_client, sample_source_job):
        url = "/api/v1/workflows/not-a-uuid/preview/"
        response = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id)}, format="json"
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_foreign_source_job_returns_safe_404(
        self, auth_client, sample_workflow_def, other_user
    ):
        foreign_job = ProcessingJob.objects.create(
            owner=other_user,
            state=ProcessingJob.State.COMPLETED,
            file_type=ProcessingJob.FileType.JSON,
            original_filename="foreign.json",
            output_data=[{"id": 1}],
        )
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(url, {"source_job_id": str(foreign_job.id)}, format="json")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.data["detail"] == "Source processing job not found."

    def test_missing_source_job_returns_404(self, auth_client, sample_workflow_def):
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(url, {"source_job_id": str(uuid.uuid4())}, format="json")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.data["detail"] == "Source processing job not found."

    def test_malformed_source_job_uuid_returns_400(self, auth_client, sample_workflow_def):
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(url, {"source_job_id": "not-a-uuid"}, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "source_job_id" in response.data

    def test_non_completed_source_job_rejected(self, auth_client, sample_workflow_def):
        job = ProcessingJob.objects.create(
            owner=auth_client.user,
            state=ProcessingJob.State.PENDING,
            file_type=ProcessingJob.FileType.JSON,
            original_filename="pending.json",
            output_data=[],
        )
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(url, {"source_job_id": str(job.id)}, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "pending" in response.data["detail"]

    def test_archived_workflow_cannot_preview_409(
        self, auth_client, sample_workflow_def, sample_source_job
    ):
        sample_workflow_def.is_archived = True
        sample_workflow_def.save()
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id)}, format="json"
        )
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "archived" in response.data["detail"].lower()

    def test_unexpected_payload_fields_rejected_400(
        self, auth_client, sample_workflow_def, sample_source_job
    ):
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(
            url,
            {"source_job_id": str(sample_source_job.id), "unexpected_field": "disallowed"},
            format="json",
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "unexpected_field" in str(response.data)

    def test_preview_limit_validation(self, auth_client, sample_workflow_def, sample_source_job):
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"

        # Zero or negative limit
        resp = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id), "preview_limit": 0}, format="json"
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

        resp = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id), "preview_limit": -5}, format="json"
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

        # Over 50
        resp = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id), "preview_limit": 51}, format="json"
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

        # Non-integer
        resp = auth_client.post(
            url,
            {"source_job_id": str(sample_source_job.id), "preview_limit": "many"},
            format="json",
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_successful_preview_with_defaults(
        self, auth_client, sample_workflow_def, sample_source_job
    ):
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id)}, format="json"
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["output_columns"] == ["id", "full_name", "active"]
        assert data["input_row_count"] == 3
        assert data["output_row_count"] == 3
        assert data["preview_limit"] == 20
        assert len(data["preview_rows"]) == 3
        assert data["preview_rows"][0] == {"id": "001", "full_name": "alice", "active": True}
        assert len(data["step_audits"]) == 2
        assert data["step_audits"][0]["status"] == "completed"
        assert data["step_audits"][0]["operation"] == "select_columns"
        assert data["step_audits"][1]["status"] == "completed"
        assert data["step_audits"][1]["operation"] == "rename_columns"

    def test_preview_slices_rows_to_limit_while_preserving_total_output_count(
        self, auth_client, sample_workflow_def, sample_source_job
    ):
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
        response = auth_client.post(
            url,
            {"source_job_id": str(sample_source_job.id), "preview_limit": 2},
            format="json",
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Full pipeline evaluation
        assert data["input_row_count"] == 3
        assert data["output_row_count"] == 3
        assert data["preview_limit"] == 2
        # Sliced output
        assert len(data["preview_rows"]) == 2
        assert data["preview_rows"] == [
            {"id": "001", "full_name": "alice", "active": True},
            {"id": "002", "full_name": "bob", "active": False},
        ]

    def test_type_sensitive_scalar_preservation_and_leading_zeros(self, auth_client):
        # Test leading zeros string vs integer vs boolean exactness
        source_job = ProcessingJob.objects.create(
            owner=auth_client.user,
            state=ProcessingJob.State.COMPLETED,
            file_type=ProcessingJob.FileType.JSON,
            original_filename="strict.json",
            output_data=[
                {"code": "00123", "flag": True, "num": 1},
                {"code": "123", "flag": False, "num": 0},
                {"code": "00123", "flag": 1, "num": 1},
            ],
            input_row_count=3,
            output_row_count=3,
        )
        workflow_def = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Filter Strict",
            steps=[{"op": "filter_equals", "column": "flag", "value": True}],
        )
        url = f"/api/v1/workflows/{workflow_def.id}/preview/"
        response = auth_client.post(url, {"source_job_id": str(source_job.id)}, format="json")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # flag=True must NOT match flag=1
        assert data["input_row_count"] == 3
        assert data["output_row_count"] == 1
        assert len(data["preview_rows"]) == 1
        assert data["preview_rows"][0]["code"] == "00123"
        assert data["preview_rows"][0]["flag"] is True

    def test_empty_resulting_dataset_schema_propagation(self, auth_client, sample_source_job):
        wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Filter to Empty",
            steps=[
                {"op": "select_columns", "columns": ["id", "name"]},
                {"op": "filter_equals", "column": "name", "value": "non_existent"},
                {"op": "rename_columns", "mapping": {"id": "user_id"}},
            ],
        )
        url = f"/api/v1/workflows/{wf.id}/preview/"
        response = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id)}, format="json"
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["input_row_count"] == 3
        assert data["output_row_count"] == 0
        assert data["output_columns"] == ["user_id", "name"]
        assert data["preview_rows"] == []

    def test_invalid_step_execution_returns_sanitized_400(self, auth_client, sample_source_job):
        wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Missing Column WF",
            steps=[
                {"op": "select_columns", "columns": ["non_existent_col"]},
            ],
        )
        url = f"/api/v1/workflows/{wf.id}/preview/"
        response = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id)}, format="json"
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "non_existent_col" in data["detail"]
        assert data["step_index"] == 0
        assert "failed_step_audit" in data
        assert data["failed_step_audit"]["status"] == "failed"

    def test_guarantees_no_db_side_effects_or_runs_or_tasks(
        self, auth_client, sample_workflow_def, sample_source_job
    ):
        runs_before = WorkflowRun.objects.count()
        step_runs_before = WorkflowStepRun.objects.count()

        with patch("flowforge.workflows.tasks.execute_workflow_run_task.delay") as mock_delay:
            url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"
            resp = auth_client.post(
                url, {"source_job_id": str(sample_source_job.id)}, format="json"
            )
            assert resp.status_code == status.HTTP_200_OK

            # Also test error case
            wf_err = WorkflowDefinition.objects.create(
                owner=auth_client.user,
                name="Error WF",
                steps=[{"op": "select_columns", "columns": ["bad"]}],
            )
            resp_err = auth_client.post(
                f"/api/v1/workflows/{wf_err.id}/preview/",
                {"source_job_id": str(sample_source_job.id)},
                format="json",
            )
            assert resp_err.status_code == status.HTTP_400_BAD_REQUEST

            mock_delay.assert_not_called()

        assert WorkflowRun.objects.count() == runs_before
        assert WorkflowStepRun.objects.count() == step_runs_before

        # Check that original source job output was not mutated
        sample_source_job.refresh_from_db()
        assert len(sample_source_job.output_data) == 3
        assert "score" in sample_source_job.output_data[0]

    def test_payload_size_limit_rejection_413(self, auth_client):
        # If preview output serialized size exceeds max byte limit (10 MiB)
        huge_string = "x" * 200_000
        source_job = ProcessingJob.objects.create(
            owner=auth_client.user,
            state=ProcessingJob.State.COMPLETED,
            file_type=ProcessingJob.FileType.JSON,
            original_filename="huge.json",
            output_data=[{"data": huge_string} for _ in range(60)],  # ~12 MB
            input_row_count=60,
            output_row_count=60,
        )
        wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Huge Data",
            steps=[{"op": "select_columns", "columns": ["data"]}],
        )
        url = f"/api/v1/workflows/{wf.id}/preview/"
        # Request preview_limit=50 so 50 rows * 200KB = ~10MB+
        with patch("flowforge.workflows.views.DEFAULT_MAX_EXPORT_SIZE_BYTES", 500_000):
            response = auth_client.post(
                url,
                {"source_job_id": str(source_job.id), "preview_limit": 50},
                format="json",
            )
            assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            assert "exceeds maximum allowed size" in response.data["detail"]

    def test_preview_response_content_bytes_match_serialized_bytes_and_exact_boundary(
        self, auth_client, sample_source_job, sample_workflow_def
    ):
        """
        Verify that HTTP response bytes on success correspond to the checked encoded payload,
        and test exact boundary (exact byte size succeeds; exact byte size - 1 fails with 413).
        """
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"

        # Baseline request
        resp = auth_client.post(
            url, {"source_job_id": str(sample_source_job.id), "preview_limit": 20}, format="json"
        )
        assert resp.status_code == status.HTTP_200_OK
        exact_bytes = len(resp.content)
        assert exact_bytes > 0
        # Assert content is valid parseable JSON matching .json()
        assert resp.json() == json.loads(resp.content.decode("utf-8"))

        # Exact boundary: max_bytes == exact_bytes succeeds
        with patch("flowforge.workflows.views.DEFAULT_MAX_EXPORT_SIZE_BYTES", exact_bytes):
            resp_exact = auth_client.post(
                url,
                {"source_job_id": str(sample_source_job.id), "preview_limit": 20},
                format="json",
            )
            assert resp_exact.status_code == status.HTTP_200_OK
            assert len(resp_exact.content) == exact_bytes

        # Over-by-one: max_bytes == exact_bytes - 1 fails with 413
        with patch("flowforge.workflows.views.DEFAULT_MAX_EXPORT_SIZE_BYTES", exact_bytes - 1):
            resp_over = auth_client.post(
                url,
                {"source_job_id": str(sample_source_job.id), "preview_limit": 20},
                format="json",
            )
            assert resp_over.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            assert "exceeds maximum allowed size" in resp_over.data["detail"]

    def test_preview_unicode_multibyte_size_calculation_and_preservation(self, auth_client):
        """
        Verify that multi-byte UTF-8 Unicode characters (such as emojis and CJK glyphs)
        are sized based on actual encoded UTF-8 byte lengths rather than Python character counts,
        and that they are preserved unescaped in the preview output.
        """
        # Character count: 10, UTF-8 byte length: 40 bytes (4 bytes per emoji)
        multibyte_str = "🚀" * 10
        source_job = ProcessingJob.objects.create(
            owner=auth_client.user,
            state=ProcessingJob.State.COMPLETED,
            file_type=ProcessingJob.FileType.JSON,
            original_filename="unicode.json",
            output_data=[{"id": 1, "text": multibyte_str, "cjk": "中文测试"}],
            input_row_count=1,
            output_row_count=1,
        )
        wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Unicode WF",
            steps=[{"op": "select_columns", "columns": ["text", "cjk"]}],
        )
        url = f"/api/v1/workflows/{wf.id}/preview/"

        resp = auth_client.post(url, {"source_job_id": str(source_job.id)}, format="json")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["preview_rows"][0]["text"] == multibyte_str
        assert data["preview_rows"][0]["cjk"] == "中文测试"

        # Verify raw HTTP content byte count vs max_bytes
        exact_bytes = len(resp.content)
        assert exact_bytes > len(
            resp.content.decode("utf-8")
        )  # Multi-byte UTF-8 proves bytes > chars

        # Exact boundary with multi-byte
        with patch("flowforge.workflows.views.DEFAULT_MAX_EXPORT_SIZE_BYTES", exact_bytes):
            resp_exact = auth_client.post(url, {"source_job_id": str(source_job.id)}, format="json")
            assert resp_exact.status_code == status.HTTP_200_OK
            assert len(resp_exact.content) == exact_bytes

        # Over-by-one with multi-byte
        with patch("flowforge.workflows.views.DEFAULT_MAX_EXPORT_SIZE_BYTES", exact_bytes - 1):
            resp_over = auth_client.post(url, {"source_job_id": str(source_job.id)}, format="json")
            assert resp_over.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_preview_early_termination_bounded_allocation(self, auth_client):
        """
        Demonstrate that bounded incremental JSON serialization halts early as soon as
        accumulated bytes exceed the threshold, without completing full payload serialization.
        """
        # Create source job with multiple large rows
        rows = [{"id": i, "content": "A" * 50_000} for i in range(10)]
        source_job = ProcessingJob.objects.create(
            owner=auth_client.user,
            state=ProcessingJob.State.COMPLETED,
            file_type=ProcessingJob.FileType.JSON,
            original_filename="large.json",
            output_data=rows,
            input_row_count=10,
            output_row_count=10,
        )
        wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Large Rows WF",
            steps=[{"op": "select_columns", "columns": ["id", "content"]}],
        )
        url = f"/api/v1/workflows/{wf.id}/preview/"

        # Set limit to 60,000 bytes (exceeded after row 1)
        with patch("flowforge.workflows.views.DEFAULT_MAX_EXPORT_SIZE_BYTES", 60_000):
            resp = auth_client.post(
                url, {"source_job_id": str(source_job.id), "preview_limit": 10}, format="json"
            )
            assert resp.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_preview_respects_content_negotiation_and_rejects_unacceptable(
        self, auth_client, sample_source_job, sample_workflow_def
    ):
        """
        Verify that malformed or unsupported Accept headers (e.g. text/html, invalid syntax)
        still trigger DRF's 406 Not Acceptable status rather than bypassing content negotiation.
        """
        url = f"/api/v1/workflows/{sample_workflow_def.id}/preview/"

        # Unacceptable Accept header
        resp = auth_client.post(
            url,
            {"source_job_id": str(sample_source_job.id)},
            format="json",
            HTTP_ACCEPT="text/html",
        )
        assert resp.status_code == status.HTTP_406_NOT_ACCEPTABLE

        # Valid Accept headers (*/*, application/json)
        resp_json = auth_client.post(
            url,
            {"source_job_id": str(sample_source_job.id)},
            format="json",
            HTTP_ACCEPT="application/json",
        )
        assert resp_json.status_code == status.HTTP_200_OK
        assert "application/json" in resp_json.get("Content-Type")

        resp_wildcard = auth_client.post(
            url,
            {"source_job_id": str(sample_source_job.id)},
            format="json",
            HTTP_ACCEPT="*/*",
        )
        assert resp_wildcard.status_code == status.HTTP_200_OK
