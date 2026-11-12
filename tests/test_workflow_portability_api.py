import json
import uuid

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition


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
        name="Customer Pipeline",
        steps=sample_steps,
    )


@pytest.mark.django_db
class TestWorkflowPortabilityExportAPI:
    def test_anonymous_export_denied(self, source_workflow):
        client = APIClient()
        response = client.get(f"/api/v1/workflows/{source_workflow.id}/export/")
        assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_export_foreign_or_missing_workflow_returns_404_parity(
        self, auth_client, other_client, sample_steps
    ):
        # 1. Nonexistent UUID
        res_nonexistent = auth_client.get(f"/api/v1/workflows/{uuid.uuid4()}/export/")
        assert res_nonexistent.status_code == status.HTTP_404_NOT_FOUND

        # 2. Foreign UUID
        foreign_wf = WorkflowDefinition.objects.create(
            owner=other_client.user,
            name="Foreign Pipeline",
            steps=sample_steps,
        )
        res_foreign = auth_client.get(f"/api/v1/workflows/{foreign_wf.id}/export/")
        assert res_foreign.status_code == status.HTTP_404_NOT_FOUND

        # 404 parity: identical response bodies
        assert res_nonexistent.json() == res_foreign.json()

    def test_export_active_workflow_success_and_headers(
        self, auth_client, source_workflow, sample_steps
    ):
        response = auth_client.get(f"/api/v1/workflows/{source_workflow.id}/export/")
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["Content-Type"] == "application/json; charset=utf-8"
        expected_disposition = f'attachment; filename="workflow-{source_workflow.id}.json"'
        assert response.headers["Content-Disposition"] == expected_disposition

        data = response.json()
        assert data == {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Customer Pipeline",
            "steps": sample_steps,
        }

    def test_export_archived_workflow_success(self, auth_client, sample_steps):
        archived_wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Archived Exportable Pipeline",
            steps=sample_steps,
            is_archived=True,
        )
        response = auth_client.get(f"/api/v1/workflows/{archived_wf.id}/export/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Archived Exportable Pipeline"
        assert data["format"] == "flowforge.workflow"
        assert data["format_version"] == 1
        assert "is_archived" not in data
        assert "owner" not in data

    def test_export_corrupt_legacy_definition_fails_with_400(self, auth_client):
        corrupt_wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Corrupt Workflow",
            steps=[{"op": "filter_rows", "column": "dept", "operator": "equals", "value": "x"}],
        )
        response = auth_client.get(f"/api/v1/workflows/{corrupt_wf.id}/export/")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "invalid" in response.json()["detail"].lower()


@pytest.mark.django_db
class TestWorkflowPortabilityImportAPI:
    def test_anonymous_import_denied(self, sample_steps):
        client = APIClient()
        payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Imported Workflow",
            "steps": sample_steps,
        }
        response = client.post("/api/v1/workflows/import/", payload, format="json")
        assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_successful_import_creates_active_definition(self, auth_client, sample_steps):
        payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Imported Workflow",
            "steps": sample_steps,
        }
        response = auth_client.post("/api/v1/workflows/import/", payload, format="json")
        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        new_id = data["id"]
        assert data["name"] == "Imported Workflow"
        assert data["steps"] == sample_steps
        assert data["is_archived"] is False
        assert data["archived_at"] is None
        assert response.headers["Location"] == f"/api/v1/workflows/{new_id}/"

        # Verify DB
        wf_db = WorkflowDefinition.objects.get(id=new_id)
        assert wf_db.owner == auth_client.user
        assert wf_db.name == "Imported Workflow"
        assert wf_db.steps == sample_steps
        assert wf_db.is_archived is False

    def test_repeated_import_creates_distinct_workflows(self, auth_client, sample_steps):
        payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Repeated Import",
            "steps": sample_steps,
        }
        res1 = auth_client.post("/api/v1/workflows/import/", payload, format="json")
        res2 = auth_client.post("/api/v1/workflows/import/", payload, format="json")
        assert res1.status_code == status.HTTP_201_CREATED
        assert res2.status_code == status.HTTP_201_CREATED
        assert res1.json()["id"] != res2.json()["id"]

    def test_import_name_whitespace_and_unicode_handling(self, auth_client, sample_steps):
        payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "  Data Cleaning \u2014 \u00c9tude \U0001f680  ",
            "steps": sample_steps,
        }
        response = auth_client.post("/api/v1/workflows/import/", payload, format="json")
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["name"] == "Data Cleaning \u2014 \u00c9tude \U0001f680"

    def test_import_rejects_non_object_json(self, auth_client):
        for bad_payload in [["list"], "string", 123, True]:
            res = auth_client.post("/api/v1/workflows/import/", bad_payload, format="json")
            assert res.status_code == status.HTTP_400_BAD_REQUEST

    def test_import_rejects_missing_or_extra_envelope_fields(self, auth_client, sample_steps):
        # 1. Missing format
        res1 = auth_client.post(
            "/api/v1/workflows/import/",
            {"format_version": 1, "name": "Test", "steps": sample_steps},
            format="json",
        )
        assert res1.status_code == status.HTTP_400_BAD_REQUEST

        # 2. Missing format_version
        res2 = auth_client.post(
            "/api/v1/workflows/import/",
            {"format": "flowforge.workflow", "name": "Test", "steps": sample_steps},
            format="json",
        )
        assert res2.status_code == status.HTTP_400_BAD_REQUEST

        # 3. Extra metadata injection fields (e.g. owner, id, is_archived)
        res3 = auth_client.post(
            "/api/v1/workflows/import/",
            {
                "format": "flowforge.workflow",
                "format_version": 1,
                "name": "Test",
                "steps": sample_steps,
                "owner": "admin",
                "is_archived": True,
            },
            format="json",
        )
        assert res3.status_code == status.HTTP_400_BAD_REQUEST

    def test_import_rejects_wrong_format_or_version(self, auth_client, sample_steps):
        # Wrong format
        res1 = auth_client.post(
            "/api/v1/workflows/import/",
            {
                "format": "other.workflow",
                "format_version": 1,
                "name": "Test",
                "steps": sample_steps,
            },
            format="json",
        )
        assert res1.status_code == status.HTTP_400_BAD_REQUEST

        # Unsupported format_version (e.g. 2, 0, "1")
        for bad_v in [2, 0, -1, "1"]:
            res = auth_client.post(
                "/api/v1/workflows/import/",
                {
                    "format": "flowforge.workflow",
                    "format_version": bad_v,
                    "name": "Test",
                    "steps": sample_steps,
                },
                format="json",
            )
            assert res.status_code == status.HTTP_400_BAD_REQUEST

    def test_import_rejects_invalid_name(self, auth_client, sample_steps):
        for bad_name in ["", "   ", "A" * 129, 123, ["name"], None]:
            res = auth_client.post(
                "/api/v1/workflows/import/",
                {
                    "format": "flowforge.workflow",
                    "format_version": 1,
                    "name": bad_name,
                    "steps": sample_steps,
                },
                format="json",
            )
            assert res.status_code == status.HTTP_400_BAD_REQUEST

    def test_import_rejects_invalid_steps(self, auth_client):
        bad_steps_list = [
            # Unsupported operation
            [{"op": "filter_rows", "column": "id", "operator": "equals", "value": 1}],
            # Missing op
            [{"columns": ["id"]}],
            # More than 10 steps
            [{"op": "select_columns", "columns": ["id"]}] * 11,
            # Empty steps list
            [],
            # Non-dict step
            ["select_columns"],
            # Non-scalar value in filter_equals
            [{"op": "filter_equals", "column": "id", "value": {"nested": "obj"}}],
            [{"op": "filter_equals", "column": "id", "value": [1, 2, 3]}],
        ]
        for bad_steps in bad_steps_list:
            res = auth_client.post(
                "/api/v1/workflows/import/",
                {
                    "format": "flowforge.workflow",
                    "format_version": 1,
                    "name": "Bad Steps",
                    "steps": bad_steps,
                },
                format="json",
            )
            assert res.status_code == status.HTTP_400_BAD_REQUEST

    def test_roundtrip_export_import_preview_and_execution(self, auth_client, source_workflow):
        # 1. Export source workflow
        res_export = auth_client.get(f"/api/v1/workflows/{source_workflow.id}/export/")
        assert res_export.status_code == status.HTTP_200_OK
        exported_envelope = res_export.json()

        # 2. Import into a new workflow
        res_import = auth_client.post("/api/v1/workflows/import/", exported_envelope, format="json")
        assert res_import.status_code == status.HTTP_201_CREATED
        imported_id = res_import.json()["id"]
        assert imported_id != str(source_workflow.id)

        # 3. Create input job
        job = ProcessingJob.objects.create(
            owner=auth_client.user,
            file_type="csv",
            state=ProcessingJob.State.COMPLETED,
            original_filename="sample.csv",
            column_names=["id", "name", "department"],
            output_data=[
                {"id": "001", "name": "Alice", "department": "Engineering"},
                {"id": "002", "name": "Bob", "department": "Marketing"},
                {"id": "003", "name": "Charlie", "department": "Engineering"},
            ],
        )

        # 4. Preview imported workflow
        res_preview = auth_client.post(
            f"/api/v1/workflows/{imported_id}/preview/",
            {"source_job_id": str(job.id)},
            format="json",
        )
        assert res_preview.status_code == status.HTTP_200_OK
        preview_data = res_preview.json()
        assert preview_data["output_row_count"] == 2
        # Check string leading zero preservation
        assert preview_data["preview_rows"][0]["id"] == "001"
        assert preview_data["preview_rows"][1]["id"] == "003"

        # 5. Execute imported workflow
        res_run = auth_client.post(
            f"/api/v1/workflows/{imported_id}/runs/",
            {"source_job_id": str(job.id)},
            format="json",
        )
        assert res_run.status_code == status.HTTP_201_CREATED
        run_data = res_run.json()
        assert run_data["state"] == "completed"
        assert run_data["output_row_count"] == 2

        # 6. Independence: patch original definition, verify imported is not mutated
        res_patch = auth_client.patch(
            f"/api/v1/workflows/{source_workflow.id}/",
            {"name": "Mutated Original Name"},
            format="json",
        )
        assert res_patch.status_code == status.HTTP_200_OK

        imported_wf = WorkflowDefinition.objects.get(id=imported_id)
        assert imported_wf.name == "Customer Pipeline"


@pytest.mark.django_db
class TestWorkflowPortabilitySizeLimits:
    def test_import_exact_64k_boundary_accepted(self, auth_client):
        # Craft a valid payload that is close to but <= 64 KiB (65,536 bytes)
        # Using string padding in step value
        base_payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Boundary Test",
            "steps": [{"op": "select_columns", "columns": ["id"]}],
        }
        # Measure size
        # Add filter step with padding to reach near 64 KiB
        # filter_equals column name max 128 chars, value string can be long
        step_template = {
            "op": "filter_equals",
            "column": "col",
            "value": "",
        }
        base_payload["steps"].append(step_template)
        # Recalculate
        current_len = len(json.dumps(base_payload).encode("utf-8"))
        pad_len = 65536 - current_len
        base_payload["steps"][1]["value"] = "X" * pad_len
        final_bytes = json.dumps(base_payload).encode("utf-8")
        assert len(final_bytes) == 65536

        response = auth_client.post(
            "/api/v1/workflows/import/",
            final_bytes,
            content_type="application/json",
        )
        assert response.status_code == status.HTTP_201_CREATED

    def test_import_over_64k_limit_rejected_with_413(self, auth_client):
        # 65,537 bytes (64 KiB + 1 byte)
        payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Oversized Import",
            "steps": [
                {
                    "op": "filter_equals",
                    "column": "col",
                    "value": "X" * 65500,
                }
            ],
        }
        raw_bytes = json.dumps(payload).encode("utf-8")
        assert len(raw_bytes) > 65536

        response = auth_client.post(
            "/api/v1/workflows/import/",
            raw_bytes,
            content_type="application/json",
        )
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_import_missing_content_length_oversized_rejected_with_413(self, auth_client):
        # Stream or raw body where Content-Length is omitted or large
        oversized_data = b" " * 70000
        # Post with WSGI environ lacking CONTENT_LENGTH
        response = auth_client.generic(
            "POST",
            "/api/v1/workflows/import/",
            data=oversized_data,
            content_type="application/json",
        )
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_import_inaccurate_small_content_length_bounded_stream_rejected_with_413(
        self, auth_client
    ):
        # Header claims 10 bytes, but body sends 70,000 bytes.
        # Bounded stream reader must read max 65537 bytes and return 413, not crash or accept.
        oversized_data = b" " * 70000
        response = auth_client.generic(
            "POST",
            "/api/v1/workflows/import/",
            data=oversized_data,
            content_type="application/json",
            CONTENT_LENGTH="10",
        )
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_import_unicode_multibyte_exact_byte_boundary(self, auth_client):
        # Multi-byte UTF-8 characters (e.g. 3-byte '€' or 4-byte '🚀')
        # Ensure exact byte measurement matches 65536 vs 65537
        base_payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Unicode Boundary",
            "steps": [{"op": "select_columns", "columns": ["id"]}],
        }
        step_template = {
            "op": "filter_equals",
            "column": "col",
            "value": "€",  # 3 bytes in UTF-8
        }
        base_payload["steps"].append(step_template)
        base_bytes = json.dumps(base_payload).encode("utf-8")
        pad_len = 65536 - len(base_bytes)
        # Pad with 1-byte ASCII characters 'A'
        base_payload["steps"][1]["value"] = "€" + ("A" * pad_len)
        exact_bytes = json.dumps(base_payload).encode("utf-8")
        assert len(exact_bytes) == 65536

        res_ok = auth_client.post(
            "/api/v1/workflows/import/", exact_bytes, content_type="application/json"
        )
        assert res_ok.status_code == status.HTTP_201_CREATED

        # Add 1 byte to make it 65537
        over_bytes = exact_bytes + b" "
        assert len(over_bytes) == 65537
        res_over = auth_client.post(
            "/api/v1/workflows/import/", over_bytes, content_type="application/json"
        )
        assert res_over.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_import_rejects_unsupported_content_type(self, auth_client):
        response = auth_client.generic(
            "POST",
            "/api/v1/workflows/import/",
            data=b"not json",
            content_type="text/plain",
        )
        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def test_import_rejects_nan_and_infinity_literals(self, auth_client):
        # Test NaN in filter_equals value
        payload_nan = (
            b'{"format": "flowforge.workflow", "format_version": 1, "name": "NaN Test", '
            b'"steps": [{"op": "filter_equals", "column": "num", "value": NaN}]}'
        )
        res_nan = auth_client.post(
            "/api/v1/workflows/import/", payload_nan, content_type="application/json"
        )
        assert res_nan.status_code == status.HTTP_400_BAD_REQUEST
        assert "Non-standard JSON constant not allowed" in str(res_nan.data) or "Malformed" in str(
            res_nan.data
        )

        # Test Infinity in filter_equals value
        payload_inf = (
            b'{"format": "flowforge.workflow", "format_version": 1, "name": "Inf Test", '
            b'"steps": [{"op": "filter_equals", "column": "num", "value": Infinity}]}'
        )
        res_inf = auth_client.post(
            "/api/v1/workflows/import/", payload_inf, content_type="application/json"
        )
        assert res_inf.status_code == status.HTTP_400_BAD_REQUEST

        # Test -Infinity in filter_equals value
        payload_ninf = (
            b'{"format": "flowforge.workflow", "format_version": 1, "name": "-Inf Test", '
            b'"steps": [{"op": "filter_equals", "column": "num", "value": -Infinity}]}'
        )
        res_ninf = auth_client.post(
            "/api/v1/workflows/import/", payload_ninf, content_type="application/json"
        )
        assert res_ninf.status_code == status.HTTP_400_BAD_REQUEST

    def test_import_and_export_preserve_valid_finite_floats_and_numbers(self, auth_client):
        # Valid numbers: 0, 42, -15, 3.14159, 1e-5, leading-zero strings "007"
        payload = {
            "format": "flowforge.workflow",
            "format_version": 1,
            "name": "Numbers Test",
            "steps": [
                {"op": "select_columns", "columns": ["id", "val"]},
                {"op": "filter_equals", "column": "val", "value": 3.14159},
            ],
        }
        res_imp = auth_client.post(
            "/api/v1/workflows/import/", json.dumps(payload), content_type="application/json"
        )
        assert res_imp.status_code == status.HTTP_201_CREATED
        created_id = res_imp.data["id"]
        assert res_imp.data["steps"][1]["value"] == 3.14159

        # Export and verify the float is preserved
        res_exp = auth_client.get(f"/api/v1/workflows/{created_id}/export/")
        assert res_exp.status_code == status.HTTP_200_OK
        exp_data = res_exp.json()
        assert exp_data["steps"][1]["value"] == 3.14159

    def test_export_corrupt_legacy_non_finite_float_fails_with_400(self, auth_client, monkeypatch):
        # Create a valid workflow first, then simulate legacy in-memory non-finite float
        valid_wf = WorkflowDefinition.objects.create(
            owner=auth_client.user,
            name="Legacy NaN WF",
            steps=[
                {"op": "select_columns", "columns": ["id"]},
                {"op": "filter_equals", "column": "score", "value": 1.0},
            ],
        )

        orig_select = WorkflowDefinition.objects.select_for_update

        def mock_select(*args, **kwargs):
            qs = orig_select(*args, **kwargs)
            orig_get = qs.get

            def mock_get(*g_args, **g_kwargs):
                wf = orig_get(*g_args, **g_kwargs)
                # Inject legacy NaN in memory after loading
                wf.steps[1]["value"] = float("nan")
                return wf

            qs.get = mock_get
            return qs

        monkeypatch.setattr(WorkflowDefinition.objects, "select_for_update", mock_select)

        res = auth_client.get(f"/api/v1/workflows/{valid_wf.id}/export/")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "finite" in str(res.data).lower() or "corrupt" in str(res.data).lower()
