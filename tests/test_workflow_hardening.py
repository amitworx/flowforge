import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_step,
    execute_workflow_pipeline,
    validate_step_definitions,
)
from flowforge.workflows.models import WorkflowDefinition


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def test_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="testuser", password="password123")


# --- Finding 1: Unhashable malformed inputs ---


def test_validate_step_definitions_rejects_unhashable_nested_select_columns():
    """select_columns with nested list or dict in columns must raise WorkflowExecutionError, not TypeError."""
    steps = [{"op": "select_columns", "columns": [["nested_list"], "id"]}]
    with pytest.raises(WorkflowExecutionError, match="Invalid column name"):
        validate_step_definitions(steps)

    steps_dict = [{"op": "select_columns", "columns": [{"nested": "dict"}]}]
    with pytest.raises(WorkflowExecutionError, match="Invalid column name"):
        validate_step_definitions(steps_dict)


def test_validate_step_definitions_rejects_unhashable_nested_rename_targets():
    """rename_columns with nested dict or list as target values must raise WorkflowExecutionError, not TypeError."""
    steps = [{"op": "rename_columns", "mapping": {"old_col": ["nested_target"]}}]
    with pytest.raises(WorkflowExecutionError, match="Invalid column mapping"):
        validate_step_definitions(steps)

    steps_dict = [{"op": "rename_columns", "mapping": {"old_col": {"nested": "target"}}}]
    with pytest.raises(WorkflowExecutionError, match="Invalid column mapping"):
        validate_step_definitions(steps_dict)


def test_validate_step_definitions_rejects_unknown_extra_fields():
    """Step definitions containing unexpected fields must be rejected."""
    steps = [{"op": "select_columns", "columns": ["id"], "unexpected_field": "exploit"}]
    with pytest.raises(WorkflowExecutionError, match="Unexpected fields"):
        validate_step_definitions(steps)


@pytest.mark.django_db
def test_api_rejects_malformed_nested_structures_with_clean_400(api_client, test_user):
    """API must return HTTP 400 Bad Request, never 500 Internal Server Error, for unhashable payloads."""
    api_client.force_authenticate(user=test_user)

    payload_nested_select = {
        "name": "Nested Select Exploit",
        "steps": [{"op": "select_columns", "columns": [["nested"], {"a": "b"}]}],
    }
    res1 = api_client.post("/api/v1/workflows/", data=payload_nested_select, format="json")
    assert res1.status_code == status.HTTP_400_BAD_REQUEST

    payload_nested_rename = {
        "name": "Nested Rename Exploit",
        "steps": [{"op": "rename_columns", "mapping": {"old": {"bad": "target"}}}],
    }
    res2 = api_client.post("/api/v1/workflows/", data=payload_nested_rename, format="json")
    assert res2.status_code == status.HTTP_400_BAD_REQUEST


# --- Finding 2: Type-sensitive filter_equals (boolean vs int, null, float) ---


def test_filter_equals_type_sensitive_boolean_vs_number():
    """
    filter_equals must strictly distinguish boolean True/False from integer 1/0.
    In Python, True == 1, but in structured data filtering, True must NOT match 1.
    """
    data = [
        {"id": 1, "is_active": True, "val": 1},
        {"id": 2, "is_active": False, "val": 0},
        {"id": 3, "is_active": 1, "val": 1},
        {"id": 4, "is_active": 0, "val": 0},
    ]

    # Filter where is_active is boolean True
    step_bool = {"op": "filter_equals", "column": "is_active", "value": True}
    res_bool = execute_step(data, step_bool)
    assert len(res_bool) == 1
    assert res_bool[0]["id"] == 1

    # Filter where is_active is integer 1
    step_int = {"op": "filter_equals", "column": "is_active", "value": 1}
    res_int = execute_step(data, step_int)
    assert len(res_int) == 1
    assert res_int[0]["id"] == 3

    # Filter where is_active is boolean False
    step_false = {"op": "filter_equals", "column": "is_active", "value": False}
    res_false = execute_step(data, step_false)
    assert len(res_false) == 1
    assert res_false[0]["id"] == 2

    # Filter where is_active is integer 0
    step_zero = {"op": "filter_equals", "column": "is_active", "value": 0}
    res_zero = execute_step(data, step_zero)
    assert len(res_zero) == 1
    assert res_zero[0]["id"] == 4


def test_filter_equals_null_handling():
    """filter_equals with null (None) must strictly match only None values."""
    data = [
        {"id": 1, "desc": None},
        {"id": 2, "desc": ""},
        {"id": 3, "desc": 0},
        {"id": 4, "desc": False},
    ]
    step = {"op": "filter_equals", "column": "desc", "value": None}
    res = execute_step(data, step)
    assert len(res) == 1
    assert res[0]["id"] == 1


# --- Finding 3: Empty results bypass runtime column/schema validation ---


def test_empty_results_validate_subsequent_step_schemas():
    """
    When an intermediate step filters out all records (0 rows remaining),
    subsequent steps MUST STILL validate their column references against the active schema.
    """
    data = [
        {"id": "001", "name": "Alice", "status": "inactive"},
    ]
    initial_schema = ["id", "name", "status"]

    # Step 0: filter_equals status == 'active' -> produces 0 rows!
    # Step 1: select_columns nonexistent 'ghost_column' -> MUST FAIL with WorkflowExecutionError!
    steps_invalid = [
        {"op": "filter_equals", "column": "status", "value": "active"},
        {"op": "select_columns", "columns": ["ghost_column"]},
    ]

    with pytest.raises(WorkflowExecutionError, match="does not exist"):
        execute_workflow_pipeline(data, steps_invalid, schema=initial_schema)


def test_empty_results_valid_subsequent_step_schemas_succeed():
    """
    When an intermediate step produces 0 rows, subsequent valid steps
    transform the schema deterministically and produce an empty output with completed audits.
    """
    data = [
        {"id": "001", "name": "Alice", "status": "inactive"},
    ]
    initial_schema = ["id", "name", "status"]

    # Step 0: filter_equals status == 'active' -> 0 rows
    # Step 1: select_columns 'id', 'name' -> valid schema transformation
    # Step 2: rename_columns 'name' -> 'full_name' -> valid schema transformation
    steps_valid = [
        {"op": "filter_equals", "column": "status", "value": "active"},
        {"op": "select_columns", "columns": ["id", "name"]},
        {"op": "rename_columns", "mapping": {"name": "full_name"}},
    ]

    output, audits = execute_workflow_pipeline(data, steps_valid, schema=initial_schema)
    assert output == []
    assert len(audits) == 3
    assert all(a["status"] == "completed" for a in audits)


@pytest.mark.django_db
def test_api_empty_result_followed_by_invalid_column_persists_failed_run(api_client, test_user):
    """
    Via API: If step 0 produces 0 rows and step 1 references an invalid column,
    the workflow run must durably fail at step 1 and record step 0 completed + step 1 failed.
    """
    api_client.force_authenticate(user=test_user)

    job = ProcessingJob.objects.create(
        owner=test_user,
        file_type="json",
        state=ProcessingJob.State.COMPLETED,
        original_filename="records.json",
        input_row_count=1,
        output_row_count=1,
        column_names=["id", "status"],
        output_data=[{"id": 1, "status": "pending"}],
    )

    wf = WorkflowDefinition.objects.create(
        owner=test_user,
        name="Empty Then Fail",
        steps=[
            {"op": "filter_equals", "column": "status", "value": "active"},  # 0 rows
            {"op": "select_columns", "columns": ["invalid_col"]},  # Must fail
        ],
    )

    res = api_client.post(
        f"/api/v1/workflows/{wf.id}/runs/",
        data={"source_job_id": str(job.id)},
        format="json",
    )
    assert res.status_code == status.HTTP_201_CREATED
    run_data = res.json()
    assert run_data["state"] == "failed"

    detail = api_client.get(f"/api/v1/workflow-runs/{run_data['id']}/").json()
    assert detail["state"] == "failed"
    assert detail["output_data"] == []
    assert len(detail["step_runs"]) == 2
    assert detail["step_runs"][0]["status"] == "completed"
    assert detail["step_runs"][0]["output_row_count"] == 0
    assert detail["step_runs"][1]["status"] == "failed"
    assert "invalid_col" in detail["step_runs"][1]["error_summary"]
