import pytest

from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_step,
    execute_workflow_pipeline,
    validate_step_definitions,
)


def test_select_columns_step_success():
    data = [
        {"id": "001", "name": "Alice", "extra": "foo"},
        {"id": "002", "name": "Bob", "extra": "bar"},
    ]
    step = {"op": "select_columns", "columns": ["name", "id"]}
    result = execute_step(data, step)
    assert result == [
        {"name": "Alice", "id": "001"},
        {"name": "Bob", "id": "002"},
    ]


def test_select_columns_preserves_leading_zeros():
    data = [{"code": "000123", "val": 42}]
    step = {"op": "select_columns", "columns": ["code"]}
    result = execute_step(data, step)
    assert result == [{"code": "000123"}]


def test_rename_columns_step_success():
    data = [{"old_id": "01", "name": "Alice"}]
    step = {"op": "rename_columns", "mapping": {"old_id": "new_id"}}
    result = execute_step(data, step)
    assert result == [{"new_id": "01", "name": "Alice"}]


def test_rename_columns_collision_fails():
    data = [{"col1": "a", "col2": "b"}]
    step = {"op": "rename_columns", "mapping": {"col1": "col2"}}
    with pytest.raises(WorkflowExecutionError, match="collision"):
        execute_step(data, step)


def test_filter_equals_string_and_scalars():
    data = [
        {"id": "001", "status": "active", "count": 10},
        {"id": "002", "status": "inactive", "count": 20},
        {"id": "003", "status": "active", "count": 10},
    ]
    step = {"op": "filter_equals", "column": "status", "value": "active"}
    result = execute_step(data, step)
    assert len(result) == 2
    assert result[0]["id"] == "001"
    assert result[1]["id"] == "003"


def test_filter_equals_empty_result_is_valid():
    data = [{"id": "001", "status": "active"}]
    step = {"op": "filter_equals", "column": "status", "value": "archived"}
    result = execute_step(data, step)
    assert result == []


def test_execute_workflow_pipeline_valid_chain():
    data = [
        {"id": "001", "user_name": "Alice", "status": "active", "dept": "eng"},
        {"id": "002", "user_name": "Bob", "status": "inactive", "dept": "hr"},
        {"id": "003", "user_name": "Charlie", "status": "active", "dept": "eng"},
    ]
    steps = [
        {"op": "select_columns", "columns": ["id", "user_name", "status"]},
        {"op": "rename_columns", "mapping": {"user_name": "name"}},
        {"op": "filter_equals", "column": "status", "value": "active"},
    ]
    result, step_audit = execute_workflow_pipeline(data, steps)
    assert len(result) == 2
    assert result[0] == {"id": "001", "name": "Alice", "status": "active"}
    assert result[1] == {"id": "003", "name": "Charlie", "status": "active"}
    assert len(step_audit) == 3
    assert all(audit["status"] == "completed" for audit in step_audit)


def test_execute_workflow_pipeline_stops_on_failure_and_preserves_audit():
    data = [{"id": "001", "name": "Alice"}]
    steps = [
        {"op": "select_columns", "columns": ["id", "name"]},
        {"op": "rename_columns", "mapping": {"nonexistent": "new_name"}},  # Fails here
        {"op": "filter_equals", "column": "id", "value": "001"},
    ]
    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_workflow_pipeline(data, steps)
    assert exc_info.value.step_index == 1
    assert len(exc_info.value.completed_step_audit) == 1
    assert exc_info.value.completed_step_audit[0]["status"] == "completed"
    assert exc_info.value.failed_step_audit["status"] == "failed"


def test_validate_step_definitions_rejects_more_than_10_steps():
    steps = [{"op": "select_columns", "columns": ["id"]}] * 11
    with pytest.raises(WorkflowExecutionError, match="maximum 10 steps"):
        validate_step_definitions(steps)


def test_validate_step_definitions_rejects_unknown_op():
    steps = [{"op": "drop_database", "columns": ["id"]}]
    with pytest.raises(WorkflowExecutionError, match="Unsupported operation"):
        validate_step_definitions(steps)
