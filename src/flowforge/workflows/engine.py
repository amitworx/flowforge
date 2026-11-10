"""Pure Python deterministic workflow execution engine."""

from typing import Any


class WorkflowExecutionError(Exception):
    def __init__(
        self,
        message: str,
        step_index: int | None = None,
        completed_step_audit: list[dict[str, Any]] | None = None,
        failed_step_audit: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.step_index = step_index
        self.completed_step_audit = completed_step_audit or []
        self.failed_step_audit = failed_step_audit


MAX_STEPS = 10
MAX_ROWS = 1000
MAX_NAME_LENGTH = 128
ALLOWED_OPERATIONS = {"select_columns", "rename_columns", "filter_equals"}
STEP_ALLOWED_FIELDS = {
    "select_columns": {"op", "columns"},
    "rename_columns": {"op", "mapping"},
    "filter_equals": {"op", "column", "value"},
}


def _is_valid_scalar(val: Any) -> bool:
    """Scalar types: str, int, float, bool, None."""
    return isinstance(val, (str, int, float, bool)) or val is None


def _is_exact_scalar_equal(a: Any, b: Any) -> bool:
    """
    Type-sensitive scalar comparison:
    - If either value is a boolean, both must be boolean and equal.
      (Prevents Python's True == 1 and False == 0 trap).
    - If both are numbers (int or float), numeric equality applies (e.g. 1 == 1.0).
    - If both are strings, exact string equality applies (preserving leading zeros).
    - None matches only None.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return False


def validate_step_definitions(steps: Any) -> list[dict[str, Any]]:
    """Validate definition-time step schemas."""
    if not isinstance(steps, list):
        raise WorkflowExecutionError("Steps must be a list of step definitions.")
    if not steps:
        raise WorkflowExecutionError("Workflow must contain at least one step.")
    if len(steps) > MAX_STEPS:
        raise WorkflowExecutionError(f"Workflow cannot exceed maximum {MAX_STEPS} steps.")

    validated = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise WorkflowExecutionError(f"Step {idx} must be a dictionary.")

        op = step.get("op")
        if not op or not isinstance(op, str) or op not in ALLOWED_OPERATIONS:
            raise WorkflowExecutionError(
                f"Step {idx}: Unsupported operation '{op}'. Supported operations are: {sorted(ALLOWED_OPERATIONS)}"
            )

        allowed_fields = STEP_ALLOWED_FIELDS[op]
        extra_fields = set(step.keys()) - allowed_fields
        if extra_fields:
            raise WorkflowExecutionError(
                f"Step {idx} ({op}): Unexpected fields {sorted(extra_fields)}. Allowed fields: {sorted(allowed_fields)}."
            )

        if op == "select_columns":
            columns = step.get("columns")
            if not isinstance(columns, list) or not columns:
                raise WorkflowExecutionError(
                    f"Step {idx} (select_columns): 'columns' must be a non-empty list of column names."
                )
            for c in columns:
                if not isinstance(c, str) or not c.strip() or len(c) > MAX_NAME_LENGTH:
                    raise WorkflowExecutionError(
                        f"Step {idx} (select_columns): Invalid column name '{c}'."
                    )
            if len(set(columns)) != len(columns):
                raise WorkflowExecutionError(
                    f"Step {idx} (select_columns): Duplicate column names in selection."
                )

        elif op == "rename_columns":
            mapping = step.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise WorkflowExecutionError(
                    f"Step {idx} (rename_columns): 'mapping' must be a non-empty dictionary."
                )
            for k, v in mapping.items():
                if (
                    not isinstance(k, str)
                    or not k.strip()
                    or not isinstance(v, str)
                    or not v.strip()
                    or len(k) > MAX_NAME_LENGTH
                    or len(v) > MAX_NAME_LENGTH
                ):
                    raise WorkflowExecutionError(
                        f"Step {idx} (rename_columns): Invalid column mapping pair '{k}': '{v}'."
                    )
            # Check target collisions
            targets = list(mapping.values())
            if len(set(targets)) != len(targets):
                raise WorkflowExecutionError(
                    f"Step {idx} (rename_columns): Mapping contains duplicate target column collision."
                )

        elif op == "filter_equals":
            col = step.get("column")
            if not isinstance(col, str) or not col.strip() or len(col) > MAX_NAME_LENGTH:
                raise WorkflowExecutionError(
                    f"Step {idx} (filter_equals): 'column' must be a valid non-empty string."
                )
            if "value" not in step:
                raise WorkflowExecutionError(
                    f"Step {idx} (filter_equals): Missing required 'value' field."
                )
            val = step.get("value")
            if not _is_valid_scalar(val):
                raise WorkflowExecutionError(
                    f"Step {idx} (filter_equals): 'value' must be a scalar (string, number, boolean, or null)."
                )

        validated.append(step)
    return validated


def execute_step(
    data: list[dict[str, Any]],
    step: dict[str, Any],
    schema: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute a single deterministic step on data, propagating and enforcing schema.
    Returns transformed data as a list of dicts.
    If schema is provided, modifies schema in-place (or caller uses execute_step_with_schema).
    """
    transformed_data, new_schema = execute_step_with_schema(data, step, schema)
    if schema is not None:
        schema.clear()
        schema.extend(new_schema)
    return transformed_data


def execute_step_with_schema(
    data: list[dict[str, Any]],
    step: dict[str, Any],
    schema: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Execute a single deterministic step on data with explicit schema tracking.
    Returns (transformed_data, new_schema).
    """
    op = step["op"]

    # Check input row count
    if len(data) > MAX_ROWS:
        raise WorkflowExecutionError(f"Input data exceeds maximum limit of {MAX_ROWS} rows.")

    # Determine active columns
    if schema is not None:
        existing_keys = list(schema)
    elif data:
        existing_keys = list(data[0].keys())
    else:
        existing_keys = []

    if op == "select_columns":
        columns = step["columns"]
        # Validate that all requested columns exist in active schema
        for col in columns:
            if col not in existing_keys:
                raise WorkflowExecutionError(
                    f"Column '{col}' requested in select_columns does not exist in input data."
                )
        new_data = [{col: row[col] for col in columns} for row in data]
        return new_data, list(columns)

    elif op == "rename_columns":
        mapping = step["mapping"]
        # Validate source columns exist in active schema
        for old_col in mapping:
            if old_col not in existing_keys:
                raise WorkflowExecutionError(
                    f"Column '{old_col}' to rename does not exist in input data."
                )

        # Check collision with existing unrenamed columns
        unrenamed_keys = set(existing_keys) - set(mapping.keys())
        for _old_col, new_col in mapping.items():
            if new_col in unrenamed_keys:
                raise WorkflowExecutionError(
                    f"Rename collision: target column '{new_col}' already exists in input data."
                )

        new_schema = [mapping.get(k, k) for k in existing_keys]

        new_data = []
        for row in data:
            new_row = {}
            for k, v in row.items():
                target_k = mapping.get(k, k)
                new_row[target_k] = v
            new_data.append(new_row)
        return new_data, new_schema

    elif op == "filter_equals":
        col = step["column"]
        target_val = step["value"]
        if col not in existing_keys:
            raise WorkflowExecutionError(f"Column '{col}' to filter does not exist in input data.")

        new_data = []
        for row in data:
            row_val = row.get(col)
            if _is_exact_scalar_equal(row_val, target_val):
                new_data.append(dict(row))
        return new_data, list(existing_keys)

    raise WorkflowExecutionError(f"Unsupported operation '{op}'")


def execute_workflow_pipeline_with_schema(
    data: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    schema: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    Execute an ordered pipeline of steps against data.
    Propagates deterministic schema across all steps, even if intermediate data becomes empty.
    Returns (output_data, step_audits, final_schema).
    Raises WorkflowExecutionError on failure, preserving completed step audits.
    """
    validate_step_definitions(steps)

    current_data = [dict(row) for row in data]
    if schema is not None:
        current_schema = list(schema)
    elif current_data:
        current_schema = list(current_data[0].keys())
    else:
        current_schema = []

    completed_audits = []

    for idx, step in enumerate(steps):
        input_count = len(current_data)
        op = step["op"]
        try:
            current_data, current_schema = execute_step_with_schema(
                current_data, step, schema=current_schema
            )
            output_count = len(current_data)
            completed_audits.append(
                {
                    "step_index": idx,
                    "operation": op,
                    "status": "completed",
                    "input_row_count": input_count,
                    "output_row_count": output_count,
                    "error_summary": None,
                }
            )
        except WorkflowExecutionError as exc:
            failed_audit = {
                "step_index": idx,
                "operation": op,
                "status": "failed",
                "input_row_count": input_count,
                "output_row_count": 0,
                "error_summary": exc.message,
            }
            raise WorkflowExecutionError(
                message=exc.message,
                step_index=idx,
                completed_step_audit=completed_audits,
                failed_step_audit=failed_audit,
            ) from exc

    return current_data, completed_audits, current_schema


def execute_workflow_pipeline(
    data: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    schema: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Execute an ordered pipeline of steps against data.
    Preserves backward compatibility returning (output_data, step_audits).
    """
    output_data, audits, _schema = execute_workflow_pipeline_with_schema(data, steps, schema=schema)
    return output_data, audits
