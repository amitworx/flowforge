"""Pure-Python serialization, formatting, and security defenses for WorkflowRun exports (CSV and JSON)."""

import csv
import io
import json
from typing import Any

# Characters that trigger spreadsheet formula execution if present at the start of a cell
DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def sanitize_csv_cell(value: Any) -> str:
    """
    Format a scalar value into a CSV-safe string.
    Protects against Formula / CSV Injection (DDE attacks).

    If the string representation of a value begins with a formula prefix
    (=, +, -, @, \\t, \\r, \\n), or has leading whitespace/control chars concealing
    one, it is safely prefixed with a single quote (').
    Null values become empty strings ("").
    Booleans become "true" or "false".
    Strings with leading zeros are preserved exactly as strings.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Numeric scalars are inherently numbers, but negative numbers shouldn't trigger formula warnings
        # unless treated as text. However, standard numbers like -5 or 42 are safe unless formatted as text starting with '-'.
        # Python csv writer handles numbers cleanly.
        # Check if negative number: -5 is safe mathematically, but in CSV text -5 could be interpreted by spreadsheets.
        # Standard: plain ints/floats are formatted directly as str(value).
        return str(value)

    # String values
    s = str(value)
    # Check if stripping leading whitespace/control characters reveals a dangerous formula prefix,
    # or if the raw string starts with a dangerous prefix (e.g. tabs or carriage returns)
    stripped = s.lstrip(" \t\r\n")
    if s.startswith(("\t", "\r", "\n")) or (stripped and stripped.startswith(("=", "+", "-", "@"))):
        return f"'{s}"

    return s


def derive_workflow_schema(initial_schema: list[str], steps: list[dict[str, Any]]) -> list[str]:
    """
    Deterministically trace column transformations through an immutable sequence of steps.
    Supports select_columns, rename_columns, and filter_equals.
    """
    schema = list(initial_schema)
    for step in steps:
        op = step.get("op")
        if op == "select_columns":
            columns = step.get("columns", [])
            schema = [c for c in columns if c in schema]
        elif op == "rename_columns":
            mapping = step.get("mapping", {})
            schema = [mapping.get(c, c) for c in schema]
        elif op == "filter_equals":
            # filter_equals does not alter the schema columns
            pass
    return schema


def export_workflow_run_to_csv(
    output_data: list[dict[str, Any]],
    output_columns: list[str],
) -> str:
    """
    Export workflow run data to an RFC 4180 compliant CSV string.
    Safely sanitizes both headers and data cells to mitigate CSV formula injection.
    When output_data is empty, produces a header-only CSV using output_columns.
    Never mutates output_data or output_columns in place.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")

    # Sanitize and write header row
    sanitized_headers = [sanitize_csv_cell(col) for col in output_columns]
    writer.writerow(sanitized_headers)

    # Write data rows
    for row in output_data:
        row_values = [sanitize_csv_cell(row.get(col)) for col in output_columns]
        writer.writerow(row_values)

    return buffer.getvalue()


def export_workflow_run_to_json(output_data: list[dict[str, Any]]) -> bytes:
    """
    Export workflow run data to a deterministic, UTF-8 encoded JSON byte string.
    Preserves scalar types, strings with leading zeros, Unicode, numbers, booleans, and nulls without modification.
    """
    json_str = json.dumps(output_data, ensure_ascii=False, indent=2)
    return json_str.encode("utf-8")
