"""Tests for WorkflowRun export functions: CSV and JSON serialization, schema derivation, and formula injection defenses."""

import csv
import io
import json

from flowforge.workflows.export import (
    derive_workflow_schema,
    export_workflow_run_to_csv,
    export_workflow_run_to_json,
    sanitize_csv_cell,
)


def test_sanitize_csv_cell_plain_scalars():
    """Verify standard scalar values format cleanly without modification."""
    assert sanitize_csv_cell("hello") == "hello"
    assert sanitize_csv_cell("00123") == "00123"
    assert sanitize_csv_cell(42) == "42"
    assert sanitize_csv_cell(3.1415) == "3.1415"
    assert sanitize_csv_cell(True) == "true"
    assert sanitize_csv_cell(False) == "false"
    assert sanitize_csv_cell(None) == ""


def test_sanitize_csv_cell_formula_injection_defense():
    """Verify dangerous formula prefixes (=, +, -, @, \\t, \\r) are safely neutralized with a single quote."""
    # Direct formula prefixes
    assert sanitize_csv_cell("=1+1") == "'=1+1"
    assert sanitize_csv_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
    assert sanitize_csv_cell("-2+3*4") == "'-2+3*4"
    assert sanitize_csv_cell("@SUM(A1:A10)") == "'@SUM(A1:A10)"

    # Leading whitespace or control characters concealing formula triggers
    assert sanitize_csv_cell("   =SUM(A1:A5)") == "'   =SUM(A1:A5)"
    assert sanitize_csv_cell("\t=123") == "'\t=123"
    assert sanitize_csv_cell("\r=123") == "'\r=123"
    assert sanitize_csv_cell("\n=123") == "'\n=123"

    # Tab or carriage return prefix itself
    assert sanitize_csv_cell("\tmalicious") == "'\tmalicious"
    assert sanitize_csv_cell("\rmalicious") == "'\rmalicious"


def test_derive_workflow_schema_across_steps():
    """Verify deterministic schema derivation through select, rename, and filter steps."""
    initial_schema = ["id", "account_num", "status", "notes"]
    steps = [
        {"op": "select_columns", "columns": ["account_num", "status", "notes"]},
        {"op": "rename_columns", "mapping": {"account_num": "acct_code", "status": "state"}},
        {"op": "filter_equals", "column": "state", "value": "active"},
    ]
    derived = derive_workflow_schema(initial_schema, steps)
    assert derived == ["acct_code", "state", "notes"]


def test_export_workflow_run_to_csv_zero_rows_emits_header_only():
    """Completed runs with 0 output rows produce a header-only CSV matching the final schema."""
    output_columns = ["customer_id", "total_spend", "category"]
    output_data = []

    csv_text = export_workflow_run_to_csv(output_data=output_data, output_columns=output_columns)
    reader = list(csv.reader(io.StringIO(csv_text)))

    assert len(reader) == 1
    assert reader[0] == ["customer_id", "total_spend", "category"]


def test_export_workflow_run_to_csv_roundtrip_and_escaping():
    """Verify RFC 4180 escaping for quotes, commas, newlines, and preserved leading zeros."""
    output_columns = ["code", "description", "is_valid", "count", "notes"]
    output_data = [
        {
            "code": "00042",
            "description": 'Item with "quotes" and, commas',
            "is_valid": True,
            "count": 10,
            "notes": "Multi-line\nvalue",
        },
        {
            "code": "00099",
            "description": "Plain text",
            "is_valid": False,
            "count": None,
            "notes": None,
        },
    ]

    csv_text = export_workflow_run_to_csv(output_data=output_data, output_columns=output_columns)
    reader = list(csv.reader(io.StringIO(csv_text)))

    assert len(reader) == 3
    assert reader[0] == ["code", "description", "is_valid", "count", "notes"]
    assert reader[1] == [
        "00042",
        'Item with "quotes" and, commas',
        "true",
        "10",
        "Multi-line\nvalue",
    ]
    assert reader[2] == ["00099", "Plain text", "false", "", ""]


def test_export_workflow_run_to_csv_header_and_data_formula_mitigation():
    """Verify headers and data containing formula characters are sanitized without mutating input dicts."""
    output_columns = ["=cmd_header", "+rate", "normal_col"]
    output_data = [
        {
            "=cmd_header": "=SUM(1,2)",
            "+rate": '@HYPERLINK("http://evil.com")',
            "normal_col": "-10% discount",
        }
    ]

    csv_text = export_workflow_run_to_csv(output_data=output_data, output_columns=output_columns)
    reader = list(csv.reader(io.StringIO(csv_text)))

    # Headers sanitized
    assert reader[0] == ["'=cmd_header", "'+rate", "normal_col"]
    # Data sanitized
    assert reader[1] == ["'=SUM(1,2)", '\'@HYPERLINK("http://evil.com")', "'-10% discount"]

    # Original output_data untouched
    assert output_data[0]["=cmd_header"] == "=SUM(1,2)"
    assert output_data[0]["normal_col"] == "-10% discount"


def test_export_workflow_run_to_json():
    """Verify JSON export produces deterministic, unmodified UTF-8 representation."""
    output_data = [
        {
            "code": "00123",
            "active": True,
            "score": 98.6,
            "metadata": None,
            "name": "München Über",
        }
    ]
    raw_json = export_workflow_run_to_json(output_data)
    loaded = json.loads(raw_json.decode("utf-8"))
    assert loaded == output_data
    assert loaded[0]["code"] == "00123"
    assert loaded[0]["name"] == "München Über"
