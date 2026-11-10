"""Pure-Python serialization, formatting, and security defenses for WorkflowRun exports (CSV and JSON)."""

import csv
import io
import json
import unicodedata
from typing import Any

# Default maximum exported payload size: 10 MiB (sensible guard given 1 MiB / 1000 row limits)
DEFAULT_MAX_EXPORT_SIZE_BYTES = 10 * 1024 * 1024

# Characters that directly trigger spreadsheet formula execution if present at the start of a cell
DANGEROUS_FORMULA_TRIGGERS = ("=", "+", "-", "@")

# Direct prefixes that may cause spreadsheet command execution or delimiter confusion
DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n", "\x0b", "\x0c")


class ExportPayloadTooLargeError(Exception):
    """Raised when serialized export payload exceeds the maximum permitted byte length."""

    def __init__(self, message: str = "Export payload exceeds maximum permitted byte size."):
        super().__init__(message)
        self.message = message


def _is_ignorable_or_whitespace_or_control(c: str) -> bool:
    """
    Check if a character is Unicode whitespace, control, or formatting/invisible character
    that spreadsheet software may ignore, strip, or normalize prior to formula evaluation.

    Includes:
    - Cc: Control characters (0x00-0x1F, 0x7F-0x9F, including vertical tab \\x0b, form feed \\x0c)
    - Cf: Format / invisible characters (BOM U+FEFF, Zero-width spaces U+200B..U+200D, direction marks U+200E..U+200F, word joiner U+2060, soft hyphen U+00AD)
    - Zs: Space separators (ASCII space, non-breaking space U+00A0, en/em spaces U+2000..U+200A, ideographic space U+3000)
    - Zl: Line separator (U+2028)
    - Zp: Paragraph separator (U+2029)
    - Any character where c.isspace() evaluates to True
    """
    cat = unicodedata.category(c)
    return cat in ("Zs", "Zl", "Zp", "Cc", "Cf") or c.isspace()


def sanitize_csv_cell(value: Any) -> str:
    """
    Format a scalar value into a CSV-safe string.
    Protects against Formula / CSV Injection (DDE attacks).

    Threat Model & Policy:
    Spreadsheet software (e.g. Microsoft Excel, LibreOffice Calc, Google Sheets) may evaluate
    a cell as a formula if its effective leading character is '=', '+', '-', or '@'.
    Attackers frequently conceal these characters behind whitespace, control characters (vertical tab, form feed),
    or zero-width / BOM formatting characters (e.g. U+FEFF, U+200B..U+200F).
    Furthermore, leading tabs (\\t), carriage returns (\\r), line feeds (\\n), vertical tabs (\\x0b),
    and form feeds (\\x0c) themselves present formula or command injection risks.

    Mitigation:
    If the string representation begins with any dangerous prefix, OR if inspecting past any
    leading Unicode whitespace, control, or format characters reveals a formula trigger (=, +, -, @),
    the string is safely prefixed with a single quote (').
    This instructs spreadsheet clients to interpret the cell strictly as literal text.

    Preservation guarantees:
    - Numeric scalars (int, float) are safe typed numbers and formatted as str(value).
    - Null values format as empty string ("").
    - Booleans format as "true" or "false".
    - Strings with leading zeros (e.g. "00123") are preserved exactly.
    - Stored database content is never mutated.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    s = str(value)

    # Scan past any leading whitespace, control, or formatting characters
    idx = 0
    while idx < len(s) and _is_ignorable_or_whitespace_or_control(s[idx]):
        idx += 1

    # Check direct dangerous prefix or concealed formula trigger
    if s.startswith(DANGEROUS_PREFIXES) or (idx < len(s) and s[idx] in DANGEROUS_FORMULA_TRIGGERS):
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
    max_bytes: int = DEFAULT_MAX_EXPORT_SIZE_BYTES,
) -> str:
    """
    Export workflow run data to an RFC 4180 compliant CSV string.
    Safely sanitizes both headers and data cells to mitigate CSV formula injection.
    When output_data is empty, produces a header-only CSV using output_columns.
    Never mutates output_data or output_columns in place.

    Enforces max_bytes UTF-8 limit incrementally during serialization to bound memory.
    Raises ExportPayloadTooLargeError if UTF-8 encoded output exceeds max_bytes.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")

    current_byte_count = 0

    # Sanitize and write header row
    sanitized_headers = [sanitize_csv_cell(col) for col in output_columns]
    writer.writerow(sanitized_headers)

    header_bytes = len(buffer.getvalue().encode("utf-8"))
    if header_bytes > max_bytes:
        raise ExportPayloadTooLargeError(
            f"CSV export size ({header_bytes} bytes) exceeds maximum limit of {max_bytes} bytes."
        )
    current_byte_count = header_bytes

    # Write data rows incrementally, tracking byte length
    for row in output_data:
        row_buffer = io.StringIO()
        row_writer = csv.writer(row_buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
        row_values = [sanitize_csv_cell(row.get(col)) for col in output_columns]
        row_writer.writerow(row_values)
        row_str = row_buffer.getvalue()
        row_bytes = len(row_str.encode("utf-8"))

        current_byte_count += row_bytes
        if current_byte_count > max_bytes:
            raise ExportPayloadTooLargeError(
                f"CSV export size exceeds maximum limit of {max_bytes} bytes."
            )
        buffer.write(row_str)

    return buffer.getvalue()


def export_workflow_run_to_json(
    output_data: list[dict[str, Any]],
    max_bytes: int = DEFAULT_MAX_EXPORT_SIZE_BYTES,
) -> bytes:
    """
    Export workflow run data to a deterministic, UTF-8 encoded JSON byte string.
    Preserves scalar types, strings with leading zeros, Unicode, numbers, booleans, and nulls without modification.
    Enforces max_bytes UTF-8 limit, raising ExportPayloadTooLargeError if exceeded.
    """
    json_str = json.dumps(output_data, ensure_ascii=False, indent=2)
    raw_bytes = json_str.encode("utf-8")
    if len(raw_bytes) > max_bytes:
        raise ExportPayloadTooLargeError(
            f"JSON export size ({len(raw_bytes)} bytes) exceeds maximum limit of {max_bytes} bytes."
        )
    return raw_bytes
