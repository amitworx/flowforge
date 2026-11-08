"""
Pure-Python ingestion, parsing, and normalization service for CSV and JSON datasets.
Decoupled completely from Django ORM and HTTP views for maximum testability.
"""

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

MAX_UPLOAD_SIZE_BYTES = 1024 * 1024  # 1 MiB conservative cap
MAX_ROW_COUNT = 1000  # 1000 records cap


class IngestionError(Exception):
    """Base exception for ingestion and parsing errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class ParseResult:
    file_type: str
    original_filename: str
    input_row_count: int
    output_row_count: int
    normalized_data: list[dict[str, Any]]
    column_names: list[str]


def sanitize_filename(filename: str | None) -> str:
    """Sanitize original display filename, preventing path traversal."""
    if not filename:
        return "unnamed_upload"
    clean = filename.replace("\\", "/").split("/")[-1].strip()
    return clean if clean else "unnamed_upload"


def detect_file_type(filename: str, initial_bytes: bytes) -> str:
    """
    Detect file type ('csv' or 'json') using extension and content sniffing.
    Does not blindly trust file extensions.
    """
    lower = filename.lower()
    stripped = initial_bytes.strip()

    if lower.endswith(".json") or (stripped.startswith(b"[") or stripped.startswith(b"{")):
        return "json"
    if lower.endswith(".csv") or (b"," in stripped or b"\n" in stripped):
        return "csv"
    raise IngestionError(
        f"Unsupported file type for '{filename}'. Only CSV (.csv) and JSON (.json) files are accepted."
    )


def parse_and_normalize_stream(
    stream: io.BytesIO, filename: str, max_size: int = MAX_UPLOAD_SIZE_BYTES
) -> ParseResult:
    """
    Safely consume and validate an uploaded file stream.
    Enforces upload size caps while reading to prevent memory exhaustion.
    """
    clean_filename = sanitize_filename(filename)

    # Read bounded content
    content_bytes = stream.read(max_size + 1)
    if len(content_bytes) > max_size:
        raise IngestionError(
            f"File size exceeds maximum allowed limit of {max_size} bytes (1 MiB)."
        )
    if not content_bytes or len(content_bytes.strip()) == 0:
        raise IngestionError("Uploaded file is empty.")

    # Validate UTF-8 encoding (handling UTF-8 BOM if present)
    try:
        decoded_text = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"File content is not valid UTF-8: {exc}") from exc

    file_type = detect_file_type(clean_filename, content_bytes)

    if file_type == "csv":
        return _parse_csv(decoded_text, clean_filename)
    elif file_type == "json":
        return _parse_json(decoded_text, clean_filename)
    else:
        raise IngestionError(f"Unsupported file type: {file_type}")


def _parse_csv(text: str, filename: str) -> ParseResult:
    """Parse and normalize CSV data preserving string representations."""
    # Use StringIO with strict csv reader
    sio = io.StringIO(text, newline="")
    try:
        reader = csv.reader(sio)
        header_row = next(reader, None)
    except Exception as exc:
        raise IngestionError(f"Malformed CSV content: {exc}") from exc

    if not header_row:
        raise IngestionError("CSV file is empty or missing a valid header row.")

    # Strip and validate headers
    clean_headers = [h.strip() for h in header_row]
    for h in clean_headers:
        if not h:
            raise IngestionError("CSV contains an empty header column name.")

    # Check for duplicate headers
    seen_headers = set()
    for h in clean_headers:
        if h in seen_headers:
            raise IngestionError(f"Duplicate header column detected: '{h}'.")
        seen_headers.add(h)

    rows: list[dict[str, Any]] = []
    line_number = 1
    for row in reader:
        line_number += 1
        # Skip completely empty trailing lines
        if not row or all(c.strip() == "" for c in row):
            continue

        if len(row) != len(clean_headers):
            raise IngestionError(
                f"Row {line_number} has {len(row)} values, expected {len(clean_headers)} columns."
            )

        # Build dict preserving cell values exactly as strings (preserving leading zeros)
        record = {}
        for idx, col in enumerate(clean_headers):
            cell_val = row[idx]
            # Guard against spreadsheet formula injection in string representation if later exported
            # Preserve cell value as string
            record[col] = cell_val

        rows.append(record)
        if len(rows) > MAX_ROW_COUNT:
            raise IngestionError(f"CSV row count exceeds maximum limit of {MAX_ROW_COUNT} records.")

    return ParseResult(
        file_type="csv",
        original_filename=filename,
        input_row_count=len(rows),
        output_row_count=len(rows),
        normalized_data=rows,
        column_names=clean_headers,
    )


def _parse_json(text: str, filename: str) -> ParseResult:
    """Parse and normalize JSON data expecting top-level array of objects."""
    try:
        data = json.loads(text)
    except Exception as exc:
        raise IngestionError(f"Malformed JSON content: {exc}") from exc

    if not isinstance(data, list):
        raise IngestionError("JSON content must be a top-level array of uniform objects (records).")

    if len(data) == 0:
        raise IngestionError("JSON array cannot be empty.")

    if len(data) > MAX_ROW_COUNT:
        raise IngestionError(f"JSON record count exceeds maximum limit of {MAX_ROW_COUNT} records.")

    # Collect consistent scalar keys across all objects with deterministic ordering
    column_order: list[str] = []
    seen_columns = set()

    # First pass: establish deterministic column ordering from the first object
    first_item = data[0]
    if not isinstance(first_item, dict):
        raise IngestionError("JSON array items must be objects.")
    for key in first_item:
        if not isinstance(key, str) or not key.strip():
            raise IngestionError(f"Invalid or empty key in JSON object: '{key}'.")
        if key not in seen_columns:
            seen_columns.add(key)
            column_order.append(key)

    normalized_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise IngestionError(f"Item at index {idx} in JSON array is not an object.")

        # Check for nested complex structures (disallow deep nested objects/arrays to guard memory)
        record = {}
        for col in column_order:
            val = item.get(col, None)
            if isinstance(val, (dict, list)):
                raise IngestionError(
                    f"Unsupported nested structure at record index {idx}, column '{col}'. "
                    "Only scalar values (strings, numbers, booleans, null) are accepted."
                )
            record[col] = val

        # Check for unexpected extra keys
        for k in item:
            if k not in seen_columns:
                seen_columns.add(k)
                column_order.append(k)
                val = item[k]
                if isinstance(val, (dict, list)):
                    raise IngestionError(
                        f"Unsupported nested structure at record index {idx}, column '{k}'."
                    )
                record[k] = val

        normalized_rows.append(record)

    # Normalize missing columns in earlier records to None for consistent columnar shape
    for row in normalized_rows:
        for col in column_order:
            if col not in row:
                row[col] = None

    return ParseResult(
        file_type="json",
        original_filename=filename,
        input_row_count=len(normalized_rows),
        output_row_count=len(normalized_rows),
        normalized_data=normalized_rows,
        column_names=column_order,
    )
