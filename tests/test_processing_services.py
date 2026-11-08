import io

import pytest

from flowforge.processing.services import (
    IngestionError,
    parse_and_normalize_stream,
    sanitize_filename,
)


def test_sanitize_filename():
    assert sanitize_filename("folder/subfolder/test.csv") == "test.csv"
    assert sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"
    assert sanitize_filename("") == "unnamed_upload"
    assert sanitize_filename(None) == "unnamed_upload"


def test_parse_valid_csv_preserves_leading_zeros_and_structure():
    csv_bytes = b"id,code,amount\n1,00042,100.50\n2,00999,250.00\n"
    res = parse_and_normalize_stream(io.BytesIO(csv_bytes), filename="data.csv")

    assert res.file_type == "csv"
    assert res.input_row_count == 2
    assert res.output_row_count == 2
    assert res.column_names == ["id", "code", "amount"]
    assert res.normalized_data[0]["code"] == "00042"
    assert res.normalized_data[1]["code"] == "00999"


def test_parse_csv_utf8_bom_handled():
    # UTF-8 with BOM prefix
    bom_csv = b"\xef\xbb\xbfname,dept\nAlice,Engineering\n"
    res = parse_and_normalize_stream(io.BytesIO(bom_csv), filename="bom.csv")
    assert res.column_names == ["name", "dept"]
    assert res.normalized_data[0]["name"] == "Alice"


def test_parse_valid_json_determines_column_order():
    json_bytes = b'[{"z": 1, "a": 2}, {"a": 3, "z": 4}]'
    res = parse_and_normalize_stream(io.BytesIO(json_bytes), filename="test.json")

    assert res.file_type == "json"
    assert res.input_row_count == 2
    assert res.column_names == ["z", "a"]
    assert res.normalized_data[0] == {"z": 1, "a": 2}


def test_parse_json_rejects_nested_structures():
    nested_json = b'[{"id": 1, "profile": {"name": "Nested"}}]'
    with pytest.raises(IngestionError) as excinfo:
        parse_and_normalize_stream(io.BytesIO(nested_json), filename="nested.json")
    assert "Unsupported nested structure" in str(excinfo.value)


def test_parse_rejects_empty_and_oversized():
    # Empty
    with pytest.raises(IngestionError) as excinfo:
        parse_and_normalize_stream(io.BytesIO(b""), filename="empty.csv")
    assert "empty" in str(excinfo.value).lower()

    # Oversized limit test
    large_bytes = b"a" * 105
    with pytest.raises(IngestionError) as excinfo:
        parse_and_normalize_stream(io.BytesIO(large_bytes), filename="big.csv", max_size=100)
    assert "exceeds maximum allowed limit" in str(excinfo.value)
