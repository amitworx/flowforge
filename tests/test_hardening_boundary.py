import io
import json

import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.services import (
    IngestionError,
    detect_file_type,
    parse_and_normalize_stream,
)


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def test_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="testuser", password="password123")


def test_unterminated_quoted_csv_raises_ingestion_error():
    """Unterminated quotes in CSV must be rejected as malformed with IngestionError (400)."""
    # A CSV row where a quote is opened but never closed
    bad_csv = b'name,code\n"Alpha,123\nBeta,456\n'
    with pytest.raises(IngestionError) as excinfo:
        parse_and_normalize_stream(io.BytesIO(bad_csv), filename="unterminated.csv")
    assert "malformed" in str(excinfo.value).lower() or "quote" in str(excinfo.value).lower()


def test_extension_content_mismatch_rejected():
    """CSV content in a file named .json or JSON content in .csv must be rejected."""
    # CSV content with .json extension
    csv_as_json = b"name,code\nAlpha,123\n"
    with pytest.raises(IngestionError) as excinfo:
        detect_file_type("data.json", csv_as_json)
    assert "mismatch" in str(excinfo.value).lower() or "unsupported" in str(excinfo.value).lower()

    # JSON content with .csv extension
    json_as_csv = b'[{"name": "Alpha"}]'
    with pytest.raises(IngestionError) as excinfo:
        detect_file_type("data.csv", json_as_csv)
    assert "mismatch" in str(excinfo.value).lower() or "unsupported" in str(excinfo.value).lower()


def test_csv_exact_1000_rows_accepted_and_1001_rejected():
    """CSV with exactly 1000 rows succeeds; 1001 rows raises IngestionError."""
    # 1000 rows
    lines_1000 = ["id,val"] + [f"{i},{i}" for i in range(1000)]
    csv_1000 = "\n".join(lines_1000).encode("utf-8")
    res_1000 = parse_and_normalize_stream(io.BytesIO(csv_1000), filename="test.csv")
    assert res_1000.output_row_count == 1000

    # 1001 rows
    lines_1001 = ["id,val"] + [f"{i},{i}" for i in range(1001)]
    csv_1001 = "\n".join(lines_1001).encode("utf-8")
    with pytest.raises(IngestionError) as excinfo:
        parse_and_normalize_stream(io.BytesIO(csv_1001), filename="test.csv")
    assert "row count exceeds maximum limit" in str(excinfo.value).lower()


def test_json_exact_1000_rows_accepted_and_1001_rejected():
    """JSON with exactly 1000 items succeeds; 1001 items raises IngestionError."""
    data_1000 = json.dumps([{"id": i} for i in range(1000)]).encode("utf-8")
    res_1000 = parse_and_normalize_stream(io.BytesIO(data_1000), filename="test.json")
    assert res_1000.output_row_count == 1000

    data_1001 = json.dumps([{"id": i} for i in range(1001)]).encode("utf-8")
    with pytest.raises(IngestionError) as excinfo:
        parse_and_normalize_stream(io.BytesIO(data_1001), filename="test.json")
    assert "record count exceeds maximum limit" in str(excinfo.value).lower()


@pytest.mark.django_db
def test_oversized_upload_rejected_at_request_boundary(api_client, test_user):
    """Uploads exceeding MAX_UPLOAD_SIZE_BYTES (1 MiB) are rejected at the view boundary with HTTP 413 or 400."""
    api_client.force_authenticate(user=test_user)

    # 1.1 MiB payload
    oversized_content = b"a" * (1024 * 1024 + 100 * 1024)
    uploaded = io.BytesIO(oversized_content)
    uploaded.name = "huge.csv"

    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded},
        format="multipart",
    )
    assert response.status_code in (
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        status.HTTP_400_BAD_REQUEST,
    )
    assert "exceeds" in response.json().get("detail", "").lower()
