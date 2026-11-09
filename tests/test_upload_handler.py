import io

import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.processing.upload_handlers import BoundedFileUploadHandler


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def test_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="uploaduser", password="password123")


@pytest.mark.django_db
def test_upload_handler_is_registered_and_invoked_on_request(api_client, test_user, monkeypatch):
    """Verify that BoundedFileUploadHandler is registered in request.upload_handlers and invoked."""
    api_client.force_authenticate(user=test_user)

    handler_invoked = False
    original_receive_data_chunk = BoundedFileUploadHandler.receive_data_chunk

    def mock_receive_data_chunk(self, raw_data: bytes, start: int) -> bytes:
        nonlocal handler_invoked
        handler_invoked = True
        return original_receive_data_chunk(self, raw_data, start)

    monkeypatch.setattr(BoundedFileUploadHandler, "receive_data_chunk", mock_receive_data_chunk)

    csv_content = b"col1,col2\nval1,val2\n"
    uploaded = io.BytesIO(csv_content)
    uploaded.name = "tracked.csv"

    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded},
        format="multipart",
    )
    assert response.status_code == status.HTTP_201_CREATED
    assert handler_invoked is True, (
        "BoundedFileUploadHandler was not invoked during multipart upload parsing"
    )


@pytest.mark.django_db
def test_real_streaming_upload_handler_aborts_without_content_length_guard(
    api_client, test_user, monkeypatch
):
    """
    Test real multipart parsing where Content-Length early rejection is suppressed.
    The file actually reaches BoundedFileUploadHandler.receive_data_chunk,
    detects oversize, raises StopUpload, returns HTTP 413 (not generic 400),
    and creates 0 jobs.
    """
    api_client.force_authenticate(user=test_user)

    initial_job_count = ProcessingJob.objects.count()

    # Track receive_data_chunk calls to prove real streaming execution
    chunk_calls = []
    original_chunk = BoundedFileUploadHandler.receive_data_chunk

    def tracked_chunk(self, raw_data: bytes, start: int) -> bytes:
        chunk_calls.append((len(raw_data), start))
        return original_chunk(self, raw_data, start)

    monkeypatch.setattr(BoundedFileUploadHandler, "receive_data_chunk", tracked_chunk)

    # Controlled test technique: suppress the early header check without faking handler behavior
    from flowforge.processing.views import ProcessingJobListCreateView

    monkeypatch.setattr(
        ProcessingJobListCreateView,
        "_check_early_content_length",
        lambda self, request: None,  # noqa: ARG005
    )

    # 1.2 MiB oversized file
    oversized_bytes = b"header1,header2\n" + (b"1234567,abcdefg\n" * 80000)
    assert len(oversized_bytes) > 1024 * 1024

    uploaded = io.BytesIO(oversized_bytes)
    uploaded.name = "real_stream.csv"

    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded},
        format="multipart",
    )

    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "exceeds" in response.json().get("detail", "").lower()
    # Verify that receive_data_chunk was called multiple times until StopUpload was raised
    assert len(chunk_calls) > 1, f"Expected multiple chunk invocations, got {len(chunk_calls)}"
    assert ProcessingJob.objects.count() == initial_job_count


@pytest.mark.django_db
def test_streaming_upload_handler_aborts_oversized_before_job_creation(api_client, test_user):
    """
    Simulate a streaming upload exceeding 1 MiB during chunk processing:
    The handler raises StopUpload, the view returns 413, and NO ProcessingJob is created in DB.
    """
    api_client.force_authenticate(user=test_user)

    initial_job_count = ProcessingJob.objects.count()

    # 1.2 MiB upload
    oversized_bytes = b"header1,header2\n" + (b"1234567,abcdefg\n" * 80000)
    assert len(oversized_bytes) > 1024 * 1024

    uploaded = io.BytesIO(oversized_bytes)
    uploaded.name = "stream_large.csv"

    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded},
        format="multipart",
    )
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "exceeds" in response.json().get("detail", "").lower()

    # Ensure no job record was persisted
    assert ProcessingJob.objects.count() == initial_job_count


@pytest.mark.django_db
def test_early_rejection_on_content_length_header(api_client, test_user):
    """Requests presenting an oversized Content-Length header are rejected early without reading."""
    api_client.force_authenticate(user=test_user)

    initial_job_count = ProcessingJob.objects.count()
    uploaded = io.BytesIO(b"small,content\n1,2\n")
    uploaded.name = "small.csv"

    # Pass an explicitly oversized Content-Length header
    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded},
        format="multipart",
        CONTENT_LENGTH=str(5 * 1024 * 1024),  # 5 MiB
    )
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "exceeds" in response.json().get("detail", "").lower()
    assert ProcessingJob.objects.count() == initial_job_count


@pytest.mark.django_db
def test_exact_1_mib_valid_csv_accepted(api_client, test_user):
    """Verify that a valid CSV file of exactly 1 MiB (1,048,576 bytes) within row limits is accepted."""
    from flowforge.processing.services import MAX_UPLOAD_SIZE_BYTES

    api_client.force_authenticate(user=test_user)

    # 10 bytes header + 999 data rows = 1,000 lines total (satisfies record cap)
    # Total byte count exactly 1,048,576 bytes
    header = b"col1,col2\n"  # 10 bytes
    rows = []
    for _ in range(615):
        rows.append(b"r," + (b"x" * 1047) + b"\n")  # 1050 bytes each (615 * 1050 = 645,750)
    for _ in range(384):
        rows.append(b"r," + (b"x" * 1046) + b"\n")  # 1049 bytes each (384 * 1049 = 402,816)
    csv_exact = header + b"".join(rows)
    assert len(csv_exact) == MAX_UPLOAD_SIZE_BYTES
    assert csv_exact.count(b"\n") == 1000

    uploaded = io.BytesIO(csv_exact)
    uploaded.name = "exact_1mib.csv"

    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded},
        format="multipart",
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["state"] == ProcessingJob.State.COMPLETED
    assert data["input_row_count"] == 999
    assert data["output_row_count"] == 999


@pytest.mark.django_db
def test_unrelated_endpoints_preserve_default_upload_handlers(rf):
    """Ensure that standard Django request handling (e.g. admin or other views) retains default handlers."""
    req = rf.get("/admin/")
    handler_classes = [type(h).__name__ for h in req.upload_handlers]
    assert "BoundedFileUploadHandler" not in handler_classes
    assert "MemoryFileUploadHandler" in handler_classes
    assert "TemporaryFileUploadHandler" in handler_classes
