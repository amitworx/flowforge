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
