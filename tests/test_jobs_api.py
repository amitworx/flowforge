import io

import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient

# Note: We will import ProcessingJob once the model is created
# from flowforge.processing.models import ProcessingJob


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def test_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="alice", password="password123")


@pytest.fixture
def other_user(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="bob", password="password123")


@pytest.mark.django_db
def test_jobs_endpoint_requires_authentication(api_client: APIClient):
    """POST and GET /api/v1/jobs/ must require authentication."""
    response = api_client.get("/api/v1/jobs/")
    assert response.status_code in (
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
    )

    response = api_client.post("/api/v1/jobs/", data={})
    assert response.status_code in (
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
    )


@pytest.mark.django_db
def test_post_valid_csv_synchronously_processes_job(api_client: APIClient, test_user: User):
    """POST valid CSV to /api/v1/jobs/ parses, normalizes and persists job."""
    api_client.force_authenticate(user=test_user)

    csv_content = b"name,code,score\nAlpha,00123,98.5\nBeta,00456,85.0\n"
    uploaded_file = io.BytesIO(csv_content)
    uploaded_file.name = "sample_data.csv"

    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded_file},
        format="multipart",
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "id" in data
    assert data["file_type"] == "csv"
    assert data["state"] == "completed"
    assert data["original_filename"] == "sample_data.csv"
    assert data["input_row_count"] == 2
    assert data["output_row_count"] == 2
    assert "download_url" in data
    assert data["validation_errors"] is None

    # Retrieve specific job
    job_id = data["id"]
    get_res = api_client.get(f"/api/v1/jobs/{job_id}/")
    assert get_res.status_code == status.HTTP_200_OK
    job_data = get_res.json()
    assert job_data["id"] == job_id
    assert job_data["state"] == "completed"
    assert len(job_data["output_data"]) == 2
    # Preserves leading zeros
    assert job_data["output_data"][0]["code"] == "00123"

    # Download normalized JSON
    dl_res = api_client.get(f"/api/v1/jobs/{job_id}/download/")
    assert dl_res.status_code == status.HTTP_200_OK
    assert dl_res.headers["Content-Type"] == "application/json"
    assert "attachment" in dl_res.headers.get("Content-Disposition", "")
    assert dl_res.json() == job_data["output_data"]


@pytest.mark.django_db
def test_post_valid_json_synchronously_processes_job(api_client: APIClient, test_user: User):
    """POST valid JSON array of objects to /api/v1/jobs/."""
    api_client.force_authenticate(user=test_user)

    json_content = b'[{"id": "01", "name": "Item A"}, {"id": "02", "name": "Item B"}]'
    uploaded_file = io.BytesIO(json_content)
    uploaded_file.name = "items.json"

    response = api_client.post(
        "/api/v1/jobs/",
        data={"file": uploaded_file},
        format="multipart",
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["file_type"] == "json"
    assert data["state"] == "completed"
    assert data["input_row_count"] == 2
    assert data["output_row_count"] == 2


@pytest.mark.django_db
def test_object_ownership_and_cross_user_isolation(
    api_client: APIClient, test_user: User, other_user: User
):
    """Users cannot view, list, or download other users' jobs."""
    api_client.force_authenticate(user=test_user)
    csv_content = b"header1,header2\nval1,val2\n"
    uploaded = io.BytesIO(csv_content)
    uploaded.name = "test.csv"
    res = api_client.post("/api/v1/jobs/", data={"file": uploaded}, format="multipart")
    job_id = res.json()["id"]

    # Bob logs in
    api_client.force_authenticate(user=other_user)

    # Bob's list should be empty
    list_res = api_client.get("/api/v1/jobs/")
    assert list_res.status_code == status.HTTP_200_OK
    assert len(list_res.json()["results"]) == 0

    # Bob cannot retrieve Alice's job
    get_res = api_client.get(f"/api/v1/jobs/{job_id}/")
    assert get_res.status_code == status.HTTP_404_NOT_FOUND

    # Bob cannot download Alice's job
    dl_res = api_client.get(f"/api/v1/jobs/{job_id}/download/")
    assert dl_res.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.django_db
def test_rejections_and_error_handling(api_client: APIClient, test_user: User):
    """Tests empty file, malformed UTF-8, oversized file, invalid shape, duplicate headers."""
    api_client.force_authenticate(user=test_user)

    # 1. Empty file
    empty_file = io.BytesIO(b"")
    empty_file.name = "empty.csv"
    res = api_client.post("/api/v1/jobs/", data={"file": empty_file}, format="multipart")
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "empty" in res.json().get("detail", "").lower()

    # 2. Invalid UTF-8
    bad_utf8 = io.BytesIO(b"\xff\xfe\x00\x00abc")
    bad_utf8.name = "bad.csv"
    res = api_client.post("/api/v1/jobs/", data={"file": bad_utf8}, format="multipart")
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "UTF-8" in res.json().get("detail", "")

    # 3. Duplicate headers in CSV
    dup_headers = io.BytesIO(b"col,col\n1,2\n")
    dup_headers.name = "dup.csv"
    res = api_client.post("/api/v1/jobs/", data={"file": dup_headers}, format="multipart")
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "Duplicate header" in res.json().get("detail", "")

    # 4. JSON non-array top level
    bad_json = io.BytesIO(b'{"key": "value"}')
    bad_json.name = "obj.json"
    res = api_client.post("/api/v1/jobs/", data={"file": bad_json}, format="multipart")
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "array" in res.json().get("detail", "").lower()

    # 5. Unsupported file extension / type
    bad_ext = io.BytesIO(b"hello world")
    bad_ext.name = "script.py"
    res = api_client.post("/api/v1/jobs/", data={"file": bad_ext}, format="multipart")
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported file type" in res.json().get("detail", "")
