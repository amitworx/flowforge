import pytest
from rest_framework import status
from rest_framework.test import APIClient


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.mark.django_db
def test_health_check_endpoint_returns_200_and_expected_structure(api_client: APIClient):
    """GET /api/v1/health/ should return status 200 and standard health payload."""
    response = api_client.get("/api/v1/health/")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data
    assert "checks" in data
    assert data["checks"]["database"] == "healthy"


@pytest.mark.django_db
def test_health_check_database_failure(api_client: APIClient, monkeypatch):
    """GET /api/v1/health/ should report unhealthy and status 503 if database check fails."""
    from django.db import connection

    def mock_cursor():
        raise Exception("Database unreachable")

    monkeypatch.setattr(connection, "cursor", mock_cursor)

    response = api_client.get("/api/v1/health/")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["checks"]["database"] == "unhealthy"
