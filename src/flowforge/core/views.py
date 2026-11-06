from datetime import UTC, datetime

from django.db import connection
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class HealthCheckView(APIView):
    """
    Health check endpoint for monitoring service and database status.

    GET /api/v1/health/
    Returns 200 OK when database is reachable, 503 SERVICE UNAVAILABLE otherwise.
    Does NOT check external networks, background workers, or cache layers (M0-001 scope).
    """

    permission_classes: list = []
    authentication_classes: list = []

    def get(self, request: Request) -> Response:  # noqa: ARG002
        checks: dict[str, str] = {}
        is_healthy = True

        # Database connectivity check
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                row = cursor.fetchone()
                if row and row[0] == 1:
                    checks["database"] = "healthy"
                else:
                    checks["database"] = "unhealthy"
                    is_healthy = False
        except Exception:
            checks["database"] = "unhealthy"
            is_healthy = False

        payload = {
            "status": "ok" if is_healthy else "unhealthy",
            "version": "0.1.0",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": checks,
        }

        response_status = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(payload, status=response_status)
