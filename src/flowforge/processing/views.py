import json

from django.db import transaction
from django.http import HttpResponse
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from flowforge.processing.models import ProcessingJob
from flowforge.processing.serializers import (
    ProcessingJobDetailSerializer,
    ProcessingJobListSerializer,
)
from flowforge.processing.services import (
    IngestionError,
    parse_and_normalize_stream,
)


class JobPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class ProcessingJobListCreateView(APIView):
    """
    API endpoint to list jobs owned by the requesting user or create/process a new job.

    POST /api/v1/jobs/
    - Accepts multipart/form-data upload with a 'file' parameter.
    - Synchronously validates, parses, normalizes and persists job record.

    GET /api/v1/jobs/
    - Lists only the requesting user's jobs (paginated).
    """

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def get(self, request: Request) -> Response:
        queryset = ProcessingJob.objects.filter(owner=request.user)
        paginator = JobPagination()
        page = paginator.paginate_queryset(queryset, request, view=self)
        serializer = ProcessingJobListSerializer(page, many=True, context={"request": request})
        return paginator.get_paginated_response(serializer.data)

    def post(self, request: Request) -> Response:
        if "file" not in request.FILES:
            return Response(
                {"detail": "A multipart file field named 'file' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        uploaded_file = request.FILES["file"]
        filename = uploaded_file.name

        try:
            # Parse and normalize content
            parse_result = parse_and_normalize_stream(uploaded_file.file, filename=filename)
        except IngestionError as exc:
            return Response(
                {"detail": exc.message},
                status=exc.status_code,
            )

        # Persist completed job
        with transaction.atomic():
            job = ProcessingJob.objects.create(
                owner=request.user,
                file_type=parse_result.file_type,
                state=ProcessingJob.State.COMPLETED,
                original_filename=parse_result.original_filename,
                input_row_count=parse_result.input_row_count,
                output_row_count=parse_result.output_row_count,
                column_names=parse_result.column_names,
                output_data=parse_result.normalized_data,
            )

        serializer = ProcessingJobListSerializer(job, context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ProcessingJobDetailView(APIView):
    """
    Retrieve metadata and bounded normalized results for a user-owned job.

    GET /api/v1/jobs/{uuid}/
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, job_id: str, user) -> ProcessingJob:
        try:
            return ProcessingJob.objects.get(id=job_id, owner=user)
        except (ProcessingJob.DoesNotExist, ValueError) as exc:
            raise NotFound(detail="Processing job not found.") from exc

    def get(self, request: Request, job_id: str) -> Response:
        job = self.get_object(job_id, request.user)
        serializer = ProcessingJobDetailSerializer(job, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class ProcessingJobDownloadView(APIView):
    """
    Download normalized JSON output with attachment headers. Accessible only to job owner.

    GET /api/v1/jobs/{uuid}/download/
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, job_id: str) -> HttpResponse:
        try:
            job = ProcessingJob.objects.get(id=job_id, owner=request.user)
        except (ProcessingJob.DoesNotExist, ValueError) as exc:
            raise NotFound(detail="Processing job not found.") from exc

        json_bytes = json.dumps(job.output_data, indent=2).encode("utf-8")
        safe_name = f"flowforge_job_{job.id}.json"

        response = HttpResponse(json_bytes, content_type="application/json")
        response["Content-Disposition"] = f'attachment; filename="{safe_name}"'
        response["Content-Length"] = len(json_bytes)
        return response
