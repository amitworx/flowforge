import json

from django.db import transaction
from django.http import HttpResponse
from rest_framework import exceptions, status
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
    MAX_UPLOAD_SIZE_BYTES,
    IngestionError,
    parse_and_normalize_stream,
)
from flowforge.processing.upload_handlers import BoundedFileUploadHandler


class PayloadTooLarge(exceptions.APIException):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    default_detail = "Request payload exceeds maximum allowed limit of 1 MiB."
    default_code = "payload_too_large"


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
    - Defense-in-depth size bounding:
        Layer 1: Content-Length header validation (rejects > 1 MiB + 64 KiB multipart overhead).
        Layer 2: Uploaded file size attribute validation before stream processing.
        Layer 3: Bounded streaming read inside parse_and_normalize_stream.

    GET /api/v1/jobs/
    - Lists only the requesting user's jobs (paginated).
    """

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def initialize_request(self, request, *args, **kwargs):
        # Scope BoundedFileUploadHandler specifically to the jobs ingestion endpoint,
        # ensuring it precedes MemoryFileUploadHandler and TemporaryFileUploadHandler
        # without altering global handlers for unrelated endpoints (e.g. Django admin).
        existing_handlers = [
            h for h in request.upload_handlers if not isinstance(h, BoundedFileUploadHandler)
        ]
        request.upload_handlers = [
            BoundedFileUploadHandler(request, max_file_size=MAX_UPLOAD_SIZE_BYTES)
        ] + existing_handlers
        return super().initialize_request(request, *args, **kwargs)

    def _check_early_content_length(self, request) -> None:
        """
        Layer 1 defense: Early boundary check against Content-Length header.
        Rejects obviously oversized payloads before parsing commences.
        Isolated into a helper method so tests can suppress or alter it to verify
        streaming handler behavior under missing/unreliable headers without faking handlers.
        """
        content_length = request.META.get("CONTENT_LENGTH")
        if content_length:
            try:
                # Add 64 KiB tolerance for multipart boundaries and headers
                if int(content_length) > (MAX_UPLOAD_SIZE_BYTES + 65536):
                    raise PayloadTooLarge(
                        detail=f"Request payload exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES} bytes (1 MiB)."
                    )
            except (ValueError, TypeError):
                pass

    def initial(self, request, *args, **kwargs):
        self._check_early_content_length(request)
        return super().initial(request, *args, **kwargs)

    def get(self, request: Request) -> Response:
        queryset = ProcessingJob.objects.filter(owner=request.user)
        paginator = JobPagination()
        page = paginator.paginate_queryset(queryset, request, view=self)
        serializer = ProcessingJobListSerializer(page, many=True, context={"request": request})
        return paginator.get_paginated_response(serializer.data)

    def post(self, request: Request) -> Response:
        # Trigger multipart stream parsing by accessing request.FILES
        files = request.FILES

        # Layer 2: Check if any upload handler aborted during streaming chunks
        for handler in getattr(request._request, "upload_handlers", []):
            if isinstance(handler, BoundedFileUploadHandler) and (
                getattr(handler, "upload_aborted", False)
                or getattr(handler, "total_upload_size", 0) > MAX_UPLOAD_SIZE_BYTES
            ):
                raise PayloadTooLarge(
                    detail=f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES} bytes (1 MiB)."
                )

        if "file" not in files:
            return Response(
                {"detail": "A multipart file field named 'file' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        uploaded_file = files["file"]
        filename = uploaded_file.name

        # Layer 2: Reject oversized file objects prior to full stream reading
        if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
            return Response(
                {
                    "detail": f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES} bytes (1 MiB)."
                },
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        try:
            # Layer 3: Parse and normalize content with stream bounding
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
