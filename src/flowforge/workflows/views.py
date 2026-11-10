import logging

from django.db import transaction
from django.http import HttpResponse
from django.utils import timezone
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.negotiation import BaseContentNegotiation
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from flowforge.processing.models import ProcessingJob
from flowforge.processing.views import JobPagination
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_workflow_pipeline_with_schema,
)
from flowforge.workflows.export import (
    DEFAULT_MAX_EXPORT_SIZE_BYTES,
    ExportPayloadTooLargeError,
    export_workflow_run_to_csv,
    export_workflow_run_to_json,
    serialize_to_bounded_json,
)
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
from flowforge.workflows.serializers import (
    WorkflowAsyncRunResponseSerializer,
    WorkflowDefinitionSerializer,
    WorkflowDefinitionUpdateSerializer,
    WorkflowPreviewRequestSerializer,
    WorkflowPreviewResponseSerializer,
    WorkflowRunCreateSerializer,
    WorkflowRunDetailSerializer,
    WorkflowRunListSerializer,
)
from flowforge.workflows.services import WorkflowServiceError, execute_workflow_run
from flowforge.workflows.tasks import execute_workflow_run_task

logger = logging.getLogger(__name__)


class WorkflowDefinitionListCreateView(APIView):
    """
    POST /api/v1/workflows/
    GET /api/v1/workflows/
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        include_archived_param = (
            request.query_params.get("include_archived", "false").lower().strip()
        )
        if include_archived_param in ("true", "1"):
            include_archived = True
        elif include_archived_param in ("false", "0"):
            include_archived = False
        else:
            return Response(
                {"detail": "Invalid 'include_archived' parameter. Must be 'true' or 'false'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        queryset = WorkflowDefinition.objects.filter(owner=request.user)
        if not include_archived:
            queryset = queryset.filter(is_archived=False)

        paginator = JobPagination()
        page = paginator.paginate_queryset(queryset, request, view=self)
        serializer = WorkflowDefinitionSerializer(page, many=True, context={"request": request})
        return paginator.get_paginated_response(serializer.data)

    def post(self, request: Request) -> Response:
        serializer = WorkflowDefinitionSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        workflow_def = serializer.save(owner=request.user)
        return Response(
            WorkflowDefinitionSerializer(workflow_def).data, status=status.HTTP_201_CREATED
        )


class WorkflowDefinitionDetailView(APIView):
    """
    GET /api/v1/workflows/{uuid}/
    PATCH /api/v1/workflows/{uuid}/
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, uuid: str, user) -> WorkflowDefinition:
        try:
            return WorkflowDefinition.objects.get(id=uuid, owner=user)
        except (WorkflowDefinition.DoesNotExist, ValueError) as exc:
            raise NotFound("Workflow definition not found.") from exc

    def get(self, request: Request, uuid: str) -> Response:
        workflow_def = self.get_object(uuid, request.user)
        return Response(WorkflowDefinitionSerializer(workflow_def).data, status=status.HTTP_200_OK)

    def patch(self, request: Request, uuid: str) -> Response:
        workflow_def = self.get_object(uuid, request.user)

        if workflow_def.is_archived:
            return Response(
                {"detail": "Cannot modify an archived workflow definition. Restore it first."},
                status=status.HTTP_409_CONFLICT,
            )

        # Validate inputs outside the lock to avoid prolonged row contention
        serializer = WorkflowDefinitionUpdateSerializer(
            workflow_def,
            data=request.data,
            partial=True,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)

        with transaction.atomic():
            locked_wf = WorkflowDefinition.objects.select_for_update().get(
                id=workflow_def.id, owner=request.user
            )
            if locked_wf.is_archived:
                return Response(
                    {"detail": "Cannot modify an archived workflow definition. Restore it first."},
                    status=status.HTTP_409_CONFLICT,
                )

            # Apply validated fields inside atomic transaction
            for attr, value in serializer.validated_data.items():
                setattr(locked_wf, attr, value)
            locked_wf.save(update_fields=list(serializer.validated_data.keys()) + ["updated_at"])

        return Response(WorkflowDefinitionSerializer(locked_wf).data, status=status.HTTP_200_OK)


class WorkflowArchiveView(APIView):
    """
    POST /api/v1/workflows/{uuid}/archive/
    Safely and idempotently archives a WorkflowDefinition.
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, uuid: str, user) -> WorkflowDefinition:
        try:
            return WorkflowDefinition.objects.get(id=uuid, owner=user)
        except (WorkflowDefinition.DoesNotExist, ValueError) as exc:
            raise NotFound("Workflow definition not found.") from exc

    def post(self, request: Request, uuid: str) -> Response:
        self.get_object(uuid, request.user)

        with transaction.atomic():
            workflow_def = WorkflowDefinition.objects.select_for_update().get(
                id=uuid, owner=request.user
            )
            if not workflow_def.is_archived:
                workflow_def.is_archived = True
                workflow_def.archived_at = timezone.now()
                workflow_def.save(update_fields=["is_archived", "archived_at", "updated_at"])

        return Response(WorkflowDefinitionSerializer(workflow_def).data, status=status.HTTP_200_OK)


class WorkflowRestoreView(APIView):
    """
    POST /api/v1/workflows/{uuid}/restore/
    Safely and idempotently restores an archived WorkflowDefinition.
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, uuid: str, user) -> WorkflowDefinition:
        try:
            return WorkflowDefinition.objects.get(id=uuid, owner=user)
        except (WorkflowDefinition.DoesNotExist, ValueError) as exc:
            raise NotFound("Workflow definition not found.") from exc

    def post(self, request: Request, uuid: str) -> Response:
        self.get_object(uuid, request.user)

        with transaction.atomic():
            workflow_def = WorkflowDefinition.objects.select_for_update().get(
                id=uuid, owner=request.user
            )
            if workflow_def.is_archived:
                workflow_def.is_archived = False
                workflow_def.archived_at = None
                workflow_def.save(update_fields=["is_archived", "archived_at", "updated_at"])

        return Response(WorkflowDefinitionSerializer(workflow_def).data, status=status.HTTP_200_OK)


class WorkflowExecuteRunView(APIView):
    """
    POST /api/v1/workflows/{uuid}/runs/
    """

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, uuid: str) -> Response:
        try:
            workflow_def = WorkflowDefinition.objects.get(id=uuid, owner=request.user)
        except (WorkflowDefinition.DoesNotExist, ValueError):
            return Response(
                {"detail": "Workflow definition not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if workflow_def.is_archived:
            return Response(
                {"detail": "Cannot execute run against archived workflow definition."},
                status=status.HTTP_409_CONFLICT,
            )

        serializer = WorkflowRunCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        source_job_id = serializer.validated_data["source_job_id"]

        try:
            source_job = ProcessingJob.objects.get(id=source_job_id, owner=request.user)
        except (ProcessingJob.DoesNotExist, ValueError):
            return Response(
                {"detail": "Source processing job not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            with transaction.atomic():
                locked_wf = WorkflowDefinition.objects.select_for_update().get(
                    id=workflow_def.id, owner=request.user
                )
                if locked_wf.is_archived:
                    return Response(
                        {"detail": "Cannot execute run against archived workflow definition."},
                        status=status.HTTP_409_CONFLICT,
                    )
                run = execute_workflow_run(locked_wf, source_job, request.user)
        except WorkflowServiceError as exc:
            return Response({"detail": exc.message}, status=exc.status_code)

        run_serializer = WorkflowRunListSerializer(run)
        return Response(run_serializer.data, status=status.HTTP_201_CREATED)


class WorkflowExecuteAsyncRunView(APIView):
    """
    POST /api/v1/workflows/{uuid}/runs/async/
    Enqueues a background asynchronous run of a WorkflowDefinition against a ProcessingJob.
    Returns HTTP 202 Accepted with state 'queued' and status URL if dispatched,
    or HTTP 503 Service Unavailable if message broker publication fails, preserving
    a recoverable persisted WorkflowRun state.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, uuid: str) -> Response:
        try:
            workflow_def = WorkflowDefinition.objects.get(id=uuid, owner=request.user)
        except (WorkflowDefinition.DoesNotExist, ValueError):
            return Response(
                {"detail": "Workflow definition not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if workflow_def.is_archived:
            return Response(
                {"detail": "Cannot execute run against archived workflow definition."},
                status=status.HTTP_409_CONFLICT,
            )

        serializer = WorkflowRunCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        source_job_id = serializer.validated_data["source_job_id"]

        try:
            source_job = ProcessingJob.objects.get(id=source_job_id, owner=request.user)
        except (ProcessingJob.DoesNotExist, ValueError):
            return Response(
                {"detail": "Source processing job not found."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if source_job.state != ProcessingJob.State.COMPLETED:
            return Response(
                {
                    "detail": f"Cannot execute workflow against source job in '{source_job.state}' state. Must be 'completed'."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        dispatch_error = None

        with transaction.atomic():
            locked_wf = WorkflowDefinition.objects.select_for_update().get(
                id=workflow_def.id, owner=request.user
            )
            if locked_wf.is_archived:
                return Response(
                    {"detail": "Cannot execute run against archived workflow definition."},
                    status=status.HTTP_409_CONFLICT,
                )
            run = WorkflowRun.objects.create(
                owner=request.user,
                workflow_definition=locked_wf,
                source_job=source_job,
                state=WorkflowRun.State.QUEUED,
                dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
                step_snapshot=list(locked_wf.steps),
                input_row_count=len(source_job.output_data or []),
                output_row_count=0,
            )

            def _publish():
                nonlocal dispatch_error
                try:
                    execute_workflow_run_task.delay(str(run.id), None)
                except Exception as exc:
                    dispatch_error = exc
                    logger.exception(
                        "Failed to publish task to Celery broker for WorkflowRun %s", run.id
                    )
                    WorkflowRun.objects.filter(id=run.id).update(
                        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
                        error_summary="Broker unavailable during async dispatch. Run is recoverable.",
                    )

            transaction.on_commit(_publish)

        if dispatch_error is not None:
            run.refresh_from_db()
            response_serializer = WorkflowAsyncRunResponseSerializer(run)
            return Response(
                {
                    **response_serializer.data,
                    "detail": "Broker unavailable; async dispatch deferred. Run recorded for recovery.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        response_serializer = WorkflowAsyncRunResponseSerializer(run)
        return Response(response_serializer.data, status=status.HTTP_202_ACCEPTED)


class WorkflowRunListView(APIView):
    """
    GET /api/v1/workflow-runs/
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        queryset = WorkflowRun.objects.filter(owner=request.user)
        paginator = JobPagination()
        page = paginator.paginate_queryset(queryset, request, view=self)
        serializer = WorkflowRunListSerializer(page, many=True, context={"request": request})
        return paginator.get_paginated_response(serializer.data)


class WorkflowRunDetailView(APIView):
    """
    GET /api/v1/workflow-runs/{uuid}/
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, uuid: str, user) -> WorkflowRun:
        try:
            return WorkflowRun.objects.get(id=uuid, owner=user)
        except (WorkflowRun.DoesNotExist, ValueError) as exc:
            raise NotFound("Workflow run not found.") from exc

    def get(self, request: Request, uuid: str) -> Response:
        run = self.get_object(uuid, request.user)
        return Response(WorkflowRunDetailSerializer(run).data, status=status.HTTP_200_OK)


class IgnoreFormatContentNegotiation(BaseContentNegotiation):
    def select_renderer(self, request, renderers, format_suffix=None):  # noqa: ARG002
        return renderers[0], renderers[0].media_type


class WorkflowRunDownloadView(APIView):
    """
    GET /api/v1/workflow-runs/{uuid}/download/?format=json|csv
    Export and download the outputs of an already completed WorkflowRun.
    """

    permission_classes = [IsAuthenticated]
    content_negotiation_class = IgnoreFormatContentNegotiation

    def get_object(self, uuid: str, user) -> WorkflowRun:
        try:
            return WorkflowRun.objects.get(id=uuid, owner=user)
        except (WorkflowRun.DoesNotExist, ValueError) as exc:
            raise NotFound("Workflow run not found.") from exc

    def get(self, request: Request, uuid: str) -> HttpResponse | Response:
        run = self.get_object(uuid, request.user)

        if run.state != WorkflowRun.State.COMPLETED:
            return Response(
                {
                    "detail": f"Cannot export workflow run in '{run.state}' state. Run must be 'completed'."
                },
                status=status.HTTP_409_CONFLICT,
            )

        export_format = request.query_params.get("format", "").lower().strip()
        if export_format not in ("json", "csv"):
            return Response(
                {"detail": "Invalid export format. Must specify ?format=json or ?format=csv."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        output_data = run.output_data or []
        output_columns = run.get_output_schema()

        try:
            if export_format == "json":
                content = export_workflow_run_to_json(
                    output_data, max_bytes=DEFAULT_MAX_EXPORT_SIZE_BYTES
                )
                response = HttpResponse(
                    content,
                    status=status.HTTP_200_OK,
                    content_type="application/json; charset=utf-8",
                )
                response["Content-Disposition"] = (
                    f'attachment; filename="workflow-run-{run.id}.json"'
                )
                return response

            elif export_format == "csv":
                content = export_workflow_run_to_csv(
                    output_data, output_columns, max_bytes=DEFAULT_MAX_EXPORT_SIZE_BYTES
                )
                response = HttpResponse(
                    content,
                    status=status.HTTP_200_OK,
                    content_type="text/csv; charset=utf-8",
                )
                response["Content-Disposition"] = (
                    f'attachment; filename="workflow-run-{run.id}.csv"'
                )
                return response
        except ExportPayloadTooLargeError as exc:
            return Response(
                {"detail": exc.message},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )


class WorkflowPreviewView(APIView):
    """
    POST /api/v1/workflows/{uuid}/preview/
    Read-only pre-run workflow preview endpoint.
    Validates a workflow pipeline against a completed processing job's dataset
    without creating persisted runs or mutating any state.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, uuid: str) -> Response:
        try:
            workflow_def = WorkflowDefinition.objects.get(id=uuid, owner=request.user)
        except (WorkflowDefinition.DoesNotExist, ValueError):
            return Response(
                {"detail": "Workflow definition not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if workflow_def.is_archived:
            return Response(
                {"detail": "Cannot preview an archived workflow definition. Restore it first."},
                status=status.HTTP_409_CONFLICT,
            )

        serializer = WorkflowPreviewRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        source_job_id = serializer.validated_data["source_job_id"]
        preview_limit = serializer.validated_data.get("preview_limit", 20)

        try:
            source_job = ProcessingJob.objects.get(id=source_job_id, owner=request.user)
        except (ProcessingJob.DoesNotExist, ValueError):
            return Response(
                {"detail": "Source processing job not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if source_job.state != ProcessingJob.State.COMPLETED:
            return Response(
                {
                    "detail": f"Cannot preview workflow against source job in '{source_job.state}' state. Must be 'completed'."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Snapshot workflow definition and source job state under a short atomic transaction + row lock
        with transaction.atomic():
            locked_wf = WorkflowDefinition.objects.select_for_update().get(
                id=workflow_def.id, owner=request.user
            )
            if locked_wf.is_archived:
                return Response(
                    {"detail": "Cannot preview an archived workflow definition. Restore it first."},
                    status=status.HTTP_409_CONFLICT,
                )
            steps_snapshot = list(locked_wf.steps)
            locked_job = ProcessingJob.objects.get(id=source_job.id, owner=request.user)
            input_data_copy = [dict(row) for row in (locked_job.output_data or [])]
            initial_schema = (
                list(locked_job.column_names)
                if locked_job.column_names
                else (list(input_data_copy[0].keys()) if input_data_copy else [])
            )

        # Execute pure deterministic pipeline in memory outside of database transaction
        input_row_count = len(input_data_copy)
        try:
            output_data, step_audits, output_columns = execute_workflow_pipeline_with_schema(
                input_data_copy, steps_snapshot, schema=initial_schema
            )
        except WorkflowExecutionError as exc:
            return Response(
                {
                    "detail": exc.message,
                    "step_index": exc.step_index,
                    "completed_step_audits": exc.completed_step_audit,
                    "failed_step_audit": exc.failed_step_audit,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        output_row_count = len(output_data)
        preview_rows = output_data[:preview_limit]

        response_payload = {
            "output_columns": output_columns,
            "input_row_count": input_row_count,
            "output_row_count": output_row_count,
            "preview_limit": preview_limit,
            "preview_rows": preview_rows,
            "step_audits": step_audits,
        }

        # Validate content negotiation to reject unsupported Accept headers (e.g. text/html)
        self.perform_content_negotiation(request)

        # Validate against response serializer
        resp_serializer = WorkflowPreviewResponseSerializer(data=response_payload)
        resp_serializer.is_valid(raise_exception=True)

        # Check payload size limit incrementally and obtain verified UTF-8 JSON bytes
        try:
            serialized_bytes = serialize_to_bounded_json(
                resp_serializer.data,
                max_bytes=DEFAULT_MAX_EXPORT_SIZE_BYTES,
                indent=2,
            )
        except ExportPayloadTooLargeError:
            return Response(
                {
                    "detail": f"Preview response size exceeds maximum allowed size of {DEFAULT_MAX_EXPORT_SIZE_BYTES} bytes."
                },
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        return HttpResponse(
            serialized_bytes,
            status=status.HTTP_200_OK,
            content_type="application/json; charset=utf-8",
        )
