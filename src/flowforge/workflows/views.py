import logging

from django.db import transaction
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from flowforge.processing.models import ProcessingJob
from flowforge.processing.views import JobPagination
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
from flowforge.workflows.serializers import (
    WorkflowAsyncRunResponseSerializer,
    WorkflowDefinitionSerializer,
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
        queryset = WorkflowDefinition.objects.filter(owner=request.user)
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
            run = execute_workflow_run(workflow_def, source_job, request.user)
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

        with transaction.atomic():
            run = WorkflowRun.objects.create(
                owner=request.user,
                workflow_definition=workflow_def,
                source_job=source_job,
                state=WorkflowRun.State.QUEUED,
                dispatch_status=WorkflowRun.DispatchStatus.DISPATCHED,
                step_snapshot=list(workflow_def.steps),
                input_row_count=len(source_job.output_data or []),
                output_row_count=0,
            )

            def _dispatch_task(run_id: str):
                try:
                    execute_workflow_run_task.delay(run_id, None)
                except Exception:
                    logger.exception(
                        "Failed to publish task to Celery broker for WorkflowRun %s", run_id
                    )
                    WorkflowRun.objects.filter(id=run_id).update(
                        dispatch_status=WorkflowRun.DispatchStatus.FAILED,
                        error_summary="Broker unavailable during async dispatch. Run is recoverable.",
                    )

            transaction.on_commit(lambda: _dispatch_task(str(run.id)))

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
