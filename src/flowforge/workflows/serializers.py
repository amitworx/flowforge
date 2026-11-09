from rest_framework import serializers

from flowforge.workflows.engine import WorkflowExecutionError, validate_step_definitions
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStepRun


class WorkflowDefinitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowDefinition
        fields = [
            "id",
            "name",
            "steps",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def validate_steps(self, value):
        try:
            return validate_step_definitions(value)
        except WorkflowExecutionError as exc:
            raise serializers.ValidationError(exc.message) from exc


class WorkflowStepRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowStepRun
        fields = [
            "id",
            "step_index",
            "operation",
            "status",
            "input_row_count",
            "output_row_count",
            "error_summary",
            "created_at",
        ]


class WorkflowRunListSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowRun
        fields = [
            "id",
            "workflow_definition",
            "source_job",
            "state",
            "dispatch_status",
            "input_row_count",
            "output_row_count",
            "error_summary",
            "created_at",
            "started_at",
            "finished_at",
        ]


class WorkflowRunDetailSerializer(serializers.ModelSerializer):
    step_runs = WorkflowStepRunSerializer(many=True, read_only=True)

    class Meta:
        model = WorkflowRun
        fields = [
            "id",
            "workflow_definition",
            "source_job",
            "state",
            "dispatch_status",
            "step_snapshot",
            "output_data",
            "input_row_count",
            "output_row_count",
            "error_summary",
            "created_at",
            "started_at",
            "finished_at",
            "step_runs",
        ]


class WorkflowRunCreateSerializer(serializers.Serializer):
    source_job_id = serializers.UUIDField(required=True)


class WorkflowAsyncRunResponseSerializer(serializers.ModelSerializer):
    status_url = serializers.SerializerMethodField()

    class Meta:
        model = WorkflowRun
        fields = [
            "id",
            "workflow_definition",
            "source_job",
            "state",
            "dispatch_status",
            "status_url",
            "created_at",
        ]

    def get_status_url(self, obj: WorkflowRun) -> str:
        return f"/api/v1/workflow-runs/{obj.id}/"
