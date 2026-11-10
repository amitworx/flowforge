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
            "is_archived",
            "archived_at",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "is_archived", "archived_at", "created_at", "updated_at"]

    def validate_name(self, value):
        stripped = value.strip()
        if not stripped:
            raise serializers.ValidationError("Workflow name cannot be empty.")
        if len(stripped) > 128:
            raise serializers.ValidationError("Workflow name cannot exceed 128 characters.")
        return stripped

    def validate_steps(self, value):
        try:
            return validate_step_definitions(value)
        except WorkflowExecutionError as exc:
            raise serializers.ValidationError(exc.message) from exc


class WorkflowDefinitionUpdateSerializer(serializers.ModelSerializer):
    """
    Serializer specifically for PATCH requests on WorkflowDefinition.
    Enforces strict field boundaries: only 'name' and/or 'steps' can be supplied.
    Rejects read-only, immutable, or unknown fields with 400 Bad Request.
    """

    class Meta:
        model = WorkflowDefinition
        fields = ["name", "steps"]

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError(
                {"detail": "Invalid payload format. Expected a JSON object."}
            )

        if not data:
            raise serializers.ValidationError(
                {"detail": "At least one field ('name' or 'steps') must be provided."}
            )

        allowed_fields = {"name", "steps"}
        input_keys = set(data.keys())
        unknown_or_disallowed = input_keys - allowed_fields
        if unknown_or_disallowed:
            sorted_disallowed = sorted(unknown_or_disallowed)
            raise serializers.ValidationError(
                dict.fromkeys(
                    sorted_disallowed,
                    "Field cannot be updated. Only 'name' and 'steps' are allowed.",
                )
            )

        return super().to_internal_value(data)

    def validate_name(self, value):
        stripped = value.strip()
        if not stripped:
            raise serializers.ValidationError("Workflow name cannot be empty.")
        if len(stripped) > 128:
            raise serializers.ValidationError("Workflow name cannot exceed 128 characters.")
        return stripped

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


class WorkflowPreviewRequestSerializer(serializers.Serializer):
    source_job_id = serializers.UUIDField(required=True)
    preview_limit = serializers.IntegerField(required=False, default=20, min_value=1, max_value=50)

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError(
                {"detail": "Invalid payload format. Expected a JSON object."}
            )
        allowed_fields = {"source_job_id", "preview_limit"}
        disallowed = set(data.keys()) - allowed_fields
        if disallowed:
            raise serializers.ValidationError(
                dict.fromkeys(
                    sorted(disallowed),
                    f"Unexpected field. Only {sorted(allowed_fields)} are allowed.",
                )
            )
        return super().to_internal_value(data)


class WorkflowPreviewStepAuditSerializer(serializers.Serializer):
    step_index = serializers.IntegerField()
    operation = serializers.CharField()
    status = serializers.CharField()
    input_row_count = serializers.IntegerField()
    output_row_count = serializers.IntegerField()
    error_summary = serializers.CharField(allow_null=True)


class WorkflowPreviewResponseSerializer(serializers.Serializer):
    output_columns = serializers.ListField(child=serializers.CharField())
    input_row_count = serializers.IntegerField()
    output_row_count = serializers.IntegerField()
    preview_limit = serializers.IntegerField()
    preview_rows = serializers.ListField(child=serializers.DictField())
    step_audits = WorkflowPreviewStepAuditSerializer(many=True)
