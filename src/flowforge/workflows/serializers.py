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
        raw_name = self.initial_data.get("name")
        if not isinstance(raw_name, str):
            raise serializers.ValidationError("Workflow name must be a string.")
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
        raw_name = self.initial_data.get("name")
        if not isinstance(raw_name, str):
            raise serializers.ValidationError("Workflow name must be a string.")
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


class WorkflowRunFilterSerializer(serializers.Serializer):
    ALLOWED_PARAMS = {
        "workflow_definition",
        "state",
        "source_job",
        "created_from",
        "created_to",
        "ordering",
        "page",
        "page_size",
    }

    workflow_definition = serializers.UUIDField(required=False)
    source_job = serializers.UUIDField(required=False)
    state = serializers.ChoiceField(choices=WorkflowRun.State.choices, required=False)
    created_from = serializers.CharField(required=False)
    created_to = serializers.CharField(required=False)
    ordering = serializers.ChoiceField(
        choices=["-created_at", "created_at"], required=False, default="-created_at"
    )

    def validate(self, attrs):
        # Check unknown parameters from initial data
        unknown = set(self.initial_data.keys()) - self.ALLOWED_PARAMS
        if unknown:
            raise serializers.ValidationError(
                dict.fromkeys(
                    sorted(unknown),
                    f"Unknown query parameter. Allowed parameters are: {sorted(self.ALLOWED_PARAMS)}.",
                )
            )

        from django.utils import timezone
        from django.utils.dateparse import parse_datetime

        def _parse_iso_aware(val: str, field_name: str):
            val_clean = val.strip()
            # Standard URL query decoding converts unencoded '+' into a space (' ')
            # Restore '+' if the part after space looks like a timezone offset
            if " " in val_clean and ("+" not in val_clean and not val_clean.endswith("Z")):
                parts = val_clean.split(" ")
                if len(parts) == 2 and (":" in parts[1] or len(parts[1]) == 4):
                    val_clean = f"{parts[0]}+{parts[1]}"

            dt = parse_datetime(val_clean)
            if dt is None:
                raise serializers.ValidationError({field_name: "Invalid ISO 8601 datetime format."})
            if not timezone.is_aware(dt):
                raise serializers.ValidationError(
                    {field_name: "Datetime must be timezone-aware (e.g. UTC 'Z' or offset)."}
                )
            return dt

        created_from_str = attrs.get("created_from")
        if created_from_str:
            attrs["created_from_dt"] = _parse_iso_aware(created_from_str, "created_from")

        created_to_str = attrs.get("created_to")
        if created_to_str:
            attrs["created_to_dt"] = _parse_iso_aware(created_to_str, "created_to")

        if (
            "created_from_dt" in attrs
            and "created_to_dt" in attrs
            and attrs["created_from_dt"] > attrs["created_to_dt"]
        ):
            raise serializers.ValidationError(
                {"created_from": "'created_from' cannot be later than 'created_to'."}
            )

        return attrs


class WorkflowCloneRequestSerializer(serializers.Serializer):
    """
    Serializer for POST /api/v1/workflows/{uuid}/clone/ payload.
    Optional 'name' string (1-128 chars). Rejects unknown fields and malformed payloads.
    """

    name = serializers.CharField(required=False, allow_blank=False, max_length=128)

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError(
                {"detail": "Invalid payload format. Expected a JSON object."}
            )

        allowed_fields = {"name"}
        input_keys = set(data.keys())
        unknown_fields = input_keys - allowed_fields
        if unknown_fields:
            raise serializers.ValidationError(
                dict.fromkeys(
                    sorted(unknown_fields),
                    "Unknown field. Only 'name' is accepted.",
                )
            )

        return super().to_internal_value(data)

    def validate_name(self, value):
        raw_name = self.initial_data.get("name")
        if not isinstance(raw_name, str):
            raise serializers.ValidationError("Workflow name must be a string.")
        stripped = value.strip()
        if not stripped:
            raise serializers.ValidationError("Workflow name cannot be empty.")
        if len(stripped) > 128:
            raise serializers.ValidationError("Workflow name cannot exceed 128 characters.")
        return stripped
