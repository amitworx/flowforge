"""Constants, serializer, and bounded helpers for portable workflow definitions (M0-011)."""

from typing import Any

from rest_framework import serializers

from flowforge.workflows.engine import WorkflowExecutionError, validate_step_definitions
from flowforge.workflows.export import serialize_to_bounded_json

# Maximum allowable encoded size for workflow portability (64 KiB)
MAX_WORKFLOW_PORTABILITY_SIZE_BYTES = 64 * 1024
WORKFLOW_INTERCHANGE_FORMAT = "flowforge.workflow"
WORKFLOW_INTERCHANGE_VERSION = 1


def serialize_workflow_definition_export(
    name: str,
    steps: list[dict[str, Any]],
    max_bytes: int = MAX_WORKFLOW_PORTABILITY_SIZE_BYTES,
) -> bytes:
    """
    Serializes a workflow definition into the deterministic v1 interchange JSON envelope.
    Verifies step validity and encodes with an incremental UTF-8 byte boundary.
    """
    # Defensive step validation before serialization
    validate_step_definitions(steps)

    envelope = {
        "format": WORKFLOW_INTERCHANGE_FORMAT,
        "format_version": WORKFLOW_INTERCHANGE_VERSION,
        "name": name,
        "steps": steps,
    }
    return serialize_to_bounded_json(envelope, max_bytes=max_bytes, indent=2)


class WorkflowImportSerializer(serializers.Serializer):
    """
    Validates a v1 portable workflow definition JSON envelope.
    Rejects unexpected keys, wrong format/version, invalid names, and invalid steps.
    """

    format = serializers.CharField(required=True)
    format_version = serializers.IntegerField(required=True)
    name = serializers.CharField(required=True, allow_blank=False, max_length=128)
    steps = serializers.ListField(required=True, allow_empty=False)

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError(
                {"detail": "Invalid payload format. Expected a JSON object."}
            )

        allowed_fields = {"format", "format_version", "name", "steps"}
        input_keys = set(data.keys())
        unknown_fields = input_keys - allowed_fields
        if unknown_fields:
            raise serializers.ValidationError(
                dict.fromkeys(
                    sorted(unknown_fields),
                    "Unknown field. Only 'format', 'format_version', 'name', and 'steps' are accepted.",
                )
            )

        return super().to_internal_value(data)

    def validate_format(self, value):
        if value != WORKFLOW_INTERCHANGE_FORMAT:
            raise serializers.ValidationError(
                f"Unsupported format '{value}'. Expected '{WORKFLOW_INTERCHANGE_FORMAT}'."
            )
        return value

    def validate_format_version(self, value):
        raw_version = self.initial_data.get("format_version")
        if not isinstance(raw_version, int) or isinstance(raw_version, bool):
            raise serializers.ValidationError("Format version must be an integer.")
        if value != WORKFLOW_INTERCHANGE_VERSION:
            raise serializers.ValidationError(
                f"Unsupported format_version '{value}'. Expected '{WORKFLOW_INTERCHANGE_VERSION}'."
            )
        return value

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
