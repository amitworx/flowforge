"""Constants, serializer, parser, and bounded helpers for portable workflow definitions (M0-011 / M0-011A)."""

import json
import math
from typing import Any

from rest_framework import serializers
from rest_framework.exceptions import ParseError
from rest_framework.parsers import BaseParser

from flowforge.workflows.engine import WorkflowExecutionError, validate_step_definitions
from flowforge.workflows.export import serialize_to_bounded_json

# Maximum allowable encoded size for workflow portability (64 KiB)
MAX_WORKFLOW_PORTABILITY_SIZE_BYTES = 64 * 1024
WORKFLOW_INTERCHANGE_FORMAT = "flowforge.workflow"
WORKFLOW_INTERCHANGE_VERSION = 1


class BoundedPayloadTooLargeError(Exception):
    """Raised when an incoming request stream exceeds MAX_WORKFLOW_PORTABILITY_SIZE_BYTES."""

    def __init__(
        self,
        message: str = f"Import payload exceeds maximum allowed size of {MAX_WORKFLOW_PORTABILITY_SIZE_BYTES} bytes.",
    ):
        super().__init__(message)
        self.message = message


def _reject_non_finite_float_constants(val: str) -> None:
    """Callback for json.loads parse_constant to disallow NaN, Infinity, -Infinity."""
    raise ValueError(f"Non-standard JSON constant not allowed: '{val}'")


def _validate_finite_numbers_in_steps(steps: list[dict[str, Any]]) -> None:
    """Ensure all floats in step definitions are finite numbers (no NaN or Infinity)."""
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        val = step.get("value")
        if isinstance(val, float) and not math.isfinite(val):
            raise WorkflowExecutionError(
                f"Step {idx} ({step.get('op', 'unknown')}): Non-finite float value '{val}' is not permitted."
            )


def read_bounded_request_stream(
    stream: Any, max_bytes: int = MAX_WORKFLOW_PORTABILITY_SIZE_BYTES
) -> bytes:
    """
    Reads at most (max_bytes + 1) bytes from a framework-provided request stream in bounded chunks.
    Raises BoundedPayloadTooLargeError as soon as accumulated bytes exceed max_bytes.
    This guarantees that unparsed payload data never consumes unbounded memory.
    Handles Django test FakePayload len semantics gracefully without reading beyond framing.
    """
    accumulated = bytearray()
    chunk_size = min(max_bytes + 1, 8192)

    while True:
        # Calculate target read size for this step
        remaining_to_max = (max_bytes + 1) - len(accumulated)
        to_read = min(chunk_size, remaining_to_max)

        # In Django test Client FakePayload, stream.read(size) asserts len(stream) >= size
        if hasattr(stream, "__len__") and not isinstance(stream, (bytes, bytearray)):
            try:
                available = len(stream)
                if available == 0:
                    break
                to_read = min(to_read, available)
            except (TypeError, Exception):
                pass

        chunk = stream.read(to_read)
        if not chunk:
            break
        accumulated.extend(chunk)
        if len(accumulated) > max_bytes:
            raise BoundedPayloadTooLargeError(
                f"Import payload exceeds maximum allowed size of {max_bytes} bytes."
            )

    return bytes(accumulated)


class BoundedJSONParser(BaseParser):
    """
    DRF Parser for application/json that enforces strict byte length constraints
    at the stream level before parsing JSON into memory, respecting the framework's
    request-scoped stream framing.

    Transport Semantics & Boundaries:
    - In standard WSGI (PEP 3333) and Django (`WSGIRequest`), `stream` wraps `environ['wsgi.input']`
      with a `LimitedStream` bounded by the declared `Content-Length`.
    - This parser reads exclusively from the framework-provided `stream` without bypassing
      or directly reading `environ['wsgi.input']`, guaranteeing that bytes from any subsequent
      pipelined HTTP request are never consumed.
    - If the incoming stream exceeds 64 KiB (65,536 bytes), reading stops at exactly 65,537 bytes
      and raises `BoundedPayloadTooLargeError`, returning HTTP 413 Payload Too Large before JSON decoding.
    - Server/Proxy framing is authoritative: if an upstream client sends an inaccurate-small Content-Length,
      the WSGI server frames the body to that declared size and does not expose additional socket bytes.
    """

    media_type = "application/json"

    def parse(self, stream, media_type=None, parser_context=None):  # noqa: ARG002
        if stream is None:
            raise ParseError("Request body cannot be empty.")

        raw_bytes = read_bounded_request_stream(stream, MAX_WORKFLOW_PORTABILITY_SIZE_BYTES)
        if not raw_bytes:
            raise ParseError("Request body cannot be empty.")

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError("Invalid payload encoding. Expected valid UTF-8 JSON.") from exc

        try:
            return json.loads(text, parse_constant=_reject_non_finite_float_constants)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ParseError(f"Malformed JSON payload: {exc}.") from exc


def serialize_workflow_definition_export(
    name: str,
    steps: list[dict[str, Any]],
    max_bytes: int = MAX_WORKFLOW_PORTABILITY_SIZE_BYTES,
) -> bytes:
    """
    Serializes a workflow definition into the deterministic v1 interchange JSON envelope.
    Verifies step validity, rejects non-finite floats, and encodes with an incremental UTF-8 byte boundary.
    """
    # Defensive step validation before serialization
    validate_step_definitions(steps)
    _validate_finite_numbers_in_steps(steps)

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
            validated = validate_step_definitions(value)
            _validate_finite_numbers_in_steps(validated)
            return validated
        except WorkflowExecutionError as exc:
            raise serializers.ValidationError(exc.message) from exc
