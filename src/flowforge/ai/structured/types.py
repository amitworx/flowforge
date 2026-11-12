"""Public type definitions for the structured output validation layer.

These are the data types that callers use to declare contracts and
receive validated results.  They depend only on the standard library
and the exceptions module; no Django, Pydantic, Celery or provider code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Supported schema keywords
# ---------------------------------------------------------------------------
#
# This module deliberately documents the *small subset* of JSON Schema that
# the validator actually implements.  Any keyword not in SUPPORTED_KEYWORDS is
# rejected at schema-validation time (SchemaError) rather than silently
# ignored, so callers can rely on predictable, auditable behaviour.
#
SUPPORTED_TYPE_NAMES: frozenset[str] = frozenset(
    {"object", "array", "string", "integer", "number", "boolean", "null"}
)

SUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
    }
)

# ---------------------------------------------------------------------------
# Hard limits
# ---------------------------------------------------------------------------

# Maximum UTF-8 byte length of raw model response text accepted for parsing.
MAX_PAYLOAD_BYTES: int = 1_048_576  # 1 MiB

# Maximum number of JSON nodes (keys + values + array items) allowed.
MAX_JSON_NODES: int = 10_000

# Maximum object nesting depth.
MAX_NESTING_DEPTH: int = 20

# Maximum number of object properties validated under a single schema node.
MAX_SCHEMA_PROPERTIES: int = 200


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedOutput:
    """Holds a successfully validated and type-coerced JSON value.

    Attributes:
        value:  The validated Python value (dict, list, str, int, float, bool, None).
                All keys are plain strings; all numbers are finite Python ints/floats.
        schema_type:  The root-level ``type`` from the schema (may be empty string
                      when the schema carries no ``type`` keyword at the top level,
                      which is legal only when validating primitive literals).
    """

    value: Any
    schema_type: str = ""


@dataclass
class SchemaContract:
    """A caller-declared JSON Schema-like contract for structured output.

    Only the keywords in ``SUPPORTED_SCHEMA_KEYWORDS`` are accepted.
    Unknown keywords cause a ``SchemaError`` to be raised before any
    model output is parsed.

    Fields map 1-to-1 to JSON Schema keywords.  The ``raw`` dict is kept
    for internal traversal and must not be mutated after construction.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, schema: dict[str, Any]) -> SchemaContract:
        """Construct and statically validate a SchemaContract from a plain dict."""
        from flowforge.ai.structured.validator import _validate_schema_shape

        _validate_schema_shape(schema, depth=0)
        return cls(raw=schema)
