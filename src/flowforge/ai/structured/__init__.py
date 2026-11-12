"""Public package for structured LLM output validation.

M0-015A — Pure Structured LLM Output Validation (Independent Track).

This package provides a pure, reusable JSON schema-based validator for
LLM model responses.  It is intentionally decoupled from the workflow
engine; integration into workflow steps is deferred to M0-015B.

Supported schema subset
-----------------------
Keywords: type, properties, required, additionalProperties, items,
          enum, minLength, maxLength, minItems, maxItems, minimum, maximum.

Types: object, array, string, integer, number, boolean, null.

Unknown keywords -> SchemaError (rejected, never silently ignored).
"""

from flowforge.ai.structured.exceptions import (
    ParseError,
    PayloadLimitError,
    SchemaError,
    StructuredOutputError,
    ValidationError,
    ValidationPath,
)
from flowforge.ai.structured.types import (
    MAX_JSON_NODES,
    MAX_NESTING_DEPTH,
    MAX_PAYLOAD_BYTES,
    SUPPORTED_SCHEMA_KEYWORDS,
    SUPPORTED_TYPE_NAMES,
    SchemaContract,
    ValidatedOutput,
)
from flowforge.ai.structured.validator import validate_structured_output

__all__ = [
    # Main entry point
    "validate_structured_output",
    # Contract
    "SchemaContract",
    "ValidatedOutput",
    # Exceptions
    "StructuredOutputError",
    "ParseError",
    "PayloadLimitError",
    "SchemaError",
    "ValidationError",
    "ValidationPath",
    # Limits (for callers that need to enforce them upstream)
    "MAX_PAYLOAD_BYTES",
    "MAX_JSON_NODES",
    "MAX_NESTING_DEPTH",
    # Introspection
    "SUPPORTED_SCHEMA_KEYWORDS",
    "SUPPORTED_TYPE_NAMES",
]
