"""Pure structured LLM output validator — M0-015A core implementation.

Public API
----------
validate_structured_output(raw_text, schema) -> ValidatedOutput
    Parse ``raw_text`` as strict JSON and validate it against ``schema``.
    Accepts ONLY ``str`` input (rejects bytes, int, None, dict, etc.).
    Raises ParseError, PayloadLimitError, ValidationError, or SchemaError.
    CPU-only, deterministic, no database or network calls.

SchemaContract.from_dict(d) -> SchemaContract
    Statically validate a JSON Schema-like dict (small subset only) and
    return a typed contract object.  The input dict is deep-copied so
    caller mutations cannot affect the contract after construction.

Supported schema subset
-----------------------
Keywords: type, properties, required, additionalProperties, items, enum,
          minLength, maxLength, minItems, maxItems, minimum, maximum.

Types: object, array, string, integer, number, boolean, null.

Unknown keywords -> SchemaError (rejected, not silently ignored).
NaN/Inf numbers -> ParseError.
Duplicate object keys -> ParseError.
Deep nesting (> MAX_NESTING_DEPTH) -> PayloadLimitError.
Oversized payload (> MAX_PAYLOAD_BYTES) -> PayloadLimitError.
Too many JSON nodes (> MAX_JSON_NODES) -> PayloadLimitError.
Non-str input -> ParseError with safe message.

Enum comparisons are type-sensitive: Python's True==1 and 1==1.0
coercions are explicitly prevented via identity-based type checks.

Schema numeric constraints are validated for finiteness, non-negativity,
integrality (for length/item counts), and min<=max consistency.

Schema size is capped (MAX_SCHEMA_NODES total nodes) to prevent
schema-bomb denial-of-service attacks.

Error messages never contain raw model output, duplicate key values,
property values, or any credential/secret content.

Integration note
----------------
Workflow output integration is explicitly NOT delivered by M0-015A.
That wiring is deferred to M0-015B, which builds on a stable M0-014.
"""

from __future__ import annotations

import json
import math
from typing import Any

from flowforge.ai.structured.exceptions import (
    ParseError,
    PayloadLimitError,
    SchemaError,
    ValidationError,
    ValidationPath,
)
from flowforge.ai.structured.types import (
    MAX_ENUM_VALUES,
    MAX_JSON_NODES,
    MAX_NESTING_DEPTH,
    MAX_PAYLOAD_BYTES,
    MAX_REQUIRED_FIELDS,
    MAX_SCHEMA_NODES,
    MAX_SCHEMA_PROPERTIES,
    SUPPORTED_SCHEMA_KEYWORDS,
    SUPPORTED_TYPE_NAMES,
    SchemaContract,
    ValidatedOutput,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_structured_output(
    raw_text: str,
    schema: SchemaContract,
) -> ValidatedOutput:
    """Parse *raw_text* as strict JSON and validate it against *schema*.

    Parameters
    ----------
    raw_text:
        UTF-8 model response text as a plain ``str``.  Non-str values
        (bytes, int, None, dict …) are rejected with a safe ParseError.
        Must not exceed ``MAX_PAYLOAD_BYTES``.
        NaN/Infinity JSON literals, duplicate object keys, deeply nested
        structures and oversized payloads are all rejected.
    schema:
        A ``SchemaContract`` that declares the expected type contract.
        Obtain one via ``SchemaContract.from_dict()``.

    Returns
    -------
    ValidatedOutput
        On success, containing the validated Python value.

    Raises
    ------
    ParseError
        When *raw_text* is not a ``str``, or cannot be decoded as valid
        strict JSON, or contains NaN/Infinity.
    PayloadLimitError
        When payload byte size, node count or nesting depth exceeds limits.
    ValidationError
        When the parsed JSON value does not conform to *schema*.
    SchemaError
        When *schema* uses unsupported keywords (re-raised as a safety net
        in case the SchemaContract was constructed without from_dict()).
    """
    # 0. Input type guard — string-only API
    if not isinstance(raw_text, str):
        raise ParseError(f"raw_text must be str, got {type(raw_text).__name__!r}")

    # 1. Byte-size guard — before any parsing
    try:
        encoded = raw_text.encode("utf-8", errors="replace")
    except (UnicodeEncodeError, UnicodeDecodeError):
        raise ParseError("raw_text contains unencodable characters") from None
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise PayloadLimitError(
            f"response is {len(encoded):,} bytes; limit is {MAX_PAYLOAD_BYTES:,} bytes"
        )

    # 2. Parse as strict JSON (custom decoder to catch NaN/Inf and duplicate keys)
    value = _strict_parse(raw_text)

    # 3. Node-count and depth limits
    _count = [0]
    _check_limits(value, depth=0, count=_count, path=ValidationPath.root())

    # 4. Schema validation (re-validate schema shape as a safety net for
    #    contracts built without from_dict)
    _validate_schema_shape(schema.raw, depth=0, node_count=[0])

    root_type = schema.raw.get("type", "")
    _validate_value(value, schema.raw, path=ValidationPath.root())

    return ValidatedOutput(value=value, schema_type=str(root_type))


# ---------------------------------------------------------------------------
# Schema shape validator (called by SchemaContract.from_dict)
# ---------------------------------------------------------------------------


def _validate_schema_shape(schema: Any, *, depth: int, node_count: list[int]) -> None:
    """Recursively verify that *schema* uses only supported keywords and types.

    Also validates:
    - Numeric constraints are finite, non-NaN, non-negative where applicable,
      integral for length/item counts, and min <= max.
    - enum is bounded (MAX_ENUM_VALUES).
    - required is bounded (MAX_REQUIRED_FIELDS).
    - Total schema node count is bounded (MAX_SCHEMA_NODES).

    Raises SchemaError on the first unsupported or invalid construct found.
    """
    if depth > MAX_NESTING_DEPTH:
        raise SchemaError("schema nesting depth exceeds limit")
    if not isinstance(schema, dict):
        raise SchemaError(f"schema node must be a dict, got {type(schema).__name__!r}")

    node_count[0] += 1
    if node_count[0] > MAX_SCHEMA_NODES:
        raise SchemaError(
            f"schema has more than {MAX_SCHEMA_NODES} total nodes; simplify the schema or split it"
        )

    unsupported = set(schema.keys()) - SUPPORTED_SCHEMA_KEYWORDS
    if unsupported:
        raise SchemaError(
            f"unsupported schema keyword detected ({len(unsupported)} unexpected keyword(s)); "
            f"supported: {', '.join(sorted(SUPPORTED_SCHEMA_KEYWORDS))}"
        )

    if "type" in schema:
        type_val = schema["type"]
        if isinstance(type_val, str):
            if type_val not in SUPPORTED_TYPE_NAMES:
                raise SchemaError(
                    f"unsupported schema type detected; "
                    f"supported: {', '.join(sorted(SUPPORTED_TYPE_NAMES))}"
                )
        elif isinstance(type_val, list):
            for t in type_val:
                if not isinstance(t, str) or t not in SUPPORTED_TYPE_NAMES:
                    raise SchemaError(
                        f"unsupported schema type detected in union; "
                        f"supported: {', '.join(sorted(SUPPORTED_TYPE_NAMES))}"
                    )
        else:
            raise SchemaError(
                f"schema 'type' must be a string or list of strings, "
                f"got {type(type_val).__name__!r}"
            )

    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict):
            raise SchemaError("schema 'properties' must be a dict")
        if len(props) > MAX_SCHEMA_PROPERTIES:
            raise SchemaError(
                f"schema has {len(props)} properties; limit is {MAX_SCHEMA_PROPERTIES}"
            )
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_name, str):
                raise SchemaError(
                    f"property key must be a string, got {type(prop_name).__name__!r}"
                )
            _validate_schema_shape(prop_schema, depth=depth + 1, node_count=node_count)

    if "required" in schema:
        req = schema["required"]
        if not isinstance(req, list):
            raise SchemaError("schema 'required' must be a list")
        if len(req) > MAX_REQUIRED_FIELDS:
            raise SchemaError(
                f"schema 'required' has {len(req)} entries; limit is {MAX_REQUIRED_FIELDS}"
            )
        for item in req:
            if not isinstance(item, str):
                raise SchemaError(f"'required' items must be strings, got {type(item).__name__!r}")

    if "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        if not isinstance(ap, bool):
            if isinstance(ap, dict):
                _validate_schema_shape(ap, depth=depth + 1, node_count=node_count)
            else:
                raise SchemaError("schema 'additionalProperties' must be a bool or schema dict")

    if "items" in schema:
        _validate_schema_shape(schema["items"], depth=depth + 1, node_count=node_count)

    if "enum" in schema:
        enum_val = schema["enum"]
        if not isinstance(enum_val, list):
            raise SchemaError("schema 'enum' must be a list")
        if len(enum_val) == 0:
            raise SchemaError("schema 'enum' must not be empty")
        if len(enum_val) > MAX_ENUM_VALUES:
            raise SchemaError(
                f"schema 'enum' has {len(enum_val)} values; limit is {MAX_ENUM_VALUES}"
            )

    # Validate numeric constraint keywords
    # Length/item constraints: must be non-negative integers
    for kw in ("minLength", "maxLength", "minItems", "maxItems"):
        if kw in schema:
            _validate_schema_integer_constraint(schema[kw], kw)

    # Numeric range constraints: must be finite reals
    for kw in ("minimum", "maximum"):
        if kw in schema:
            _validate_schema_numeric_constraint(schema[kw], kw)

    # Cross-constraint consistency checks
    if (
        "minLength" in schema
        and "maxLength" in schema
        and schema["minLength"] > schema["maxLength"]
    ):
        raise SchemaError(
            f"schema minLength ({schema['minLength']}) exceeds maxLength ({schema['maxLength']})"
        )
    if "minItems" in schema and "maxItems" in schema and schema["minItems"] > schema["maxItems"]:
        raise SchemaError(
            f"schema minItems ({schema['minItems']}) exceeds maxItems ({schema['maxItems']})"
        )
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise SchemaError(
            f"schema minimum ({schema['minimum']}) exceeds maximum ({schema['maximum']})"
        )


def _validate_schema_integer_constraint(value: Any, keyword: str) -> None:
    """Validate that a length/count schema constraint is a non-negative integer."""
    if isinstance(value, bool):
        raise SchemaError(f"schema '{keyword}' must be a non-negative integer, got bool")
    if not isinstance(value, (int, float)):
        raise SchemaError(
            f"schema '{keyword}' must be a non-negative integer, got {type(value).__name__!r}"
        )
    # Reject NaN and Inf
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise SchemaError(
            f"schema '{keyword}' must be a finite non-negative integer, got non-finite float"
        )
    # Reject fractional values
    if isinstance(value, float) and not value.is_integer():
        raise SchemaError(
            f"schema '{keyword}' must be an integer value, got fractional float {value!r}"
        )
    # Reject negative values
    int_val = int(value)
    if int_val < 0:
        raise SchemaError(f"schema '{keyword}' must be non-negative, got {int_val}")


def _validate_schema_numeric_constraint(value: Any, keyword: str) -> None:
    """Validate that a numeric range schema constraint is a finite real number."""
    if isinstance(value, bool):
        raise SchemaError(f"schema '{keyword}' must be a finite number, got bool")
    if not isinstance(value, (int, float)):
        raise SchemaError(f"schema '{keyword}' must be a number, got {type(value).__name__!r}")
    if math.isnan(value) or math.isinf(value):
        raise SchemaError(
            f"schema '{keyword}' must be a finite number, got non-finite value ({value!r})"
        )


# ---------------------------------------------------------------------------
# Strict JSON parsing
# ---------------------------------------------------------------------------


def _strict_parse(text: str) -> Any:
    """Parse *text* as strict JSON.

    Rejects:
    - NaN / Infinity (not valid JSON but Python's json module may accept them
      when ``parse_constant`` is in play; we guard with parse_float/parse_int,
      parse_constant, and post-parse checks)
    - Duplicate object keys
    - Control characters in strings (delegated to the stdlib decoder)
    - Anything that is not valid JSON
    """
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_safe_parse_float,
            parse_int=_safe_parse_int,
            parse_constant=_reject_nan_inf_constant,
        )
    except json.JSONDecodeError as exc:
        # Strip the offending text: expose only position info
        raise ParseError(f"invalid JSON at position {exc.pos}: {exc.msg}") from None
    except _NaNInfError as exc:
        raise ParseError(str(exc)) from None
    except _DuplicateKeyError as exc:
        raise ParseError(str(exc)) from None

    return value


class _NaNInfError(ValueError):
    pass


class _DuplicateKeyError(ValueError):
    pass


def _reject_nan_inf_constant(constant: str) -> None:
    """parse_constant hook: reject NaN, Infinity, -Infinity."""
    raise _NaNInfError(f"non-finite constant {constant!r} is not allowed in JSON")


def _safe_parse_float(s: str) -> float:
    """Custom float parser; rejects NaN and Infinity."""
    try:
        value = float(s)
    except ValueError:
        raise _NaNInfError("invalid float literal") from None
    if math.isnan(value) or math.isinf(value):
        raise _NaNInfError("non-finite number is not allowed")
    return value


def _safe_parse_int(s: str) -> int:
    """Custom int parser; rejects suspiciously large integers."""
    # Python handles arbitrary precision ints, but we cap them to prevent
    # schema-bomb style attacks where a single integer has millions of digits.
    if len(s) > 40:
        raise _NaNInfError(f"integer literal is too long ({len(s)} digits)")
    return int(s)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """object_pairs_hook that raises on duplicate keys.

    The error message includes only the fact that a duplicate exists and
    a safe description — it does NOT echo the key's associated values.
    """
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            # Safe error: mention that a duplicate exists but NOT the value
            raise _DuplicateKeyError("duplicate object key detected")
        seen.add(key)
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# Path helper for safe traversal within validator
# ---------------------------------------------------------------------------


def _enter_path(path: ValidationPath, segment: str | int) -> ValidationPath:
    """Enter a path segment safely within the validator.

    - Integer segments represent array indices and are preserved as numeric indices.
    - String segments represent object properties and are ALWAYS entered with
      `is_safe=False` (yielding the generic opaque placeholder '[property]'),
      ensuring that caller-controlled schema property names and model keys cannot
      leak into error paths, even if the public ValidationPath.enter default is True.
    """
    if isinstance(segment, int):
        return path.enter(segment)
    return path.enter(segment, is_safe=False)


# ---------------------------------------------------------------------------
# Payload limit checks (depth and node count)
# ---------------------------------------------------------------------------


def _check_limits(
    value: Any,
    *,
    depth: int,
    count: list[int],
    path: ValidationPath,
) -> None:
    """Recursively check nesting depth and total node count."""
    if depth > MAX_NESTING_DEPTH:
        raise PayloadLimitError(f"nesting depth exceeds {MAX_NESTING_DEPTH} at {path}")
    count[0] += 1
    if count[0] > MAX_JSON_NODES:
        raise PayloadLimitError(f"document exceeds {MAX_JSON_NODES:,} JSON nodes")

    if isinstance(value, dict):
        for _k, v in value.items():
            _check_limits(v, depth=depth + 1, count=count, path=_enter_path(path, "[property]"))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_limits(item, depth=depth + 1, count=count, path=_enter_path(path, i))


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_value(value: Any, schema: dict[str, Any], *, path: ValidationPath) -> None:
    """Validate *value* against *schema* at *path*.  Raises ValidationError on failure."""

    # Type check
    expected_type = schema.get("type")
    if expected_type:
        types = [expected_type] if isinstance(expected_type, str) else expected_type
        if not any(_matches_type(value, t) for t in types):
            type_names = " | ".join(types)
            raise ValidationError(
                f"expected {type_names}, got {_safe_type_name(value)}",
                path=path,
                schema_type=type_names,
                code="TYPE_MISMATCH",
            )

    # Enum — strict type-sensitive comparison
    if "enum" in schema:
        allowed = schema["enum"]
        if not _enum_contains(value, allowed):
            raise ValidationError(
                f"value is not one of the allowed enum values "
                f"(got {_safe_type_name(value)} with {len(allowed)} choices)",
                path=path,
                schema_type="enum",
                code="ENUM_MISMATCH",
            )

    # String constraints
    if isinstance(value, str):
        _validate_string(value, schema, path)

    # Numeric constraints
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value, schema, path)

    # Object
    if isinstance(value, dict):
        _validate_object(value, schema, path)

    # Array
    if isinstance(value, list):
        _validate_array(value, schema, path)


def _enum_contains(value: Any, allowed: list[Any]) -> bool:
    """Type-sensitive enum membership check.

    Python's ``in`` operator uses ``==`` which silently coerces:
    - ``True == 1`` -> True
    - ``False == 0`` -> True
    - ``1 == 1.0`` -> True

    This function prevents those coercions by additionally checking that
    the JSON type of *value* matches the JSON type of each candidate.
    """
    value_type = _safe_type_name(value)
    for candidate in allowed:
        if _safe_type_name(candidate) == value_type and candidate == value:
            return True
    return False


def _matches_type(value: Any, type_name: str) -> bool:
    """Return True if *value* matches the JSON Schema *type_name*."""
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        # Strictly check Python bool: int values True/False must NOT match
        # integer or number types — only "boolean"
        return isinstance(value, bool)
    if type_name == "integer":
        # bool is a subclass of int in Python; reject bool-as-int
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        # bool is a subclass of int in Python; reject bool-as-number
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _safe_type_name(value: Any) -> str:
    """Return a JSON Schema type name for *value* safe for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_string(value: str, schema: dict[str, Any], path: ValidationPath) -> None:
    min_len = schema.get("minLength")
    max_len = schema.get("maxLength")
    if min_len is not None and len(value) < min_len:
        raise ValidationError(
            f"string length {len(value)} is below minimum {min_len}",
            path=path,
            schema_type="string",
            code="STRING_TOO_SHORT",
        )
    if max_len is not None and len(value) > max_len:
        raise ValidationError(
            f"string length {len(value)} exceeds maximum {max_len}",
            path=path,
            schema_type="string",
            code="STRING_TOO_LONG",
        )


def _validate_number(value: int | float, schema: dict[str, Any], path: ValidationPath) -> None:
    # Reject non-finite at validation time too (belt-and-suspenders)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValidationError(
            "non-finite number is not allowed",
            path=path,
            schema_type="number",
            code="NON_FINITE_NUMBER",
        )
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and value < minimum:
        raise ValidationError(
            f"value is below minimum {minimum}",
            path=path,
            schema_type="number",
            code="NUMBER_TOO_SMALL",
        )
    if maximum is not None and value > maximum:
        raise ValidationError(
            f"value exceeds maximum {maximum}",
            path=path,
            schema_type="number",
            code="NUMBER_TOO_LARGE",
        )


def _validate_object(value: dict[str, Any], schema: dict[str, Any], path: ValidationPath) -> None:
    # Required fields
    required = schema.get("required", [])
    missing_count = 0
    for field_name in required:
        if field_name not in value:
            missing_count += 1

    if missing_count > 0:
        prop_str = "property" if missing_count == 1 else "properties"
        raise ValidationError(
            f"required {prop_str} missing (found {missing_count} missing required {prop_str})",
            path=path,
            schema_type="object",
            code="MISSING_REQUIRED",
        )

    # Properties
    properties = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        if prop_name in value:
            _validate_value(value[prop_name], prop_schema, path=_enter_path(path, prop_name))

    # Additional properties
    additional = schema.get("additionalProperties")
    if additional is False:
        extra_keys = sorted(set(value.keys()) - set(properties.keys()))
        if extra_keys:
            # Safe error: report ONLY generic bounded count metadata.
            # Never leak raw, truncated, or escaped model-controlled property names.
            count = len(extra_keys)
            prop_str = "property" if count == 1 else "properties"
            raise ValidationError(
                f"additional properties not allowed (found {count} unexpected {prop_str})",
                path=path,
                schema_type="object",
                code="ADDITIONAL_PROPERTIES",
            )
    elif isinstance(additional, dict):
        # additionalProperties is itself a schema — validate extra keys against it.
        # Dynamic extra keys are untrusted model output, so enter path safely with is_safe=False.
        extra_keys = set(value.keys()) - set(properties.keys())
        for k in sorted(extra_keys):
            _validate_value(value[k], additional, path=_enter_path(path, k))


def _validate_array(value: list[Any], schema: dict[str, Any], path: ValidationPath) -> None:
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if min_items is not None and len(value) < min_items:
        raise ValidationError(
            f"array has {len(value)} items, minimum is {min_items}",
            path=path,
            schema_type="array",
            code="ARRAY_TOO_SHORT",
        )
    if max_items is not None and len(value) > max_items:
        raise ValidationError(
            f"array has {len(value)} items, maximum is {max_items}",
            path=path,
            schema_type="array",
            code="ARRAY_TOO_LONG",
        )

    items_schema = schema.get("items")
    if items_schema is not None:
        for i, item in enumerate(value):
            _validate_value(item, items_schema, path=_enter_path(path, i))
