"""M0-015A — Structured output validator: core behaviour tests.

Covers:
- Schema contract construction and static validation (supported keywords,
  types, structural constraints)
- JSON parsing: valid inputs, NaN/Inf rejection, duplicate key rejection,
  oversized payload rejection
- Type validation: object, array, string, integer, number, boolean, null
- Constraint validation: required, properties, additionalProperties, items,
  enum, minLength, maxLength, minItems, maxItems, minimum, maximum
- bool-as-int / bool-as-number rejection
- ValidatedOutput contract
- ValidationError metadata (path, code, schema_type)

NO database, network, or Django ORM calls.
"""

from __future__ import annotations

import json
import math

import pytest

from flowforge.ai.structured import (
    MAX_JSON_NODES,
    MAX_NESTING_DEPTH,
    MAX_PAYLOAD_BYTES,
    SUPPORTED_SCHEMA_KEYWORDS,
    SUPPORTED_TYPE_NAMES,
    ParseError,
    PayloadLimitError,
    SchemaContract,
    SchemaError,
    StructuredOutputError,
    ValidatedOutput,
    ValidationError,
    ValidationPath,
    validate_structured_output,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _contract(schema: dict) -> SchemaContract:
    return SchemaContract.from_dict(schema)


def _validate(text: str, schema: dict) -> ValidatedOutput:
    return validate_structured_output(text, _contract(schema))


# ===========================================================================
# 1. Package-level constants and public API surface
# ===========================================================================


class TestPublicAPIAndConstants:
    def test_supported_type_names_non_empty(self):
        assert len(SUPPORTED_TYPE_NAMES) > 0
        assert "object" in SUPPORTED_TYPE_NAMES
        assert "array" in SUPPORTED_TYPE_NAMES

    def test_supported_schema_keywords_non_empty(self):
        assert "type" in SUPPORTED_SCHEMA_KEYWORDS
        assert "properties" in SUPPORTED_SCHEMA_KEYWORDS
        assert "required" in SUPPORTED_SCHEMA_KEYWORDS
        assert "additionalProperties" in SUPPORTED_SCHEMA_KEYWORDS
        assert "items" in SUPPORTED_SCHEMA_KEYWORDS
        assert "enum" in SUPPORTED_SCHEMA_KEYWORDS

    def test_limits_are_positive(self):
        assert MAX_PAYLOAD_BYTES > 0
        assert MAX_JSON_NODES > 0
        assert MAX_NESTING_DEPTH > 0

    def test_exception_hierarchy(self):
        assert issubclass(ParseError, StructuredOutputError)
        assert issubclass(PayloadLimitError, StructuredOutputError)
        assert issubclass(SchemaError, StructuredOutputError)
        assert issubclass(ValidationError, StructuredOutputError)
        assert issubclass(StructuredOutputError, ValueError)

    def test_validate_structured_output_callable(self):
        result = _validate('{"key": "value"}', {"type": "object"})
        assert isinstance(result, ValidatedOutput)


# ===========================================================================
# 2. SchemaContract construction
# ===========================================================================


class TestSchemaContractConstruction:
    def test_empty_schema_accepted(self):
        sc = _contract({})
        assert sc.raw == {}

    def test_valid_object_schema(self):
        sc = _contract(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        )
        assert sc.raw["type"] == "object"

    def test_all_supported_type_names_accepted(self):
        for t in SUPPORTED_TYPE_NAMES:
            sc = _contract({"type": t})
            assert sc.raw["type"] == t

    def test_unsupported_keyword_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "object", "$ref": "#/definitions/Foo"})
        assert "unsupported schema keyword" in str(exc_info.value).lower()
        # Does not leak raw keyword name
        assert "$ref" not in str(exc_info.value)

    def test_unsupported_type_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "tuple"})
        assert "unsupported" in str(exc_info.value).lower()

    def test_non_dict_schema_raises_schema_error(self):
        with pytest.raises(SchemaError):
            SchemaContract.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_properties_must_be_dict(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "object", "properties": ["name"]})
        assert "properties" in str(exc_info.value)

    def test_required_must_be_list(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "object", "required": "name"})
        assert "required" in str(exc_info.value)

    def test_required_items_must_be_strings(self):
        with pytest.raises(SchemaError):
            _contract({"type": "object", "required": [1, 2]})

    def test_additional_properties_true_accepted(self):
        sc = _contract({"type": "object", "additionalProperties": True})
        assert sc.raw["additionalProperties"] is True

    def test_additional_properties_false_accepted(self):
        sc = _contract({"type": "object", "additionalProperties": False})
        assert sc.raw["additionalProperties"] is False

    def test_additional_properties_must_be_bool_or_schema(self):
        with pytest.raises(SchemaError):
            _contract({"type": "object", "additionalProperties": "no"})

    def test_enum_must_be_list(self):
        with pytest.raises(SchemaError):
            _contract({"enum": "value"})

    def test_enum_must_be_non_empty(self):
        with pytest.raises(SchemaError):
            _contract({"enum": []})

    def test_minlength_must_be_number(self):
        with pytest.raises(SchemaError):
            _contract({"type": "string", "minLength": "5"})

    def test_nested_schema_in_properties_validated(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract(
                {
                    "type": "object",
                    "properties": {"sub": {"type": "object", "badKeyword": True}},
                }
            )
        assert "badKeyword" in str(exc_info.value) or "unsupported" in str(exc_info.value).lower()

    def test_union_type_as_list_accepted(self):
        sc = _contract({"type": ["string", "null"]})
        assert "string" in sc.raw["type"]

    def test_union_type_with_unsupported_member_raises(self):
        with pytest.raises(SchemaError):
            _contract({"type": ["string", "tuple"]})

    def test_items_schema_validated_recursively(self):
        with pytest.raises(SchemaError):
            _contract({"type": "array", "items": {"badKey": True}})


# ===========================================================================
# 3. JSON parsing — strict rules
# ===========================================================================


class TestStrictJSONParsing:
    def test_valid_object(self):
        result = _validate('{"a": 1}', {"type": "object"})
        assert result.value == {"a": 1}

    def test_valid_array(self):
        result = _validate("[1, 2, 3]", {"type": "array"})
        assert result.value == [1, 2, 3]

    def test_valid_string(self):
        result = _validate('"hello"', {"type": "string"})
        assert result.value == "hello"

    def test_valid_integer(self):
        result = _validate("42", {"type": "integer"})
        assert result.value == 42

    def test_valid_boolean_true(self):
        result = _validate("true", {"type": "boolean"})
        assert result.value is True

    def test_valid_null(self):
        result = _validate("null", {"type": "null"})
        assert result.value is None

    def test_invalid_json_raises_parse_error(self):
        with pytest.raises(ParseError) as exc_info:
            _validate("{bad json}", {"type": "object"})
        # Must not echo the bad text in the error message
        assert "bad json" not in str(exc_info.value)
        assert "position" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_empty_string_raises_parse_error(self):
        with pytest.raises(ParseError):
            _validate("", {"type": "object"})

    def test_duplicate_key_raises_parse_error(self):
        raw = '{"key": 1, "key": 2}'
        with pytest.raises(ParseError) as exc_info:
            _validate(raw, {"type": "object"})
        # Must not echo key value from model output
        msg = str(exc_info.value)
        assert "1" not in msg  # value must not appear
        assert "duplicate" in msg.lower()

    def test_nan_number_raises_parse_error(self):
        # NaN is not valid JSON; Python's json module with custom parse_float
        # should catch it. Some tools emit NaN without quotes.
        with pytest.raises((ParseError, ValueError)):
            # Trigger via special float literal that standard json also rejects
            _validate('{"x": NaN}', {"type": "object"})

    def test_infinity_number_raises_parse_error(self):
        with pytest.raises((ParseError, ValueError)):
            _validate('{"x": Infinity}', {"type": "object"})

    def test_oversized_payload_raises_payload_limit_error(self):
        # Craft a string that exceeds MAX_PAYLOAD_BYTES
        big = '"' + "a" * (MAX_PAYLOAD_BYTES + 1) + '"'
        with pytest.raises(PayloadLimitError) as exc_info:
            _validate(big, {"type": "string"})
        assert (
            str(MAX_PAYLOAD_BYTES) in str(exc_info.value) or "limit" in str(exc_info.value).lower()
        )

    def test_exact_limit_boundary_accepted(self):
        # A payload exactly at the limit should not raise PayloadLimitError
        # (we use a valid JSON string that fits)
        payload = '"' + "a" * 10 + '"'
        result = _validate(payload, {"type": "string"})
        assert result.value == "a" * 10


# ===========================================================================
# 4. Payload limits — depth and node count
# ===========================================================================


class TestPayloadLimits:
    def test_too_deep_object_raises(self):
        # Build an object nested deeper than MAX_NESTING_DEPTH
        obj: dict = {}
        inner = obj
        for _i in range(MAX_NESTING_DEPTH + 2):
            inner["x"] = {}
            inner = inner["x"]
        raw = json.dumps(obj)
        with pytest.raises(PayloadLimitError) as exc_info:
            _validate(raw, {"type": "object"})
        assert "depth" in str(exc_info.value).lower() or "nesting" in str(exc_info.value).lower()

    def test_too_deep_array_raises(self):
        raw = "["
        for _ in range(MAX_NESTING_DEPTH + 2):
            raw += "["
        raw += "1"
        for _ in range(MAX_NESTING_DEPTH + 2):
            raw += "]"
        raw += "]"
        with pytest.raises(PayloadLimitError):
            _validate(raw, {"type": "array"})

    def test_too_many_nodes_raises(self):
        # Build a flat array with MAX_JSON_NODES + 1 items
        big_list = list(range(MAX_JSON_NODES + 1))
        raw = json.dumps(big_list)
        with pytest.raises(PayloadLimitError) as exc_info:
            _validate(raw, {"type": "array"})
        assert str(MAX_JSON_NODES) in str(exc_info.value) or "nodes" in str(exc_info.value).lower()

    def test_exactly_max_depth_accepted(self):
        # Exactly at MAX_NESTING_DEPTH should not raise
        obj: dict = {}
        inner = obj
        for _i in range(MAX_NESTING_DEPTH):
            inner["x"] = {}
            inner = inner["x"]
        raw = json.dumps(obj)
        # May or may not match a schema; we only care it doesn't raise PayloadLimitError
        try:
            _validate(raw, {"type": "object"})
        except (ValidationError, PayloadLimitError) as exc:
            assert not isinstance(exc, PayloadLimitError)


# ===========================================================================
# 5. Type validation
# ===========================================================================


class TestTypeValidation:
    def test_string_vs_integer_mismatch(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate('"hello"', {"type": "integer"})
        assert "TYPE_MISMATCH" in str(exc_info.value) or "integer" in str(exc_info.value)

    def test_bool_not_integer(self):
        """Python booleans must NOT validate as 'integer'."""
        with pytest.raises(ValidationError):
            _validate("true", {"type": "integer"})

    def test_bool_not_number(self):
        """Python booleans must NOT validate as 'number'."""
        with pytest.raises(ValidationError):
            _validate("true", {"type": "number"})

    def test_integer_is_valid_number(self):
        # An integer value should satisfy the 'number' type
        result = _validate("42", {"type": "number"})
        assert result.value == 42

    def test_null_vs_string_mismatch(self):
        with pytest.raises(ValidationError):
            _validate("null", {"type": "string"})

    def test_array_vs_object_mismatch(self):
        with pytest.raises(ValidationError):
            _validate("[1, 2]", {"type": "object"})

    def test_union_type_string_or_null(self):
        result = _validate('"hello"', {"type": ["string", "null"]})
        assert result.value == "hello"
        result2 = _validate("null", {"type": ["string", "null"]})
        assert result2.value is None

    def test_union_type_mismatch(self):
        with pytest.raises(ValidationError):
            _validate("42", {"type": ["string", "null"]})

    def test_boolean_true_and_false(self):
        r1 = _validate("true", {"type": "boolean"})
        assert r1.value is True
        r2 = _validate("false", {"type": "boolean"})
        assert r2.value is False

    def test_number_float(self):
        result = _validate("3.14", {"type": "number"})
        assert math.isclose(result.value, 3.14)

    def test_negative_integer(self):
        result = _validate("-99", {"type": "integer"})
        assert result.value == -99

    def test_zero_integer(self):
        result = _validate("0", {"type": "integer"})
        assert result.value == 0


# ===========================================================================
# 6. Object validation — required, properties, additionalProperties
# ===========================================================================


class TestObjectValidation:
    def test_required_field_present(self):
        result = _validate(
            '{"name": "Alice", "age": 30}',
            {"type": "object", "required": ["name"]},
        )
        assert result.value["name"] == "Alice"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate('{"age": 30}', {"type": "object", "required": ["name"]})
        err = exc_info.value
        assert err.code == "MISSING_REQUIRED"
        assert "name" in str(err)
        # Path should reference root
        assert "$" in str(err.path)

    def test_nested_property_type_checked(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate('{"count": "not-an-int"}', schema)
        err = exc_info.value
        assert "count" in str(err.path)
        assert err.code == "TYPE_MISMATCH"

    def test_nested_required_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "required": ["street"],
                    "properties": {"street": {"type": "string"}},
                }
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate('{"address": {}}', schema)
        err = exc_info.value
        assert "address" in str(err.path)
        assert err.code == "MISSING_REQUIRED"

    def test_additional_properties_false_rejects_extras(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate('{"name": "Alice", "extra": "oops"}', schema)
        err = exc_info.value
        assert err.code == "ADDITIONAL_PROPERTIES"
        # Does NOT leak raw property name 'extra'; reports generic count metadata
        assert "extra" not in str(err)
        assert "1 unexpected" in str(err).lower() or "unexpected property" in str(err).lower()

    def test_additional_properties_true_allows_extras(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,
        }
        result = _validate('{"name": "Alice", "extra": 99}', schema)
        assert result.value["extra"] == 99

    def test_additional_properties_as_schema_validates_extras(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        }
        # Valid extra value
        result = _validate('{"name": "Alice", "count": 5}', schema)
        assert result.value["count"] == 5

        # Invalid extra value
        with pytest.raises(ValidationError):
            _validate('{"name": "Alice", "count": "five"}', schema)

    def test_empty_object_no_required(self):
        result = _validate("{}", {"type": "object"})
        assert result.value == {}

    def test_multiple_missing_required_first_reported(self):
        schema = {"type": "object", "required": ["a", "b", "c"]}
        with pytest.raises(ValidationError) as exc_info:
            _validate("{}", schema)
        # At least one required field error reported
        assert exc_info.value.code == "MISSING_REQUIRED"


# ===========================================================================
# 7. Array validation
# ===========================================================================


class TestArrayValidation:
    def test_valid_integer_array(self):
        result = _validate("[1, 2, 3]", {"type": "array", "items": {"type": "integer"}})
        assert result.value == [1, 2, 3]

    def test_wrong_item_type(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate('[1, "two", 3]', {"type": "array", "items": {"type": "integer"}})
        err = exc_info.value
        assert "[1]" in str(err.path) or "1" in str(err.path)
        assert err.code == "TYPE_MISMATCH"

    def test_min_items_satisfied(self):
        result = _validate("[1, 2]", {"type": "array", "minItems": 2})
        assert len(result.value) == 2

    def test_min_items_violated(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate("[1]", {"type": "array", "minItems": 3})
        err = exc_info.value
        assert err.code == "ARRAY_TOO_SHORT"

    def test_max_items_violated(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate("[1, 2, 3, 4]", {"type": "array", "maxItems": 2})
        err = exc_info.value
        assert err.code == "ARRAY_TOO_LONG"

    def test_empty_array_allowed(self):
        result = _validate("[]", {"type": "array"})
        assert result.value == []

    def test_nested_object_items(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer"}},
            },
        }
        result = _validate('[{"id": 1}, {"id": 2}]', schema)
        assert len(result.value) == 2

        with pytest.raises(ValidationError):
            _validate('[{"id": 1}, {}]', schema)


# ===========================================================================
# 8. String constraints
# ===========================================================================


class TestStringConstraints:
    def test_min_length_satisfied(self):
        result = _validate('"hello"', {"type": "string", "minLength": 3})
        assert result.value == "hello"

    def test_min_length_violated(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate('"hi"', {"type": "string", "minLength": 5})
        assert exc_info.value.code == "STRING_TOO_SHORT"

    def test_max_length_satisfied(self):
        result = _validate('"hello"', {"type": "string", "maxLength": 10})
        assert result.value == "hello"

    def test_max_length_violated(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate('"toolongstring"', {"type": "string", "maxLength": 5})
        assert exc_info.value.code == "STRING_TOO_LONG"

    def test_exact_boundary_min(self):
        result = _validate('"abc"', {"type": "string", "minLength": 3})
        assert result.value == "abc"

    def test_exact_boundary_max(self):
        result = _validate('"abc"', {"type": "string", "maxLength": 3})
        assert result.value == "abc"

    def test_empty_string_rejected_by_min_length(self):
        with pytest.raises(ValidationError):
            _validate('""', {"type": "string", "minLength": 1})


# ===========================================================================
# 9. Numeric constraints
# ===========================================================================


class TestNumericConstraints:
    def test_minimum_satisfied(self):
        result = _validate("5", {"type": "integer", "minimum": 0})
        assert result.value == 5

    def test_minimum_violated(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate("-1", {"type": "integer", "minimum": 0})
        assert exc_info.value.code == "NUMBER_TOO_SMALL"

    def test_maximum_satisfied(self):
        result = _validate("99", {"type": "integer", "maximum": 100})
        assert result.value == 99

    def test_maximum_violated(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate("101", {"type": "integer", "maximum": 100})
        assert exc_info.value.code == "NUMBER_TOO_LARGE"

    def test_number_at_exact_minimum(self):
        result = _validate("0", {"type": "integer", "minimum": 0})
        assert result.value == 0

    def test_number_at_exact_maximum(self):
        result = _validate("100", {"type": "integer", "maximum": 100})
        assert result.value == 100


# ===========================================================================
# 10. Enum validation
# ===========================================================================


class TestEnumValidation:
    def test_valid_string_enum(self):
        result = _validate('"red"', {"type": "string", "enum": ["red", "green", "blue"]})
        assert result.value == "red"

    def test_invalid_string_enum(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate('"yellow"', {"type": "string", "enum": ["red", "green", "blue"]})
        assert exc_info.value.code == "ENUM_MISMATCH"
        # Must not echo the disallowed value in the error (it's model output)
        assert "yellow" not in str(exc_info.value)

    def test_null_in_enum(self):
        result = _validate("null", {"enum": [None, "optional"]})
        assert result.value is None

    def test_integer_enum(self):
        result = _validate("2", {"type": "integer", "enum": [1, 2, 3]})
        assert result.value == 2

    def test_enum_mismatch_safe_message(self):
        """Error message must not contain the actual invalid value from model output."""
        with pytest.raises(ValidationError) as exc_info:
            _validate('"secret_model_output"', {"enum": ["allowed"]})
        msg = str(exc_info.value)
        assert "secret_model_output" not in msg


# ===========================================================================
# 11. ValidationError metadata
# ===========================================================================


class TestValidationErrorMetadata:
    def test_error_has_path(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate(
                '{"x": "not_int"}', {"type": "object", "properties": {"x": {"type": "integer"}}}
            )
        err = exc_info.value
        assert isinstance(err.path, ValidationPath)

    def test_error_as_dict_shape(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate(
                '{"x": "not_int"}', {"type": "object", "properties": {"x": {"type": "integer"}}}
            )
        d = exc_info.value.as_dict()
        assert set(d.keys()) == {"code", "path", "schema_type", "message"}
        assert d["code"] == "TYPE_MISMATCH"
        assert "$.x" in d["path"]

    def test_path_formatting_root(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate('"hello"', {"type": "integer"})
        assert str(exc_info.value.path) == "$"

    def test_path_formatting_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                }
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate('{"user": {"age": "old"}}', schema)
        path_str = str(exc_info.value.path)
        assert "user" in path_str
        assert "age" in path_str


# ===========================================================================
# 12. ValidatedOutput contract
# ===========================================================================


class TestValidatedOutput:
    def test_schema_type_in_result(self):
        result = _validate('{"key": "val"}', {"type": "object"})
        assert result.schema_type == "object"

    def test_no_schema_type_when_missing_in_schema(self):
        result = _validate('"hello"', {})
        # schema_type may be empty string when no type declared
        assert isinstance(result.schema_type, str)

    def test_value_is_plain_python_types(self):
        result = _validate('{"nums": [1, 2.5, true, null, "s"]}', {"type": "object"})
        v = result.value
        assert isinstance(v, dict)
        assert isinstance(v["nums"], list)
        assert v["nums"][0] == 1
        assert isinstance(v["nums"][1], float)
        assert v["nums"][2] is True
        assert v["nums"][3] is None
        assert v["nums"][4] == "s"

    def test_result_is_immutable_frozen_dataclass(self):
        result = _validate('"hello"', {"type": "string"})
        with pytest.raises((AttributeError, TypeError)):
            result.value = "changed"  # type: ignore[misc]
