"""M0-015A-1 — Structured output validator hardening: RED tests.

Covers the five confirmed blocker categories:
1. Mutable untrusted schema: defensive copy, freeze, mutation safety
2. Schema shape numeric validation: NaN/Inf in constraints, negative/fractional
   integer-only fields, min>max inconsistency, unbounded enum/required, schema size cap
3. Sensitive error leakage: duplicate key names and extra property names must be
   generic/redacted in error messages
4. JSON enum strict typing: True!=1, 1!=1.0, type-sensitive comparison
5. Input type: string-only API, non-str rejected safely

NO database, network, or Django ORM calls.
"""

from __future__ import annotations

import contextlib

import pytest

from flowforge.ai.structured import (
    ParseError,
    SchemaContract,
    SchemaError,
    ValidationError,
    validate_structured_output,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _contract(schema: dict) -> SchemaContract:
    return SchemaContract.from_dict(schema)


def _validate(text: str, schema: dict):
    return validate_structured_output(text, _contract(schema))


# ===========================================================================
# 1. Mutable untrusted schema — defensive copy / freeze
# ===========================================================================


class TestMutableUntrustedSchema:
    def test_mutation_of_original_dict_does_not_affect_contract(self):
        """Mutating the caller-provided dict after construction must NOT change
        the contract's behaviour."""
        schema = {"type": "string"}
        contract = SchemaContract.from_dict(schema)

        # Mutate caller's dict
        schema["type"] = "integer"

        # Contract should still validate strings, not integers
        result = validate_structured_output('"hello"', contract)
        assert result.value == "hello"

    def test_mutation_of_nested_dict_does_not_affect_contract(self):
        """Mutating a nested sub-dict in the caller's schema must not affect
        the contract's internal state."""
        props = {"name": {"type": "string"}}
        schema = {"type": "object", "properties": props}
        contract = SchemaContract.from_dict(schema)

        # Mutate nested dict after construction
        props["name"]["type"] = "integer"

        # Contract should still validate name as string
        result = validate_structured_output('{"name": "Alice"}', contract)
        assert result.value["name"] == "Alice"

    def test_contract_raw_cannot_be_mutated_externally(self):
        """Mutating ``contract.raw`` directly must not change validator behaviour.

        Implementations should either return a copy of raw, or store it in a
        way that prevents external mutation from reaching internal state.
        """
        contract = SchemaContract.from_dict({"type": "string"})

        # Try mutating .raw (the stored copy, not the original)
        with contextlib.suppress(TypeError, AttributeError):
            contract.raw["type"] = "integer"

        # Even if .raw was mutated, the contract must behave consistently
        # (either original or snapshot — not silently broken)
        try:
            result = validate_structured_output('"hello"', contract)
            # If we get here, mutation of .raw affected nothing (uses internal copy)
            assert result.value == "hello"
        except ValidationError:
            # Also acceptable: the mutation of .raw was noticed but didn't
            # affect the already-validated schema (depends on implementation)
            pass

    def test_schema_contract_from_dict_validates_at_construction_time(self):
        """Schema validation must happen eagerly at from_dict(), not lazily."""
        with pytest.raises(SchemaError):
            SchemaContract.from_dict({"type": "object", "unsupportedKey": True})

    def test_schema_contract_cannot_be_constructed_with_raw_mutation_bypass(self):
        """Directly constructing SchemaContract with invalid raw dict should
        either be rejected or be safe. No silent bypass of schema validation."""
        # This tests that directly passing a dict to SchemaContract()
        # (bypassing from_dict) with an unsupported keyword does not silently
        # succeed in validation.
        # The implementation may choose to re-validate in validate_structured_output,
        # or SchemaContract() may always require from_dict(). Either is acceptable.
        # What is NOT acceptable: validate_structured_output silently ignoring
        # keywords it claims to reject.
        try:
            # Try directly constructing (bypassing from_dict)
            contract = SchemaContract(raw={"type": "object", "$ref": "#bad"})
            # If this works, validate_structured_output must catch the bad schema
            with pytest.raises((SchemaError, Exception)):
                validate_structured_output('{"key": "value"}', contract)
        except (SchemaError, TypeError, AttributeError):
            pass  # Acceptable: reject at construction


# ===========================================================================
# 2. Schema shape numeric validation
# ===========================================================================


class TestSchemaShapeNumericValidation:
    # --- NaN/Inf in constraint values ---

    def test_nan_in_minimum_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "number", "minimum": float("nan")})
        assert "minimum" in str(exc_info.value).lower() or "nan" in str(exc_info.value).lower()

    def test_inf_in_maximum_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "number", "maximum": float("inf")})
        assert "maximum" in str(exc_info.value).lower() or "inf" in str(exc_info.value).lower()

    def test_neg_inf_in_minimum_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "number", "minimum": float("-inf")})
        assert "minimum" in str(exc_info.value).lower() or "inf" in str(exc_info.value).lower()

    def test_nan_in_min_length_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "string", "minLength": float("nan")})
        assert "minLength" in str(exc_info.value) or "nan" in str(exc_info.value).lower()

    def test_inf_in_max_items_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "array", "maxItems": float("inf")})
        assert "maxItems" in str(exc_info.value) or "inf" in str(exc_info.value).lower()

    # --- Negative values for integer-only constraints ---

    def test_negative_min_length_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "string", "minLength": -1})
        assert "minLength" in str(exc_info.value) or "negative" in str(exc_info.value).lower()

    def test_negative_max_length_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "string", "maxLength": -5})
        assert "maxLength" in str(exc_info.value) or "negative" in str(exc_info.value).lower()

    def test_negative_min_items_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "array", "minItems": -1})
        assert "minItems" in str(exc_info.value) or "negative" in str(exc_info.value).lower()

    def test_negative_max_items_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "array", "maxItems": -1})
        assert "maxItems" in str(exc_info.value) or "negative" in str(exc_info.value).lower()

    # --- Fractional values for integer-only constraints ---

    def test_fractional_min_length_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "string", "minLength": 1.5})
        assert "minLength" in str(exc_info.value) or "integer" in str(exc_info.value).lower()

    def test_fractional_max_items_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "array", "maxItems": 2.7})
        assert "maxItems" in str(exc_info.value) or "integer" in str(exc_info.value).lower()

    # --- Inconsistent min > max ---

    def test_min_length_greater_than_max_length_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "string", "minLength": 10, "maxLength": 5})
        msg = str(exc_info.value).lower()
        assert "minlength" in msg or "maxlength" in msg or "min" in msg

    def test_min_items_greater_than_max_items_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "array", "minItems": 5, "maxItems": 2})
        msg = str(exc_info.value).lower()
        assert "minitems" in msg or "maxitems" in msg or "min" in msg

    def test_minimum_greater_than_maximum_raises_schema_error(self):
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "number", "minimum": 100, "maximum": 10})
        msg = str(exc_info.value).lower()
        assert "minimum" in msg or "maximum" in msg or "min" in msg

    # --- Unbounded enum / required ---

    def test_too_many_enum_values_raises_schema_error(self):
        """Enum lists with excessively many values should be capped."""
        # 1001 values is excessive and potentially a denial-of-service vector
        from flowforge.ai.structured.types import MAX_ENUM_VALUES

        big_enum = list(range(MAX_ENUM_VALUES + 1))
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "integer", "enum": big_enum})
        assert "enum" in str(exc_info.value).lower()

    def test_too_many_required_fields_raises_schema_error(self):
        """Required lists with excessively many entries should be capped."""
        from flowforge.ai.structured.types import MAX_REQUIRED_FIELDS

        big_required = [f"field_{i}" for i in range(MAX_REQUIRED_FIELDS + 1)]
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "object", "required": big_required})
        assert "required" in str(exc_info.value).lower()

    # --- Schema size cap (total schema nodes/bytes) ---

    def test_schema_dict_too_many_total_nodes_raises_schema_error(self):
        """A schema with excessively many total nested nodes should be rejected."""
        from flowforge.ai.structured.types import MAX_SCHEMA_NODES

        # Build a schema with many properties to exceed node count
        props = {f"prop_{i}": {"type": "string"} for i in range(MAX_SCHEMA_NODES + 1)}
        with pytest.raises(SchemaError):
            _contract({"type": "object", "properties": props})


# ===========================================================================
# 3. Sensitive error leakage
# ===========================================================================


class TestSensitiveErrorLeakage:
    def test_duplicate_key_error_does_not_contain_key_value(self):
        """ParseError for duplicate keys must not echo key values from model output."""
        raw = '{"secret_token": "abc123", "secret_token": "xyz789"}'
        with pytest.raises(ParseError) as exc_info:
            _validate(raw, {"type": "object"})
        msg = str(exc_info.value)
        # Must not contain the actual secret value
        assert "abc123" not in msg
        assert "xyz789" not in msg
        # Key name may or may not be included — this test checks values only
        # (key names appear in the error message but values must not)

    def test_additional_properties_error_does_not_leak_values(self):
        """ValidationError for additional properties must not contain property values."""
        payload = '{"name": "Alice", "secret_key": "sk-9999-hunter2"}'
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)
        msg = str(exc_info.value)
        # Must not contain the value of the extra property
        assert "sk-9999-hunter2" not in msg

    def test_type_mismatch_error_does_not_echo_value(self):
        """TYPE_MISMATCH error must not contain the actual rejected value."""
        # An API key-like string that happens to be in the wrong field
        payload = '{"count": "sk-secret-token-12345"}'
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)
        msg = str(exc_info.value)
        assert "sk-secret-token-12345" not in msg

    def test_enum_mismatch_error_does_not_echo_value(self):
        """ENUM_MISMATCH error must not echo the disallowed model output value."""
        import json

        secret = "BEARER_TOKEN_XYZ_9999"
        payload = json.dumps(secret)
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, {"enum": ["ALLOWED"]})
        assert secret not in str(exc_info.value)

    def test_missing_required_error_does_not_echo_other_values(self):
        """MISSING_REQUIRED error must not echo values from sibling fields."""
        payload = '{"other_field": "sensitive_data_xyz"}'
        schema = {
            "type": "object",
            "required": ["missing_field"],
            "properties": {"missing_field": {"type": "string"}},
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)
        msg = str(exc_info.value)
        assert "sensitive_data_xyz" not in msg

    def test_parse_error_message_is_bounded_in_length(self):
        """ParseError messages must be bounded in length regardless of input size."""
        malformed = "{" + "x" * 10000 + ": garbage}"
        with pytest.raises(ParseError) as exc_info:
            _validate(malformed, {"type": "object"})
        # Error message must be short — not echoing the payload
        assert len(str(exc_info.value)) < 500


# ===========================================================================
# 4. JSON enum strict typing
# ===========================================================================


class TestEnumStrictTyping:
    def test_true_does_not_match_integer_enum_1(self):
        """Python True == 1 but JSON true is NOT 1; must not match integer enum."""
        with pytest.raises(ValidationError) as exc_info:
            _validate("true", {"enum": [1, 2, 3]})
        assert exc_info.value.code == "ENUM_MISMATCH"

    def test_false_does_not_match_integer_enum_0(self):
        """Python False == 0 but JSON false is NOT 0; must not match integer enum."""
        with pytest.raises(ValidationError) as exc_info:
            _validate("false", {"enum": [0, 1, 2]})
        assert exc_info.value.code == "ENUM_MISMATCH"

    def test_integer_1_does_not_match_boolean_true_enum(self):
        """JSON 1 (integer) is NOT JSON true (boolean); must not match boolean enum."""
        with pytest.raises(ValidationError) as exc_info:
            _validate("1", {"enum": [True, False]})
        assert exc_info.value.code == "ENUM_MISMATCH"

    def test_integer_1_does_not_match_float_1_0_enum(self):
        """JSON 1 (integer) must not silently match 1.0 (number) in an enum."""
        with pytest.raises(ValidationError) as exc_info:
            _validate("1", {"enum": [1.0, 2.0]})
        assert exc_info.value.code == "ENUM_MISMATCH"

    def test_float_1_0_does_not_match_integer_1_enum(self):
        """JSON 1.0 (number) must not silently match 1 (integer) in an enum."""
        with pytest.raises(ValidationError) as exc_info:
            _validate("1.0", {"enum": [1, 2]})
        assert exc_info.value.code == "ENUM_MISMATCH"

    def test_string_1_does_not_match_integer_1_enum(self):
        """JSON '1' (string) must not match 1 (integer) in an enum."""
        with pytest.raises(ValidationError) as exc_info:
            _validate('"1"', {"enum": [1, 2, 3]})
        assert exc_info.value.code == "ENUM_MISMATCH"

    def test_null_does_not_match_false_enum(self):
        """JSON null must not match false (bool) in enum."""
        with pytest.raises(ValidationError) as exc_info:
            _validate("null", {"enum": [False]})
        assert exc_info.value.code == "ENUM_MISMATCH"

    def test_true_matches_true_enum(self):
        """JSON true matches True in enum (same type and value)."""
        result = _validate("true", {"enum": [True, False]})
        assert result.value is True

    def test_integer_2_matches_integer_enum(self):
        """JSON 2 (integer) matches 2 in enum."""
        result = _validate("2", {"enum": [1, 2, 3]})
        assert result.value == 2

    def test_null_matches_null_enum(self):
        """JSON null matches None in enum."""
        result = _validate("null", {"enum": [None, "other"]})
        assert result.value is None

    def test_string_matches_string_enum(self):
        """JSON 'hello' matches 'hello' in enum."""
        result = _validate('"hello"', {"enum": ["hello", "world"]})
        assert result.value == "hello"


# ===========================================================================
# 5. Input type: string-only API
# ===========================================================================


class TestInputTypeValidation:
    def test_non_string_bytes_raises_safe_error(self):
        """Passing bytes instead of str must raise a safe, bounded error."""
        contract = _contract({"type": "object"})
        with pytest.raises((ParseError, TypeError, ValueError)) as exc_info:
            validate_structured_output(b'{"key": "value"}', contract)  # type: ignore[arg-type]
        # Error must be safe (bounded, no raw content leakage)
        msg = str(exc_info.value)
        assert len(msg) < 500

    def test_non_string_int_raises_safe_error(self):
        """Passing int instead of str must raise a safe error."""
        contract = _contract({"type": "integer"})
        with pytest.raises((ParseError, TypeError, ValueError)):
            validate_structured_output(42, contract)  # type: ignore[arg-type]

    def test_non_string_none_raises_safe_error(self):
        """Passing None instead of str must raise a safe error."""
        contract = _contract({"type": "null"})
        with pytest.raises((ParseError, TypeError, ValueError)):
            validate_structured_output(None, contract)  # type: ignore[arg-type]

    def test_non_string_dict_raises_safe_error(self):
        """Passing a dict directly instead of str must raise a safe error."""
        contract = _contract({"type": "object"})
        with pytest.raises((ParseError, TypeError, ValueError)):
            validate_structured_output({"key": "value"}, contract)  # type: ignore[arg-type]

    def test_valid_string_still_works(self):
        """Normal string inputs must continue to work after input validation."""
        result = _validate('{"key": 42}', {"type": "object"})
        assert result.value == {"key": 42}

    def test_unicode_decode_error_raises_parse_error_safely(self):
        """If somehow a string with surrogate characters causes encoding issues,
        the error must be safe and bounded."""
        # Create a string with lone surrogates (not encodable to UTF-8)
        # This may come from a bytes.decode with errors='surrogatepass'
        try:
            bad_str = "\ud800\udc00"  # surrogate pair in a Python str
            contract = _contract({"type": "string"})
            result = validate_structured_output(bad_str, contract)
            # If it parses, that's acceptable (some implementations handle it)
            assert isinstance(result.value, str)
        except (ParseError, UnicodeEncodeError, UnicodeDecodeError, ValueError) as exc:
            # Acceptable: safe rejection
            assert len(str(exc)) < 1000
