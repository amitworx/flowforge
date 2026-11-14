"""M0-015A-6 — Nested Schema Path Secrecy: RED tests.

Covers the security requirements:
1. For all schema property-key traversal in `_validate_object`, use a generic opaque
   stable path segment (fixed placeholder '[property]') instead of raw/truncated
   property names in `ValidationPath.enter`.
2. Prevent any caller-supplied schema property name from leaking into `path.parts`,
   `str(ValidationError)`, `repr(ValidationError)`, or `as_dict().path`.
3. Keep numeric array indices for array traversal and maintain stable error codes.
4. Ensure the error path implementation's public `is_safe=True` default cannot
   accidentally be used for unsafe field names within validator.
5. Adversarial cases:
   - Present nested secret-looking schema property names (e.g.
     `properties: {"SECRET_CUSTOM_TOKEN_ABCDE": {"type": "integer"}}` with invalid value).
   - Deeply nested objects with secret property names.
   - Array of objects containing secret property names (verifying numeric index is kept).
   - Long ASCII identifiers and Unicode identifiers looking like ordinary schema keys.
   - additionalProperties schema nested with secret property names.
   - Verify that NO raw key appears on any error surface (`str`, `repr`, `args`,
     `path.parts`, `as_dict`).

NO database, network, or Django ORM calls.
"""

from __future__ import annotations

import json

import pytest

from flowforge.ai.structured import (
    SchemaContract,
    ValidationError,
    validate_structured_output,
)


def _contract(schema: dict) -> SchemaContract:
    return SchemaContract.from_dict(schema)


def _validate(text: str, schema: dict):
    return validate_structured_output(text, _contract(schema))


class TestNestedSchemaPathSecrecyRED:
    """RED tests asserting zero caller-supplied schema property names leak in error paths."""

    def test_nested_secret_property_name_does_not_leak_on_type_mismatch(self):
        """When an object property has an invalid type, the schema property name
        must NOT appear in path.parts, str(err), repr(err), or as_dict()."""
        secret_prop = "SECRET_CUSTOM_TOKEN_ABCDE"
        schema = {
            "type": "object",
            "properties": {secret_prop: {"type": "integer"}},
        }
        payload = json.dumps({secret_prop: "not_an_int"})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        # Stable error code
        assert err.code == "TYPE_MISMATCH"

        # Secret must NOT appear on any error surface
        assert secret_prop not in str(err)
        assert secret_prop not in repr(err)
        for arg in err.args:
            assert secret_prop not in str(arg)

        # Path parts must NOT contain the secret
        for part in err.path.parts:
            assert part != secret_prop
            assert secret_prop not in str(part)

        # as_dict must NOT contain the secret
        dict_rep = err.as_dict()
        assert secret_prop not in dict_rep["path"]
        assert secret_prop not in dict_rep["message"]

        # Path must use generic placeholder '[property]'
        assert str(err.path) == "$[property]"
        assert err.path.parts == ("[property]",)

    def test_deeply_nested_secret_property_names_do_not_leak(self):
        """Nested schema property names must be replaced with generic opaque
        segments at each level of nesting."""
        secret_outer = "SECRET_LAYER_ONE_ALPHA"
        secret_inner = "SECRET_LAYER_TWO_BRAVO"
        schema = {
            "type": "object",
            "properties": {
                secret_outer: {
                    "type": "object",
                    "properties": {
                        secret_inner: {"type": "integer"},
                    },
                }
            },
        }
        payload = json.dumps({secret_outer: {secret_inner: "string_value"}})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        assert err.code == "TYPE_MISMATCH"

        # Neither secret may appear anywhere
        for secret in (secret_outer, secret_inner):
            assert secret not in str(err)
            assert secret not in repr(err)
            assert secret not in err.as_dict()["path"]
            assert secret not in err.as_dict()["message"]
            for part in err.path.parts:
                assert secret not in str(part)

        # Path must be generic $[property][property]
        assert str(err.path) == "$[property][property]"
        assert err.path.parts == ("[property]", "[property]")

    def test_array_of_objects_secret_property_keeps_numeric_index(self):
        """Array traversal must keep numeric array indices while replacing object
        properties with generic placeholders."""
        secret_prop = "SECRET_ITEM_FIELD_XYZ"
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    secret_prop: {"type": "integer"},
                },
            },
        }
        payload = json.dumps(
            [
                {secret_prop: 100},
                {secret_prop: "invalid_type"},
            ]
        )
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        assert err.code == "TYPE_MISMATCH"

        # Secret must not appear
        assert secret_prop not in str(err)
        assert secret_prop not in repr(err)
        assert secret_prop not in err.as_dict()["path"]
        for part in err.path.parts:
            assert secret_prop not in str(part)

        # Path must preserve numeric index 1 and use generic [property]
        assert str(err.path) == "$[1][property]"
        assert err.path.parts == (1, "[property]")
        assert isinstance(err.path.parts[0], int)
        assert err.path.parts[0] == 1

    def test_nested_object_inside_array_inside_object_secret_keys(self):
        """Complex nesting object -> array -> object must preserve numeric index
        and sanitize all property names."""
        secret_container = "SECRET_CONTAINER_TOK"
        secret_field = "SECRET_NESTED_FIELD_VAL"
        schema = {
            "type": "object",
            "properties": {
                secret_container: {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            secret_field: {"type": "string", "minLength": 5},
                        },
                    },
                }
            },
        }
        payload = json.dumps({secret_container: [{secret_field: "ab"}]})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        assert err.code == "STRING_TOO_SHORT"

        for secret in (secret_container, secret_field):
            assert secret not in str(err)
            assert secret not in repr(err)
            assert secret not in err.as_dict()["path"]
            assert secret not in err.as_dict()["message"]
            for part in err.path.parts:
                assert secret not in str(part)

        # Path must be $[property][0][property]
        assert str(err.path) == "$[property][0][property]"
        assert err.path.parts == ("[property]", 0, "[property]")

    def test_long_ascii_and_unicode_property_names_do_not_leak(self):
        """Long ASCII and Unicode schema property names must never leak on error surfaces."""
        long_ascii_prop = "secret_audit_token_" + "x" * 50
        unicode_prop = "секретный_токен_идентификатор"

        schema = {
            "type": "object",
            "properties": {
                long_ascii_prop: {"type": "number"},
                unicode_prop: {"type": "boolean"},
            },
        }

        # Test long ASCII prop failure
        payload1 = json.dumps({long_ascii_prop: "not_a_number"})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload1, schema)

        err1 = exc_info.value
        assert "secret_audit_token" not in str(err1)
        assert "secret_audit_token" not in repr(err1)
        assert "secret_audit_token" not in err1.as_dict()["path"]
        assert err1.path.parts == ("[property]",)

        # Test Unicode prop failure
        payload2 = json.dumps({unicode_prop: "not_a_bool"})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload2, schema)

        err2 = exc_info.value
        assert "секретный" not in str(err2)
        assert "секретный" not in repr(err2)
        assert "секретный" not in err2.as_dict()["path"]
        assert err2.path.parts == ("[property]",)

    def test_additional_properties_schema_nested_with_secret_keys(self):
        """When additionalProperties is a schema with properties, invalid inner properties
        must not leak schema key names or dynamic key names."""
        secret_dynamic_key = "MODEL_GENERATED_SECRET_KEY_123"
        secret_schema_prop = "SECRET_INNER_SCHEMA_PROP_456"
        schema = {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    secret_schema_prop: {"type": "integer"},
                },
            },
        }
        payload = json.dumps(
            {
                secret_dynamic_key: {secret_schema_prop: "not_int"},
            }
        )
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        assert err.code == "TYPE_MISMATCH"

        for secret in (secret_dynamic_key, secret_schema_prop):
            assert secret not in str(err)
            assert secret not in repr(err)
            assert secret not in err.as_dict()["path"]
            assert secret not in err.as_dict()["message"]
            for part in err.path.parts:
                assert secret not in str(part)

        assert str(err.path) == "$[property][property]"
        assert err.path.parts == ("[property]", "[property]")
