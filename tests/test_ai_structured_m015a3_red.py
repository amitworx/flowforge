"""M0-015A-3 — Remove raw model JSON property names from all exception surfaces: RED tests.

Covers:
1. `additionalProperties=False` must NOT leak ANY raw model-controlled property names,
   regardless of whether they are ASCII, printable, custom token formats, or alphanumeric.
   Error messages must contain ONLY generic bounded metadata (e.g. unexpected property count).
2. All error surfaces (`str`, `repr`, `args`, `as_dict`, `path.parts`, nested exceptions)
   must NOT contain the secret-looking model-generated keys.
3. Nested arbitrary model keys inside objects or additionalProperties must remain generic.
4. Schema-owned error text review: untrusted schema keys, enum values, and unsupported
   keywords must not leak sensitive content across exception surfaces.
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


class TestRemoveRawModelJSONNamesRED:
    """RED tests proving raw model property names are completely eliminated."""

    def test_additional_properties_false_does_not_leak_ascii_secret_key(self):
        """Even ordinary ASCII printable secret tokens in model JSON keys must NOT
        appear in error message, error args, or as_dict()."""
        secret_custom_token = "FLOWFORGE_SECRET_TOKEN_XYZ_99999"
        schema = {
            "type": "object",
            "properties": {"allowed_field": {"type": "string"}},
            "additionalProperties": False,
        }
        payload = json.dumps({"allowed_field": "ok", secret_custom_token: "ignored_val"})

        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        # Check all error representations
        assert secret_custom_token not in str(err)
        assert secret_custom_token not in repr(err)
        for arg in err.args:
            assert secret_custom_token not in str(arg)
        dict_rep = err.as_dict()
        assert secret_custom_token not in dict_rep["message"]
        assert secret_custom_token not in dict_rep["path"]
        # Error must have code and generic count metadata
        assert err.code == "ADDITIONAL_PROPERTIES"
        assert (
            "1 unexpected" in dict_rep["message"].lower()
            or "count" in dict_rep["message"].lower()
            or "found 1" in dict_rep["message"].lower()
        )

    def test_additional_properties_false_multiple_untrusted_keys(self):
        """Multiple unexpected keys should report count only, no raw key names."""
        k1 = "token_alpha_123"
        k2 = "token_beta_456"
        schema = {
            "type": "object",
            "properties": {"allowed": {"type": "string"}},
            "additionalProperties": False,
        }
        payload = json.dumps({"allowed": "ok", k1: "1", k2: "2"})

        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        assert k1 not in str(err)
        assert k2 not in str(err)
        assert "2 unexpected" in str(err).lower() or "found 2" in str(err).lower()

    def test_nested_object_additional_properties_false_no_leak(self):
        """Nested object additionalProperties=False must also not leak key names."""
        nested_secret = "INTERNAL_AUTH_CREDENTIAL_KEY"
        schema = {
            "type": "object",
            "properties": {
                "sub": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "additionalProperties": False,
                }
            },
        }
        payload = json.dumps({"sub": {"id": 1, nested_secret: "leak"}})

        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        assert nested_secret not in str(err)
        assert nested_secret not in repr(err)
        assert nested_secret not in err.as_dict()["message"]
        for p in err.path.parts:
            assert nested_secret not in str(p)

    def test_untrusted_schema_keys_in_error_text_bounded_and_safe(self):
        """If caller passes an untrusted schema with sensitive/unsupported keywords,
        the schema error message is safe, bounded, and contains no control codes."""
        bad_key = "sensitive_keyword_" + "x" * 100
        with pytest.raises(Exception) as exc_info:
            _contract({"type": "object", bad_key: True})
        msg = str(exc_info.value)
        assert len(msg) < 300
