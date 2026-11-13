"""M0-015A-2 — Structured output error path sanitization: RED tests.

Covers the security requirements:
1. Dynamic model-controlled JSON key names must NEVER appear in error paths:
   - Nested depth error paths must not leak sensitive keys.
   - additionalProperties errors when validating against a schema must use
     safe generic placeholders (e.g. '[property]' or indexed property) rather than
     raw model-controlled key names.
   - additionalProperties: False must NOT print untrusted raw key names in the
     error message if they look like secrets/tokens or contain malicious text.
2. Safe schema-defined property names are preserved, but sanitized (bounded length,
   escaped control characters).
3. Inspection of error objects:
   - str(err), repr(err), err.args, err.as_dict(), err.path.parts, err.reason
   - Ensure none leak arbitrary raw model keys, secrets, bearer tokens, or
     unbounded Unicode strings.
4. Adversarial cases:
   - Malicious long Unicode names in JSON keys.
   - Password / bearer token / api-key looking JSON keys.
   - Depth limit exceeded inside malicious key.
   - Node limit exceeded with malicious keys.
   - additionalProperties invalid value error path.

NO database, network, or Django ORM calls.
"""

from __future__ import annotations

import json

import pytest

from flowforge.ai.structured import (
    MAX_NESTING_DEPTH,
    PayloadLimitError,
    SchemaContract,
    ValidationError,
    ValidationPath,
    validate_structured_output,
)


def _contract(schema: dict) -> SchemaContract:
    return SchemaContract.from_dict(schema)


def _validate(text: str, schema: dict):
    return validate_structured_output(text, _contract(schema))


class TestErrorPathSanitizationRED:
    """RED tests asserting strict path and error attribute sanitization."""

    def test_depth_limit_error_path_does_not_leak_dynamic_keys(self):
        """When depth limit is exceeded, path in PayloadLimitError must NOT echo
        untrusted model-supplied object keys."""
        secret_key = "Bearer_eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        # Create an object nested deeper than MAX_NESTING_DEPTH with secret keys
        obj = {"outer": 1}
        curr = obj
        for _ in range(MAX_NESTING_DEPTH + 5):
            curr[secret_key] = {}
            curr = curr[secret_key]

        payload = json.dumps(obj)
        with pytest.raises(PayloadLimitError) as exc_info:
            _validate(payload, {"type": "object"})

        err_str = str(exc_info.value)
        assert secret_key not in err_str
        assert "eyJhbGci" not in err_str
        for arg in exc_info.value.args:
            assert secret_key not in str(arg)

    def test_additional_properties_schema_validation_path_does_not_leak_dynamic_key(self):
        """When additionalProperties is a schema, validation errors on extra properties
        must use a safe placeholder like '[property]' or similar in the path rather
        than echoing untrusted model-supplied key names."""
        secret_key = "secret_password_123456"
        schema = {
            "type": "object",
            "properties": {"known_safe_prop": {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        }
        # Model returns the secret_key as an additional property, with invalid type (string)
        payload = json.dumps({"known_safe_prop": "ok", secret_key: "not_an_int"})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        # Path string and as_dict must NOT contain secret_key
        assert secret_key not in str(err)
        assert secret_key not in str(err.path)
        assert secret_key not in err.as_dict()["path"]
        assert secret_key not in err.as_dict()["message"]
        # Path should contain a safe generic placeholder like [property]
        assert "[property]" in str(err.path) or "property" in str(err.path).lower()

    def test_additional_properties_false_does_not_leak_untrusted_key_names(self):
        """When additionalProperties is False, error message must NOT echo raw
        untrusted key names directly if they are sensitive or unbounded."""
        secret_token = "sk-proj-9999999999999999999999999999"
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        payload = json.dumps({"name": "test", secret_token: "ignored_val"})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        err_msg = str(err)
        assert secret_token not in err_msg
        assert secret_token not in err.as_dict()["message"]
        assert err.code == "ADDITIONAL_PROPERTIES"

    def test_malicious_long_unicode_key_in_depth_error_sanitized(self):
        """Keys containing extreme Unicode, control characters, or excessive length
        must never appear unescaped or unbounded in any error representation."""
        malicious_key = "\x00\x1b[31m" + "🔥" * 200 + "\r\n"
        obj = {}
        curr = obj
        for _ in range(MAX_NESTING_DEPTH + 5):
            curr[malicious_key] = {}
            curr = curr[malicious_key]

        payload = json.dumps(obj)
        with pytest.raises(PayloadLimitError) as exc_info:
            _validate(payload, {"type": "object"})

        err_text = str(exc_info.value)
        assert "\x00" not in err_text
        assert "\x1b" not in err_text
        assert "🔥" * 10 not in err_text
        assert len(err_text) < 300

    def test_validation_path_parts_contain_no_raw_untrusted_keys(self):
        """ValidationPath.parts itself must not store raw untrusted dynamic keys."""
        secret_key = "sensitive_internal_claim_sub"
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "additionalProperties": {
                "type": "object",
                "properties": {"val": {"type": "integer"}},
            },
        }
        payload = json.dumps({secret_key: {"val": "should_be_int"}})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        # Check all parts of path tuple
        for part in err.path.parts:
            assert part != secret_key
            assert secret_key not in str(part)

    def test_safe_schema_defined_properties_are_preserved(self):
        """Safe property names defined in the schema SHOULD be preserved in path,
        allowing callers to locate schema violations."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {"age": {"type": "integer"}},
                        }
                    },
                }
            },
        }
        payload = json.dumps({"user": {"profile": {"age": "twenty"}}})
        with pytest.raises(ValidationError) as exc_info:
            _validate(payload, schema)

        err = exc_info.value
        assert str(err.path) == "$.user.profile.age"
        assert err.as_dict()["path"] == "$.user.profile.age"

    def test_schema_defined_property_with_control_chars_is_sanitized_in_path(self):
        """Even if a schema property name has special/control chars, path formatting
        must safely sanitize and bound it."""
        path = ValidationPath.root().enter("user\nname\x00").enter(0)
        formatted = str(path)
        assert "\n" not in formatted
        assert "\x00" not in formatted
