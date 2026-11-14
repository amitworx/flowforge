"""M0-015A-5 — Remaining Schema-Controlled Field Error Disclosure: RED tests.

Covers:
1. In `_validate_object`, missing required field errors must NOT leak raw or truncated
   caller-controlled property names into `ValidationError.args`, `str(err)`, `repr(err)`,
   or `as_dict().message`.
   - Error code must remain `MISSING_REQUIRED`.
   - Error message must use generic bounded metadata (e.g. "required property is missing"
     or "required property at index N is missing").
2. Secret-looking required property names (e.g. `SECRET_CUSTOM_TOKEN_ABCDE`), custom
   tokens, nested schema property names, type mismatches must NOT appear across `str`,
   `repr`, `args`, `path.parts`, or `as_dict`.
3. ValidationPath.enter and path parts for schema-defined keys:
   - Untrusted caller keys must never be blindly passed or leaked.
   - Any sensitive fragment must be absent from all exception representations.
"""

from __future__ import annotations

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


class TestSchemaControlledFieldErrorDisclosureRED:
    """RED tests asserting zero caller-supplied sensitive field names leak into exceptions."""

    def test_missing_required_field_does_not_leak_secret_key_name(self):
        """When a required property name is a secret token, missing-required error
        must NOT contain the secret token name on any error surface."""
        secret_required = "SECRET_CUSTOM_TOKEN_ABCDE"
        schema = {
            "type": "object",
            "required": [secret_required],
            "properties": {secret_required: {"type": "string"}},
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate("{}", schema)

        err = exc_info.value
        # Check all error representations
        assert secret_required not in str(err)
        assert secret_required not in repr(err)
        for arg in err.args:
            assert secret_required not in str(arg)
        dict_rep = err.as_dict()
        assert secret_required not in dict_rep["message"]
        for part in err.path.parts:
            assert secret_required not in str(part)
        assert err.code == "MISSING_REQUIRED"
        assert "required" in str(err).lower()

    def test_nested_missing_required_field_does_not_leak_secret_key_name(self):
        """Nested missing required property name must not leak."""
        secret_nested_required = "BEARER_AUTH_TOKEN_KEY_99999"
        schema = {
            "type": "object",
            "properties": {
                "safe_sub": {
                    "type": "object",
                    "required": [secret_nested_required],
                    "properties": {secret_nested_required: {"type": "integer"}},
                }
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate('{"safe_sub": {}}', schema)

        err = exc_info.value
        assert secret_nested_required not in str(err)
        assert secret_nested_required not in repr(err)
        assert secret_nested_required not in err.as_dict()["message"]
        for part in err.path.parts:
            assert secret_nested_required not in str(part)
        assert err.code == "MISSING_REQUIRED"

    def test_missing_required_error_has_generic_bounded_metadata(self):
        """The missing required field error message should be bounded and generic."""
        schema = {
            "type": "object",
            "required": ["field1"],
            "properties": {"field1": {"type": "string"}},
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate("{}", schema)

        err = exc_info.value
        assert err.code == "MISSING_REQUIRED"
        # Bounded generic description
        msg = str(err)
        assert "required" in msg.lower()
        assert "missing" in msg.lower()
        assert len(msg) < 200
