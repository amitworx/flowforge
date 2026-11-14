"""M0-015A-4 — Schema-Owned Error Metadata Secrecy: RED tests.

Covers:
1. Schema-owned error text must NEVER interpolate untrusted caller-supplied
   raw strings, keys, or types into exception messages, args, repr, or dicts:
   - Unsupported schema keywords: `MY_UNKNOWN_SECRET_KEY_123` must not appear.
   - Unsupported schema types: `MY_SECRET_CUSTOM_TYPE` must not appear.
   - Unsupported union types: `["string", "MY_SECRET_UNION_TYPE"]` must not appear.
   - Non-string property keys or invalid property values: secret keys must not appear.
   - Non-string required field entries: secrets must not appear.
   - Missing required field: only safe schema identifiers, no secret fragments or raw injection.
2. Error messages must use safe, fixed, constant error codes and bounded generic descriptions:
   - e.g. "unsupported schema keyword detected"
   - e.g. "unsupported schema type detected"
"""

from __future__ import annotations

import pytest

from flowforge.ai.structured import (
    SchemaContract,
    SchemaError,
    ValidationError,
    validate_structured_output,
)


def _contract(schema: dict) -> SchemaContract:
    return SchemaContract.from_dict(schema)


def _validate(text: str, schema: dict):
    return validate_structured_output(text, _contract(schema))


class TestSchemaOwnedErrorMetadataSecrecyRED:
    """RED tests asserting no raw schema-controlled sensitive strings leak into exceptions."""

    def test_unsupported_schema_keyword_does_not_leak_raw_keyword_name(self):
        """When an unsupported keyword is present in schema, SchemaError must NOT
        echo the caller's raw keyword string if it could be a secret."""
        secret_keyword = "SECRET_CONFIG_KEY_ABC123"
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "object", secret_keyword: "foo"})

        err = exc_info.value
        assert secret_keyword not in str(err)
        assert secret_keyword not in repr(err)
        for arg in err.args:
            assert secret_keyword not in str(arg)

    def test_unsupported_schema_type_does_not_leak_raw_type_string(self):
        """When an unsupported type string is passed, SchemaError must NOT
        echo the raw string."""
        secret_type = "SECRET_CUSTOM_TYPE_XYZ456"
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": secret_type})

        err = exc_info.value
        assert secret_type not in str(err)
        assert secret_type not in repr(err)
        for arg in err.args:
            assert secret_type not in str(arg)

    def test_unsupported_union_type_does_not_leak_raw_type_string(self):
        """When an unsupported type name is in a union list, SchemaError must NOT
        echo the raw string."""
        secret_union_type = "SECRET_UNION_TYPE_999"
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": ["string", secret_union_type]})

        err = exc_info.value
        assert secret_union_type not in str(err)
        assert secret_union_type not in repr(err)
        for arg in err.args:
            assert secret_union_type not in str(arg)

    def test_non_string_required_entry_does_not_leak_repr(self):
        """When required contains a non-string object (e.g. dict or secret representation),
        SchemaError does not echo its contents."""
        secret_in_required = 9988776655
        with pytest.raises(SchemaError) as exc_info:
            _contract({"type": "object", "required": [secret_in_required]})

        err = exc_info.value
        assert str(secret_in_required) not in str(err)
        assert str(secret_in_required) not in repr(err)

    def test_nested_property_unsupported_keyword_does_not_leak(self):
        """Nested sub-schema with unsupported keyword does not leak raw keyword."""
        secret_nested_keyword = "NESTED_SECRET_HEADER_KEY"
        schema = {
            "type": "object",
            "properties": {
                "safe_field": {
                    "type": "object",
                    secret_nested_keyword: True,
                }
            },
        }
        with pytest.raises(SchemaError) as exc_info:
            _contract(schema)

        err = exc_info.value
        assert secret_nested_keyword not in str(err)
        assert secret_nested_keyword not in repr(err)
        for arg in err.args:
            assert secret_nested_keyword not in str(arg)

    def test_missing_required_field_does_not_leak_unbounded_field_name(self):
        """If schema defines a required field with malicious long/control chars,
        missing required field error must sanitize or bound it."""
        malicious_required = "req_" + "A" * 100 + "\x00\n"
        schema = {
            "type": "object",
            "required": [malicious_required],
            "properties": {malicious_required: {"type": "string"}},
        }
        with pytest.raises(ValidationError) as exc_info:
            _validate("{}", schema)

        err = exc_info.value
        err_str = str(err)
        assert "\x00" not in err_str
        assert "\n" not in err_str
        assert len(err_str) < 250
