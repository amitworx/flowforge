"""Exceptions for the structured output validator.

All exception messages are safe to surface externally: they carry
only schema path, type details and size information — never raw model
output, prompt text, API keys or credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Maximum length for an individual path segment in string representation
MAX_PATH_SEGMENT_LEN: int = 40
# Maximum total formatted path string length
MAX_FORMATTED_PATH_LEN: int = 200

# Pattern to check if a schema-defined property key is safe to display as-is
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")


def _sanitize_segment(segment: str | int) -> str:
    """Format and sanitize a single path segment for safe string representation.

    - Integers are formatted as [i].
    - Safe identifiers are formatted as .prop.
    - Control characters, extreme unicode, or long/unsafe strings are sanitized
      to a safe, bounded representation without leaking arbitrary model contents.
    """
    if isinstance(segment, int):
        return f"[{segment}]"

    seg_str = str(segment)
    if _SAFE_IDENTIFIER_RE.match(seg_str):
        return f".{seg_str}"

    # Generic or dynamic placeholders
    if seg_str.startswith("[") and seg_str.endswith("]"):
        safe_inner = re.sub(r"[^\w\-]", "", seg_str[1:-1])[:20]
        return f"[{safe_inner}]"

    # For other strings, escape non-printable and truncate
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", seg_str)
    clean = re.sub(r"[^\w\-.]", "_", clean)[:MAX_PATH_SEGMENT_LEN]
    if not clean:
        clean = "property"
    return f".{clean}"


@dataclass(frozen=True)
class ValidationPath:
    """Represents a JSON path in a validated document.

    Only schema-defined properties or safe generic placeholders should be entered.
    Arbitrary model-controlled keys must use safe placeholders (like '[property]').
    """

    parts: tuple[str | int, ...] = field(default_factory=tuple)

    @classmethod
    def root(cls) -> ValidationPath:
        return cls(parts=())

    def enter(self, key: str | int, *, is_safe: bool = True) -> ValidationPath:
        """Enter a path segment.

        If `is_safe` is False, the key is replaced with the generic placeholder
        `'[property]'` so untrusted model key names are never stored or leaked.
        """
        if not is_safe:
            part: str | int = "[property]"
        else:
            if isinstance(key, int):
                part = key
            else:
                clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", str(key))[:MAX_PATH_SEGMENT_LEN]
                part = clean if clean else "[property]"
        return ValidationPath(parts=(*self.parts, part))

    def __str__(self) -> str:
        if not self.parts:
            return "$"
        segments = [_sanitize_segment(p) for p in self.parts]
        formatted = "$" + "".join(segments)
        if len(formatted) > MAX_FORMATTED_PATH_LEN:
            return formatted[: MAX_FORMATTED_PATH_LEN - 3] + "..."
        return formatted


class StructuredOutputError(ValueError):
    """Base class for all structured output validation errors.

    Safe to log externally: contains only schema metadata and path information,
    never raw model output, prompt content, API keys, or credentials.
    """

    pass


class ValidationError(StructuredOutputError):
    """Raised when model output fails to conform to the declared schema contract.

    Attributes:
        path: The JSON path at which validation failed.
        message: A safe, human-readable description of the failure.
        schema_type: The expected schema type or keyword that caused failure.
        code: Machine-readable error code for deterministic handling.
    """

    def __init__(
        self,
        message: str,
        *,
        path: ValidationPath | None = None,
        schema_type: str = "",
        code: str = "",
    ) -> None:
        self.path = path or ValidationPath.root()
        self.schema_type = schema_type
        self.code = code
        super().__init__(message)

    def __str__(self) -> str:
        prefix = f"[{self.code}] " if self.code else ""
        path_str = str(self.path)
        return f"{prefix}At {path_str}: {super().__str__()}"

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable, deterministically-ordered summary of this error."""
        return {
            "code": self.code,
            "path": str(self.path),
            "schema_type": self.schema_type,
            "message": super().__str__(),
        }


class ParseError(StructuredOutputError):
    """Raised when model output cannot be parsed as valid JSON.

    Carries only a safe reason string — never the raw model text.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"JSON parse failed: {reason}")


class PayloadLimitError(StructuredOutputError):
    """Raised when model output exceeds configured size or depth limits."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Payload limit exceeded: {reason}")


class SchemaError(StructuredOutputError):
    """Raised when the caller-supplied schema is invalid or uses unsupported keywords."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid schema: {reason}")
