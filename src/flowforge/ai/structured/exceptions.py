"""Exceptions for the structured output validator.

All exception messages are safe to surface externally: they carry
only schema path, type details and size information — never raw model
output, prompt text, API keys or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationPath:
    """Represents a JSON path in a validated document."""

    parts: tuple[str | int, ...] = field(default_factory=tuple)

    @classmethod
    def root(cls) -> ValidationPath:
        return cls(parts=())

    def enter(self, key: str | int) -> ValidationPath:
        return ValidationPath(parts=(*self.parts, key))

    def __str__(self) -> str:
        if not self.parts:
            return "$"
        segments = []
        for part in self.parts:
            if isinstance(part, int):
                segments.append(f"[{part}]")
            else:
                segments.append(f".{part}")
        return "$" + "".join(segments)


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
