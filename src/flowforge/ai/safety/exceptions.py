"""Safe error categories and exceptions for pure AI execution safety."""

from typing import Any


class AISafetyError(Exception):
    """Base exception for all AI safety errors.

    String representation and representation methods are safe by default and
    automatically redact credentials and sensitive tokens.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = str(message)
        self.details = details or {}

    def __str__(self) -> str:
        return self.sanitized_message()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sanitized_message()!r})"

    def sanitized_message(self) -> str:
        """Return a message guaranteed not to leak secret tokens or API keys."""
        from flowforge.ai.safety.redaction import redact_secrets_from_text

        return redact_secrets_from_text(self.message)

    def sanitized_details(self) -> dict[str, Any]:
        """Return details dictionary with sensitive keys and values redacted."""
        from flowforge.ai.safety.redaction import redact_secrets_from_dict

        return redact_secrets_from_dict(self.details)


class AIConfigurationError(AISafetyError):
    """Raised when an invalid safety configuration or limit is provided."""

    pass


class AIPolicyLimitError(AISafetyError):
    """Raised when an input or output violates configured safety limits (fail-closed)."""

    def __init__(
        self,
        message: str,
        *,
        limit_name: str,
        actual_value: int | float | None = None,
        limit_value: int | float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.limit_name = limit_name
        self.actual_value = actual_value
        self.limit_value = limit_value


class AIRetryableTransportError(AISafetyError):
    """Raised for transient transport/network failures eligible for retry (e.g. 429, 502, 503, 504)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.status_code = status_code


class AINonRetryableProviderError(AISafetyError):
    """Raised when a provider returns a fatal or non-retryable response (e.g. 400, 401, 403, 404)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.status_code = status_code


class AIMalformedResponseError(AISafetyError):
    """Raised when a model response is malformed, truncated, or unparseable."""

    pass
