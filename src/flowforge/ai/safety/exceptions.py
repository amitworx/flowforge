"""Safe error categories and exceptions for pure AI execution safety."""

from typing import Any


class AISafetyError(Exception):
    """Base exception for all AI safety errors.

    Sanitizes message, args, and details at construction time so that
    inspectors, serializers, logging frameworks, and __str__/__repr__
    never access unredacted secrets.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        from flowforge.ai.safety.redaction import redact_secrets_from_dict, redact_secrets_from_text

        sanitized_msg = redact_secrets_from_text(str(message))
        sanitized_det = redact_secrets_from_dict(details or {})

        super().__init__(sanitized_msg)
        self.message = sanitized_msg
        self.details = sanitized_det

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"

    def sanitized_message(self) -> str:
        """Return the sanitized message string."""
        return self.message

    def sanitized_details(self) -> dict[str, Any]:
        """Return details dictionary with sensitive keys and values redacted."""
        return self.details


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
