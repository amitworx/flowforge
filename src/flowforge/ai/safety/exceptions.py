"""Safe error categories and exceptions for pure AI execution safety."""


class AISafetyError(Exception):
    """Base exception for all AI safety errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def sanitized_message(self) -> str:
        """Return a message guaranteed not to leak secret tokens or API keys."""
        from src.flowforge.ai.safety.redaction import redact_secrets_from_text

        return redact_secrets_from_text(self.message)


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
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.limit_name = limit_name
        self.actual_value = actual_value
        self.limit_value = limit_value


class AIRetryableTransportError(AISafetyError):
    """Raised for transient transport/network failures eligible for retry (e.g. 429, 502, 503, 504)."""

    def __init__(
        self, message: str, *, status_code: int | None = None, details: dict | None = None
    ) -> None:
        super().__init__(message, details=details)
        self.status_code = status_code


class AINonRetryableProviderError(AISafetyError):
    """Raised when a provider returns a fatal or non-retryable response (e.g. 400, 401, 403, 404)."""

    def __init__(
        self, message: str, *, status_code: int | None = None, details: dict | None = None
    ) -> None:
        super().__init__(message, details=details)
        self.status_code = status_code


class AIMalformedResponseError(AISafetyError):
    """Raised when a model response is malformed, truncated, or unparseable."""

    pass
