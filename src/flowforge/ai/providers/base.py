"""Base class and common utilities for LLM providers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from flowforge.ai.providers.exceptions import ProviderError
from flowforge.ai.providers.transport import HTTPTransport, StdlibHTTPTransport
from flowforge.ai.providers.types import LLMRequest, LLMResponse


def sanitize_error_message(msg: str, secrets: list[str] | None = None) -> str:
    """Sanitize error messages to avoid leaking API keys or request data."""
    sanitized = msg
    if secrets:
        for secret in secrets:
            if secret and len(secret) > 3:
                sanitized = sanitized.replace(secret, "[REDACTED]")

    # Redact common bearer token patterns
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED]", sanitized)
    sanitized = re.sub(r"sk-[A-Za-z0-9\-_]{8,}", "[REDACTED]", sanitized)
    return sanitized


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        transport: HTTPTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport or StdlibHTTPTransport()

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from the provider."""
        ...

    def _sanitize_exception(self, exc: Exception) -> Exception:
        """Wrap and sanitize exceptions to protect secrets and request payload."""
        secrets = [self.api_key] if self.api_key else []
        clean_msg = sanitize_error_message(str(exc), secrets=secrets)
        if isinstance(exc, ProviderError):
            exc.args = (clean_msg,)
            return exc
        return ProviderError(clean_msg)
