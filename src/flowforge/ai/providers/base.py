"""Base class and common utilities for LLM providers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from flowforge.ai.providers.exceptions import ProviderError
from flowforge.ai.providers.transport import HTTPTransport, StdlibHTTPTransport
from flowforge.ai.providers.types import LLMRequest, LLMResponse


def sanitize_error_message(msg: str, secrets: list[str] | None = None) -> str:
    """Sanitize error messages to avoid leaking API keys, credentials, or request data."""
    sanitized = str(msg)
    if secrets:
        for secret in secrets:
            if secret and len(secret) > 3:
                sanitized = sanitized.replace(secret, "[REDACTED]")

    # Redact common bearer token patterns
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED]", sanitized)
    sanitized = re.sub(r"sk-[A-Za-z0-9\-_]{8,}", "[REDACTED]", sanitized)
    # Redact basic URL user:password credentials if any slipped in
    sanitized = re.sub(r"://[^/]+:[^/@]+@", "://[REDACTED]:[REDACTED]@", sanitized)
    return sanitized


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        transport: HTTPTransport | None = None,
        *,
        allow_loopback: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport or StdlibHTTPTransport(allow_loopback=allow_loopback)

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from the provider."""
        ...

    def _sanitize_exception(self, exc: Exception) -> Exception:
        """Create a new sanitized exception, completely detaching the raw exception and cause."""
        secrets = [self.api_key] if self.api_key else []
        clean_msg = sanitize_error_message(str(exc), secrets=secrets)

        # Build a fresh exception of identical type if possible, or fallback to ProviderError
        exc_cls = type(exc) if issubclass(type(exc), ProviderError) else ProviderError
        new_exc = exc_cls(clean_msg)
        new_exc.__cause__ = None
        new_exc.__suppress_context__ = True
        return new_exc
