"""Custom exceptions for FlowForge AI providers."""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for all AI provider errors."""


class ProviderConfigError(ProviderError):
    """Raised when provider configuration or endpoint URL is invalid."""


class InvalidRequestError(ProviderError):
    """Raised when an LLMRequest or ChatMessage fails validation."""


class ProviderAuthenticationError(ProviderError):
    """Raised when authentication credentials are missing or invalid."""


class ProviderConnectionError(ProviderError):
    """Raised when a network connection to a provider fails."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an invalid status, malformed response, or payload error."""
