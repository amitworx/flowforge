"""Public interface for FlowForge AI providers."""

from __future__ import annotations

from flowforge.ai.providers.base import LLMProvider, sanitize_error_message
from flowforge.ai.providers.exceptions import (
    InvalidRequestError,
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from flowforge.ai.providers.ollama import OllamaProvider
from flowforge.ai.providers.openai import OpenAICompatibleProvider
from flowforge.ai.providers.transport import HTTPTransport, RequestsHTTPTransport
from flowforge.ai.providers.types import (
    MAX_MESSAGES_COUNT,
    MAX_RESPONSE_BYTES,
    MAX_TOTAL_REQUEST_CHARS,
    ChatMessage,
    ChatRole,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from flowforge.ai.providers.validation import validate_endpoint_url

__all__ = [
    "ChatMessage",
    "ChatRole",
    "HTTPTransport",
    "InvalidRequestError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "MAX_MESSAGES_COUNT",
    "MAX_RESPONSE_BYTES",
    "MAX_TOTAL_REQUEST_CHARS",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "ProviderAuthenticationError",
    "ProviderConfigError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "RequestsHTTPTransport",
    "TokenUsage",
    "sanitize_error_message",
    "validate_endpoint_url",
]
