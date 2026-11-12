"""Data contracts for FlowForge AI providers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from flowforge.ai.providers.exceptions import InvalidRequestError

# Safety limits
MAX_MESSAGES_COUNT = 200
MAX_TOTAL_REQUEST_CHARS = 500_000
MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB limit


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise InvalidRequestError("Message content must be string")
        if not self.content.strip():
            raise InvalidRequestError("Message content cannot be empty")

        if isinstance(self.role, str) and not isinstance(self.role, ChatRole):
            try:
                object.__setattr__(self, "role", ChatRole(self.role))
            except ValueError as err:
                raise InvalidRequestError(f"Invalid message role: {self.role}") from err
        elif not isinstance(self.role, ChatRole):
            raise InvalidRequestError(f"Invalid message role: {self.role}")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: Sequence[ChatMessage]
    temperature: float = 0.7
    max_output_tokens: int = 1000
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.model or not self.model.strip():
            raise InvalidRequestError("Model name cannot be empty")

        if not self.messages or len(self.messages) == 0:
            raise InvalidRequestError("Messages list cannot be empty")

        if len(self.messages) > MAX_MESSAGES_COUNT:
            raise InvalidRequestError(
                f"Number of messages exceeds maximum allowed ({MAX_MESSAGES_COUNT})"
            )

        total_chars = sum(len(m.content) for m in self.messages)
        if total_chars > MAX_TOTAL_REQUEST_CHARS:
            raise InvalidRequestError(
                f"Total message characters ({total_chars}) exceeds maximum allowed ({MAX_TOTAL_REQUEST_CHARS})"
            )

        if not (0.0 <= self.temperature <= 2.0):
            raise InvalidRequestError("Temperature must be between 0.0 and 2.0")

        if self.max_output_tokens <= 0:
            raise InvalidRequestError("max_output_tokens must be positive")

        if self.timeout_seconds <= 0:
            raise InvalidRequestError("timeout_seconds must be positive")

        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "temperature", float(self.temperature))


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: TokenUsage | None = None
