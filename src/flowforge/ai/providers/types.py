"""Data contracts for FlowForge AI providers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from flowforge.ai.providers.exceptions import InvalidRequestError

# Safety limits
MAX_MESSAGES_COUNT = 200
MAX_TOTAL_REQUEST_CHARS = 500_000
MAX_REQUEST_BYTES = 2 * 1024 * 1024  # 2 MB limit for serialized request body
MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB limit for response
MAX_MODEL_NAME_LEN = 128
MAX_OUTPUT_TOKENS_LIMIT = 131_072


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if isinstance(self.content, bool) or not isinstance(self.content, str):
            raise InvalidRequestError("Message content must be string")
        if not self.content.strip():
            raise InvalidRequestError("Message content cannot be empty")

        if isinstance(self.role, str) and not isinstance(self.role, ChatRole):
            try:
                object.__setattr__(self, "role", ChatRole(self.role))
            except ValueError:
                raise InvalidRequestError(f"Invalid message role: {self.role}") from None
        elif not isinstance(self.role, ChatRole):
            raise InvalidRequestError(f"Invalid message role: {self.role}")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise InvalidRequestError(f"Token counts must be integers ({field_name})")
            if val < 0:
                raise InvalidRequestError(f"Token counts cannot be negative ({field_name})")


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: Sequence[ChatMessage]
    temperature: float = 0.7
    max_output_tokens: int = 1000
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.model, bool) or not isinstance(self.model, str):
            raise InvalidRequestError("model must be a string")
        if not self.model.strip():
            raise InvalidRequestError("Model name cannot be empty")
        if len(self.model) > MAX_MODEL_NAME_LEN:
            raise InvalidRequestError(f"Model name exceeds maximum length of {MAX_MODEL_NAME_LEN}")

        if not isinstance(self.messages, Sequence) or isinstance(self.messages, (str, bytes)):
            raise InvalidRequestError("messages must be a sequence of ChatMessage objects")

        if len(self.messages) == 0:
            raise InvalidRequestError("Messages list cannot be empty")

        if len(self.messages) > MAX_MESSAGES_COUNT:
            raise InvalidRequestError(
                f"Number of messages exceeds maximum allowed ({MAX_MESSAGES_COUNT})"
            )

        # Validate each message and shallow copy to immutable tuple
        copied_messages = []
        total_chars = 0
        for idx, m in enumerate(self.messages):
            if not isinstance(m, ChatMessage):
                raise InvalidRequestError(
                    f"Each message must be an instance of ChatMessage (item at index {idx} is {type(m).__name__})"
                )
            copied_messages.append(m)
            total_chars += len(m.content)

        if total_chars > MAX_TOTAL_REQUEST_CHARS:
            raise InvalidRequestError(
                f"Total message characters ({total_chars}) exceeds maximum allowed ({MAX_TOTAL_REQUEST_CHARS})"
            )

        object.__setattr__(self, "messages", tuple(copied_messages))

        # Temperature validation
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise InvalidRequestError("temperature must be a finite float")
        temp_val = float(self.temperature)
        if not math.isfinite(temp_val):
            raise InvalidRequestError("temperature must be a finite float")
        if not (0.0 <= temp_val <= 2.0):
            raise InvalidRequestError("Temperature must be between 0.0 and 2.0")
        object.__setattr__(self, "temperature", temp_val)

        # max_output_tokens validation
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int):
            raise InvalidRequestError("max_output_tokens must be an integer")
        if self.max_output_tokens <= 0 or self.max_output_tokens > MAX_OUTPUT_TOKENS_LIMIT:
            raise InvalidRequestError(
                f"max_output_tokens must be positive and at most {MAX_OUTPUT_TOKENS_LIMIT}"
            )

        # timeout_seconds validation
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise InvalidRequestError("timeout_seconds must be a positive finite number")
        timeout_val = float(self.timeout_seconds)
        if not math.isfinite(timeout_val) or timeout_val <= 0:
            raise InvalidRequestError("timeout_seconds must be a positive finite number")
        object.__setattr__(self, "timeout_seconds", timeout_val)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: TokenUsage | None = None
