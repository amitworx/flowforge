"""Typed safety bounds and validation for AI inputs and outputs."""

from dataclasses import dataclass

from src.flowforge.ai.safety.exceptions import AIConfigurationError, AIPolicyLimitError


@dataclass(frozen=True)
class SafetyLimits:
    """Immutable validation limits for AI prompts and generation responses."""

    max_prompt_chars: int = 100_000
    max_prompt_bytes: int = 400_000
    max_response_bytes: int = 2_000_000
    max_output_tokens: int = 8_192
    max_workflow_ai_calls: int = 50
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_prompt_chars <= 0:
            raise AIConfigurationError("max_prompt_chars must be a strictly positive integer")
        if self.max_prompt_bytes <= 0:
            raise AIConfigurationError("max_prompt_bytes must be a strictly positive integer")
        if self.max_response_bytes <= 0:
            raise AIConfigurationError("max_response_bytes must be a strictly positive integer")
        if self.max_output_tokens <= 0:
            raise AIConfigurationError("max_output_tokens must be a strictly positive integer")
        if self.max_workflow_ai_calls <= 0:
            raise AIConfigurationError("max_workflow_ai_calls must be a strictly positive integer")
        if self.timeout_seconds <= 0.0:
            raise AIConfigurationError("timeout_seconds must be a strictly positive finite float")


def validate_prompt_input(prompt: str, limits: SafetyLimits) -> None:
    """Validate prompt string against character length and UTF-8 byte size limits."""
    if not isinstance(prompt, str):
        raise AIPolicyLimitError(
            "Prompt must be a string",
            limit_name="max_prompt_chars",
        )

    char_count = len(prompt)
    if char_count > limits.max_prompt_chars:
        raise AIPolicyLimitError(
            f"Prompt length ({char_count} chars) exceeds maximum allowed limit ({limits.max_prompt_chars} chars)",
            limit_name="max_prompt_chars",
            actual_value=char_count,
            limit_value=limits.max_prompt_chars,
        )

    byte_count = len(prompt.encode("utf-8"))
    if byte_count > limits.max_prompt_bytes:
        raise AIPolicyLimitError(
            f"Prompt UTF-8 size ({byte_count} bytes) exceeds maximum allowed limit ({limits.max_prompt_bytes} bytes)",
            limit_name="max_prompt_bytes",
            actual_value=byte_count,
            limit_value=limits.max_prompt_bytes,
        )


def validate_response_output(
    response_bytes: bytes, output_tokens: int, limits: SafetyLimits
) -> None:
    """Validate model response byte size and output tokens."""
    if not isinstance(response_bytes, (bytes, bytearray)):
        raise AIPolicyLimitError(
            "Response must be bytes",
            limit_name="max_response_bytes",
        )

    byte_len = len(response_bytes)
    if byte_len > limits.max_response_bytes:
        raise AIPolicyLimitError(
            f"Response size ({byte_len} bytes) exceeds maximum allowed limit ({limits.max_response_bytes} bytes)",
            limit_name="max_response_bytes",
            actual_value=byte_len,
            limit_value=limits.max_response_bytes,
        )

    if output_tokens > limits.max_output_tokens:
        raise AIPolicyLimitError(
            f"Output token count ({output_tokens}) exceeds maximum allowed limit ({limits.max_output_tokens})",
            limit_name="max_output_tokens",
            actual_value=output_tokens,
            limit_value=limits.max_output_tokens,
        )
