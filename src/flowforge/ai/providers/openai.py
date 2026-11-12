"""OpenAI-compatible LLM provider adapter."""

from __future__ import annotations

import json
from typing import Any

from flowforge.ai.providers.base import LLMProvider
from flowforge.ai.providers.exceptions import (
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
)
from flowforge.ai.providers.transport import HTTPTransport
from flowforge.ai.providers.types import (
    MAX_RESPONSE_BYTES,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from flowforge.ai.providers.validation import validate_endpoint_url


class OpenAICompatibleProvider(LLMProvider):
    """Generic OpenAI-compatible chat completions provider."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        *,
        transport: HTTPTransport | None = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ProviderConfigError("base_url is required for OpenAICompatibleProvider")

        if not api_key or not api_key.strip():
            raise ProviderAuthenticationError("api_key is required for OpenAICompatibleProvider")

        validated_url = validate_endpoint_url(base_url, allow_custom_host=True)
        super().__init__(base_url=validated_url, api_key=api_key, transport=transport)

    def generate(self, request: LLMRequest) -> LLMResponse:
        endpoint = f"{self.base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }

        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            status_code, resp_bytes = self.transport.request(
                method="POST",
                url=endpoint,
                headers=headers,
                body=body_bytes,
                timeout=request.timeout_seconds,
            )

            if len(resp_bytes) > MAX_RESPONSE_BYTES:
                raise ProviderResponseError(
                    f"Response size exceeds maximum allowed ({MAX_RESPONSE_BYTES} bytes)"
                )

            if status_code == 401 or status_code == 403:
                raise ProviderAuthenticationError(
                    f"Authentication failed with provider status {status_code}"
                )

            if status_code != 200:
                raise ProviderResponseError(f"Provider returned HTTP error {status_code}")

            try:
                data = json.loads(resp_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as err:
                raise ProviderResponseError("Invalid JSON response from provider") from err

            if not isinstance(data, dict):
                raise ProviderResponseError("Malformed response structure from provider")

            choices = data.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                raise ProviderResponseError("No choices returned in provider response")

            first_choice = choices[0]
            if not isinstance(first_choice, dict) or "message" not in first_choice:
                raise ProviderResponseError("Missing choice message in provider response")

            msg_obj = first_choice["message"]
            if not isinstance(msg_obj, dict) or "content" not in msg_obj:
                raise ProviderResponseError("Missing content in choice message")

            text = msg_obj.get("content", "")
            model_name = data.get("model", request.model)

            usage: TokenUsage | None = None
            usage_obj = data.get("usage")
            if isinstance(usage_obj, dict):
                usage = TokenUsage(
                    prompt_tokens=usage_obj.get("prompt_tokens", 0),
                    completion_tokens=usage_obj.get("completion_tokens", 0),
                    total_tokens=usage_obj.get("total_tokens", 0),
                )

            return LLMResponse(text=text, model=model_name, usage=usage)

        except ProviderError as err:
            raise self._sanitize_exception(err) from err
        except Exception as err:
            raise self._sanitize_exception(err) from err
