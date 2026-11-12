"""Ollama LLM provider adapter."""

from __future__ import annotations

import json
from typing import Any

from flowforge.ai.providers.base import LLMProvider
from flowforge.ai.providers.exceptions import (
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

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider(LLMProvider):
    """Local-first Ollama LLM provider adapter."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        *,
        allow_custom_host: bool = False,
        transport: HTTPTransport | None = None,
    ) -> None:
        validated_url = validate_endpoint_url(
            base_url,
            allow_custom_host=allow_custom_host,
            default_loopback_only=True,
        )
        super().__init__(base_url=validated_url, api_key=None, transport=transport)

    def generate(self, request: LLMRequest) -> LLMResponse:
        endpoint = f"{self.base_url}/api/chat"

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }

        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

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

            if status_code != 200:
                raise ProviderResponseError(f"Ollama server returned HTTP {status_code}")

            try:
                data = json.loads(resp_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as err:
                raise ProviderResponseError("Invalid JSON response from Ollama") from err

            if not isinstance(data, dict):
                raise ProviderResponseError("Malformed response structure from Ollama")

            msg_obj = data.get("message")
            if not isinstance(msg_obj, dict) or "content" not in msg_obj:
                raise ProviderResponseError("Missing expected message output from Ollama response")

            text = msg_obj.get("content", "")
            model_name = data.get("model", request.model)

            prompt_tokens = data.get("prompt_eval_count")
            eval_tokens = data.get("eval_count")
            usage: TokenUsage | None = None
            if prompt_tokens is not None and eval_tokens is not None:
                usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=eval_tokens,
                    total_tokens=prompt_tokens + eval_tokens,
                )

            return LLMResponse(text=text, model=model_name, usage=usage)

        except ProviderError as err:
            raise self._sanitize_exception(err) from err
        except Exception as err:
            raise self._sanitize_exception(err) from err
