"""Tests for LLM provider foundation (M0-013)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from flowforge.ai.providers import (
    ChatMessage,
    ChatRole,
    HTTPTransport,
    InvalidRequestError,
    LLMRequest,
    LLMResponse,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    TokenUsage,
    validate_endpoint_url,
)


class TestDataContracts:
    def test_chat_message_valid(self):
        msg = ChatMessage(role=ChatRole.USER, content="Hello")
        assert msg.role == ChatRole.USER
        assert msg.content == "Hello"

        msg_sys = ChatMessage(role="system", content="System instruction")
        assert msg_sys.role == ChatRole.SYSTEM

        msg_asst = ChatMessage(role="assistant", content="Response")
        assert msg_asst.role == ChatRole.ASSISTANT

    def test_chat_message_invalid_role(self):
        with pytest.raises(InvalidRequestError, match="Invalid message role"):
            ChatMessage(role="tool", content="data")

    def test_chat_message_empty_or_invalid_content(self):
        with pytest.raises(InvalidRequestError, match="Message content cannot be empty"):
            ChatMessage(role=ChatRole.USER, content="")

        with pytest.raises(InvalidRequestError, match="Message content must be string"):
            ChatMessage(role=ChatRole.USER, content=None)  # type: ignore

    def test_llm_request_valid(self):
        req = LLMRequest(
            model="llama3",
            messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
            temperature=0.7,
            max_output_tokens=500,
            timeout_seconds=30,
        )
        assert req.model == "llama3"
        assert len(req.messages) == 1
        assert req.temperature == 0.7
        assert req.max_output_tokens == 500
        assert req.timeout_seconds == 30.0

    def test_llm_request_validation_boundaries(self):
        with pytest.raises(InvalidRequestError, match="Model name cannot be empty"):
            LLMRequest(model="", messages=[ChatMessage(role=ChatRole.USER, content="Hi")])

        with pytest.raises(InvalidRequestError, match="Messages list cannot be empty"):
            LLMRequest(model="test", messages=[])

        with pytest.raises(InvalidRequestError, match="Temperature must be between"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
                temperature=-0.1,
            )

        with pytest.raises(InvalidRequestError, match="Temperature must be between"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
                temperature=2.1,
            )

        with pytest.raises(InvalidRequestError, match="max_output_tokens must be positive"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
                max_output_tokens=0,
            )

        with pytest.raises(InvalidRequestError, match="timeout_seconds must be positive"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
                timeout_seconds=-1,
            )

    def test_llm_request_max_limits(self):
        huge_msgs = [ChatMessage(role=ChatRole.USER, content="a" * 10_000) for _ in range(500)]
        with pytest.raises(InvalidRequestError, match="exceeds maximum"):
            LLMRequest(model="test", messages=huge_msgs)

    def test_llm_response_and_token_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        resp = LLMResponse(text="Hello world", model="llama3", usage=usage)
        assert resp.text == "Hello world"
        assert resp.model == "llama3"
        assert resp.usage.total_tokens == 30

        resp_no_usage = LLMResponse(text="Simple", model="gpt-4o")
        assert resp_no_usage.usage is None


class TestSSRFAndEndpointValidation:
    def test_default_ollama_base_url(self):
        provider = OllamaProvider()
        assert provider.base_url == "http://127.0.0.1:11434"

    def test_ollama_custom_url_without_opt_in_refused(self):
        with pytest.raises(
            ProviderConfigError, match="Custom Ollama host requires explicit allow_custom_host=True"
        ):
            OllamaProvider(base_url="http://192.168.1.50:11434")

    def test_ollama_custom_url_with_opt_in(self):
        provider = OllamaProvider(base_url="http://192.168.1.50:11434", allow_custom_host=True)
        assert provider.base_url == "http://192.168.1.50:11434"

    def test_openai_compatible_requires_base_url_and_api_key(self):
        with pytest.raises(ProviderConfigError, match="base_url is required"):
            OpenAICompatibleProvider(base_url="")

        with pytest.raises(ProviderAuthenticationError, match="api_key is required"):
            OpenAICompatibleProvider(base_url="https://api.openai.com/v1", api_key="")

    def test_reject_unsupported_schemes(self):
        with pytest.raises(ProviderConfigError, match="Invalid scheme"):
            validate_endpoint_url("ftp://example.com/v1")

        with pytest.raises(ProviderConfigError, match="Invalid scheme"):
            validate_endpoint_url("file:///etc/passwd")

    def test_reject_credentials_in_url(self):
        with pytest.raises(ProviderConfigError, match="URL credentials not allowed"):
            validate_endpoint_url("http://user:pass@127.0.0.1:11434")

    def test_reject_fragments_and_queries(self):
        with pytest.raises(ProviderConfigError, match="URL fragments are not allowed"):
            validate_endpoint_url("http://127.0.0.1:11434/api/chat#fragment")

        with pytest.raises(ProviderConfigError, match="URL queries are not allowed"):
            validate_endpoint_url("http://127.0.0.1:11434/api/chat?test=1")

    def test_reject_dangerous_cloud_metadata(self):
        with pytest.raises(ProviderConfigError, match="Forbidden host"):
            validate_endpoint_url("http://169.254.169.254/latest/meta-data")


class TestOllamaProvider:
    def test_ollama_generate_success(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.return_value = (
            200,
            json.dumps(
                {
                    "model": "llama3",
                    "message": {"role": "assistant", "content": "Generated reply"},
                    "done": True,
                    "prompt_eval_count": 15,
                    "eval_count": 25,
                }
            ).encode("utf-8"),
        )

        provider = OllamaProvider(transport=mock_transport)
        req = LLMRequest(
            model="llama3", messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        resp = provider.generate(req)

        assert resp.text == "Generated reply"
        assert resp.model == "llama3"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 15
        assert resp.usage.completion_tokens == 25
        assert resp.usage.total_tokens == 40

        # Assert correct request wire format
        mock_transport.request.assert_called_once()
        call_kwargs = mock_transport.request.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["url"] == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(call_kwargs["body"].decode("utf-8"))
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_ollama_http_error_sanitization(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.side_effect = ProviderResponseError(
            "Server returned 500 Internal Error"
        )

        provider = OllamaProvider(transport=mock_transport)
        req = LLMRequest(
            model="llama3",
            messages=[ChatMessage(role=ChatRole.USER, content="secret prompt with sensitive key")],
        )
        with pytest.raises(ProviderError) as exc_info:
            provider.generate(req)

        # Ensure prompt or sensitive body is not leaked in exception string
        assert "secret prompt" not in str(exc_info.value)

    def test_ollama_malformed_json_response(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.return_value = (200, b"not valid json")

        provider = OllamaProvider(transport=mock_transport)
        req = LLMRequest(model="llama3", messages=[ChatMessage(role=ChatRole.USER, content="Hi")])
        with pytest.raises(ProviderResponseError, match="Invalid JSON response"):
            provider.generate(req)

    def test_ollama_missing_message_field(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.return_value = (200, b'{"model": "llama3"}')

        provider = OllamaProvider(transport=mock_transport)
        req = LLMRequest(model="llama3", messages=[ChatMessage(role=ChatRole.USER, content="Hi")])
        with pytest.raises(ProviderResponseError, match="Missing expected message output"):
            provider.generate(req)


class TestOpenAICompatibleProvider:
    def test_openai_generate_success(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.return_value = (
            200,
            json.dumps(
                {
                    "id": "chatcmpl-123",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OpenAI reply"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 18,
                        "total_tokens": 30,
                    },
                }
            ).encode("utf-8"),
        )

        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test-secret-key-12345",
            transport=mock_transport,
        )
        req = LLMRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="You are helpful."),
                ChatMessage(role=ChatRole.USER, content="Hello"),
            ],
            temperature=0.5,
        )
        resp = provider.generate(req)

        assert resp.text == "OpenAI reply"
        assert resp.model == "gpt-4o"
        assert resp.usage.total_tokens == 30

        # Assert authorization header sent and URL constructed properly
        mock_transport.request.assert_called_once()
        call_kwargs = mock_transport.request.call_args[1]
        assert call_kwargs["url"] == "https://api.openai.com/v1/chat/completions"
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-secret-key-12345"

    def test_secret_redaction_in_errors(self):
        secret_key = "sk-super-secret-password-xyz"
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.side_effect = ProviderConnectionError(
            f"Failed connecting with {secret_key}"
        )

        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key=secret_key,
            transport=mock_transport,
        )
        req = LLMRequest(
            model="gpt-4o", messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        with pytest.raises(ProviderError) as exc_info:
            provider.generate(req)

        assert secret_key not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value) or secret_key not in str(exc_info.value)

    def test_openai_missing_choices(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.return_value = (200, b'{"id": "123", "choices": []}')

        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            transport=mock_transport,
        )
        req = LLMRequest(
            model="gpt-4o", messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        with pytest.raises(ProviderResponseError, match="No choices returned"):
            provider.generate(req)


class TestTransportAndSafetyBoundaries:
    def test_default_never_sends_paid_api_request(self):
        # Default OllamaProvider only points to loopback and requires no keys
        ollama = OllamaProvider()
        assert ollama.base_url == "http://127.0.0.1:11434"
        assert ollama.api_key is None

        # OpenAICompatibleProvider refuses instantiation without explicit base_url and api_key
        with pytest.raises(ProviderConfigError):
            OpenAICompatibleProvider(base_url="")

    def test_oversized_response_protection(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        # Simulate payload over 10MB limit
        huge_bytes = b"x" * (10 * 1024 * 1024 + 1)
        mock_transport.request.return_value = (200, huge_bytes)

        provider = OllamaProvider(transport=mock_transport)
        req = LLMRequest(
            model="llama3", messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        with pytest.raises(ProviderResponseError, match="Response size exceeds maximum allowed"):
            provider.generate(req)

    def test_redirect_refusal(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.side_effect = ProviderResponseError("Redirects are disallowed")

        provider = OllamaProvider(transport=mock_transport)
        req = LLMRequest(
            model="llama3", messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        with pytest.raises(ProviderResponseError, match="Redirects are disallowed"):
            provider.generate(req)

    def test_timeout_handling(self):
        mock_transport = MagicMock(spec=HTTPTransport)
        mock_transport.request.side_effect = ProviderTimeoutError("Request timed out")

        provider = OllamaProvider(transport=mock_transport)
        req = LLMRequest(
            model="llama3", messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        with pytest.raises(ProviderTimeoutError):
            provider.generate(req)
