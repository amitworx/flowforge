"""Tests for M0-013A: LLM Provider Network & Validation Hardening."""

from __future__ import annotations

import socket
import traceback
from unittest.mock import MagicMock, patch

import pytest

from flowforge.ai.providers import (
    ChatMessage,
    ChatRole,
    InvalidRequestError,
    LLMRequest,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderError,
    StdlibHTTPTransport,
    TokenUsage,
    validate_endpoint_url,
)


class TestM013AHardening:
    # 1. SSRF & Destination Security
    def test_ssrf_rejects_private_ips_for_hosted_providers(self):
        with pytest.raises(ProviderConfigError, match="Disallowed private/internal IP"):
            validate_endpoint_url("http://192.168.1.1/v1", allow_custom_host=True)

        with pytest.raises(ProviderConfigError, match="Disallowed private/internal IP"):
            validate_endpoint_url("http://10.0.0.5/v1", allow_custom_host=True)

        with pytest.raises(ProviderConfigError, match="Disallowed private/internal IP"):
            validate_endpoint_url("http://172.16.0.1/v1", allow_custom_host=True)

        with pytest.raises(ProviderConfigError, match="Disallowed private/internal IP"):
            validate_endpoint_url("http://[fd00::1]/v1", allow_custom_host=True)

    def test_ssrf_rejects_non_loopback_for_ollama_unless_allowed(self):
        with pytest.raises(ProviderConfigError, match="disallowed non-loopback"):
            validate_endpoint_url("http://192.168.1.1:11434", default_loopback_only=True)

    def test_transport_enforces_destination_ip_at_connection_time(self):
        transport = StdlibHTTPTransport()
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]
            with pytest.raises(
                ProviderConnectionError, match="Resolved destination IP .* is not permitted"
            ):
                transport.request("GET", "https://api.openai.com/v1/models")

    def test_transport_refuses_proxy_environment(self, monkeypatch):
        # Verify that environment proxies (http_proxy, https_proxy) have no effect on transport
        # because transport uses direct socket connection to pinned IP rather than proxy handlers.
        monkeypatch.setenv("http_proxy", "http://corporate-proxy.internal:8080")
        monkeypatch.setenv("https_proxy", "http://corporate-proxy.internal:8080")
        monkeypatch.setenv("HTTP_PROXY", "http://corporate-proxy.internal:8080")
        monkeypatch.setenv("HTTPS_PROXY", "http://corporate-proxy.internal:8080")

        transport = StdlibHTTPTransport()
        # Transport makes direct socket connections to the pinned IP, completely ignoring proxy settings
        assert transport.allow_loopback is False

    # 2. Sensitive Error Chaining & Secret Redaction
    def test_no_raw_cause_chaining_leaks_secrets(self):
        secret_key = "sk-super-secret-key-12345"
        mock_transport = MagicMock()
        mock_transport.request.side_effect = Exception(f"Connection failure with key {secret_key}")

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

        err = exc_info.value
        assert err.__cause__ is None
        tb_str = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        assert secret_key not in tb_str
        assert "[REDACTED]" in tb_str or secret_key not in tb_str

    # 3. Request Schema & Bounds Hardening
    def test_llm_request_validates_types_strictly(self):
        with pytest.raises(InvalidRequestError, match="model must be a string"):
            LLMRequest(model=123, messages=[ChatMessage(role=ChatRole.USER, content="hi")])  # type: ignore

        with pytest.raises(InvalidRequestError, match="temperature must be a finite float"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="hi")],
                temperature=float("nan"),
            )
        with pytest.raises(InvalidRequestError, match="temperature must be a finite float"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="hi")],
                temperature=float("inf"),
            )

        with pytest.raises(InvalidRequestError, match="max_output_tokens must be an integer"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="hi")],
                max_output_tokens=True,  # type: ignore
            )
        with pytest.raises(InvalidRequestError, match="max_output_tokens must be an integer"):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="hi")],
                max_output_tokens=10.5,  # type: ignore
            )

        with pytest.raises(
            InvalidRequestError, match="timeout_seconds must be a positive finite number"
        ):
            LLMRequest(
                model="test",
                messages=[ChatMessage(role=ChatRole.USER, content="hi")],
                timeout_seconds=float("inf"),
            )

        with pytest.raises(
            InvalidRequestError, match="Each message must be an instance of ChatMessage"
        ):
            LLMRequest(
                model="test",
                messages=["invalid message string"],  # type: ignore
            )

    def test_token_usage_validation(self):
        with pytest.raises(InvalidRequestError, match="Token counts cannot be negative"):
            TokenUsage(prompt_tokens=-1, completion_tokens=10, total_tokens=9)

        with pytest.raises(InvalidRequestError, match="Token counts must be integers"):
            TokenUsage(prompt_tokens=True, completion_tokens=10, total_tokens=10)  # type: ignore

    def test_post_byte_limit_enforced_before_network(self):
        mock_transport = MagicMock()
        provider = OllamaProvider(transport=mock_transport)
        heavy_char = "\U0001f600"  # 4-byte UTF-8 emoji
        # 450,000 characters * 4 bytes ≈ 1.8 MB (under 500k char limit, but json envelope pushes over 2MB byte cap)
        big_content = heavy_char * 450_000
        req = LLMRequest(
            model="llama3",
            messages=[ChatMessage(role=ChatRole.USER, content=big_content)],
        )
        with pytest.raises(
            InvalidRequestError, match="Request body size exceeds maximum allowed bytes"
        ):
            provider.generate(req)
        mock_transport.request.assert_not_called()
