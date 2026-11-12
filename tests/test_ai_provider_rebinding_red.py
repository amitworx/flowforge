"""Tests for M0-013B DNS Pinning & Transport Hardening."""

from __future__ import annotations

import socket
import ssl
from unittest.mock import MagicMock, patch

import pytest

from flowforge.ai.providers import (
    ProviderConnectionError,
    ProviderResponseError,
    StdlibHTTPTransport,
)
from flowforge.ai.providers.transport import (
    PinnedHTTPConnection,
    PinnedHTTPSConnection,
    resolve_and_validate_pinned_ip,
)


class TestM013BDNSPinning:
    def test_pinned_ip_resolution_public_host(self):
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ]
            ip = resolve_and_validate_pinned_ip("example.com", 443, allow_loopback=False)
            assert ip == "93.184.216.34"

    def test_pinned_ip_resolution_rejects_private_and_rebinding(self):
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.100", 443))
            ]
            with pytest.raises(ProviderConnectionError, match="not permitted"):
                resolve_and_validate_pinned_ip("malicious.internal", 443, allow_loopback=False)

    def test_pinned_ip_resolution_loopback_allowed_for_ollama(self):
        ip = resolve_and_validate_pinned_ip("127.0.0.1", 11434, allow_loopback=True)
        assert ip == "127.0.0.1"

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434))
            ]
            ip_resolved = resolve_and_validate_pinned_ip("localhost", 11434, allow_loopback=True)
            assert ip_resolved == "127.0.0.1"

    def test_pinned_ip_resolution_loopback_denied_when_disallowed(self):
        with pytest.raises(ProviderConnectionError, match="not permitted"):
            resolve_and_validate_pinned_ip("127.0.0.1", 80, allow_loopback=False)

    def test_adversarial_dns_rebinding_never_connects_to_private_ip(self):
        """Adversarial scenario:

        DNS returns a public IP during resolution, but an attacker tries to make the socket
        connect to a re-resolved private IP. Because the socket connects directly to `pinned_ip`,
        the private address is NEVER reached.
        """
        public_ip = "93.184.216.34"
        transport = StdlibHTTPTransport(allow_loopback=False)

        connected_endpoints = []

        mock_socket_instance = MagicMock(spec=socket.socket)

        def mock_create_connection(address, timeout=None, source_address=None):  # noqa: ARG001
            connected_endpoints.append(address)
            return mock_socket_instance

        with (
            patch("socket.getaddrinfo") as mock_gai,
            patch("socket.create_connection", side_effect=mock_create_connection),
            patch.object(PinnedHTTPSConnection, "connect", autospec=True) as mock_conn,
        ):
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (public_ip, 443))]

            def fake_connect(inst):
                mock_create_connection((inst.pinned_ip, inst.port))
                mock_sock = MagicMock()
                inst.sock = mock_sock

            mock_conn.side_effect = fake_connect

            with patch.object(PinnedHTTPSConnection, "getresponse") as mock_resp:
                mock_http_resp = MagicMock()
                mock_http_resp.status = 200
                mock_http_resp.read.side_effect = [b'{"ok": true}', b""]
                mock_resp.return_value = mock_http_resp

                status, content = transport.request("GET", "https://api.openai.com/v1/models")
                assert status == 200
                assert content == b'{"ok": true}'

        # Verify that socket.create_connection was called with the PINNED IP, NOT the domain name
        assert len(connected_endpoints) == 1
        connected_ip, connected_port = connected_endpoints[0]
        assert connected_ip == public_ip
        assert connected_port == 443
        assert connected_ip != "api.openai.com"
        assert not connected_ip.startswith("10.")
        assert not connected_ip.startswith("192.168.")
        assert not connected_ip.startswith("127.")

    def test_preserves_sni_and_host_header(self):
        pinned_conn = PinnedHTTPSConnection(
            pinned_ip="93.184.216.34",
            host="api.openai.com",
            port=443,
        )

        with patch("socket.create_connection") as mock_conn:
            mock_raw_sock = MagicMock(spec=socket.socket)
            mock_conn.return_value = mock_raw_sock

            mock_context = MagicMock(spec=ssl.SSLContext)
            mock_ssl_sock = MagicMock(spec=ssl.SSLSocket)
            mock_context.wrap_socket.return_value = mock_ssl_sock
            pinned_conn._context = mock_context

            pinned_conn.connect()

            # Verify SNI server_hostname set to original domain name, not IP
            mock_context.wrap_socket.assert_called_once_with(
                mock_raw_sock,
                server_hostname="api.openai.com",
            )
            mock_conn.assert_called_once_with(
                ("93.184.216.34", 443),
                30.0,
                None,
            )
            assert pinned_conn.sock == mock_ssl_sock

    def test_redirects_refused(self):
        transport = StdlibHTTPTransport(allow_loopback=True)

        with patch("socket.create_connection") as mock_conn:
            mock_sock = MagicMock()
            mock_conn.return_value = mock_sock

            with patch.object(PinnedHTTPConnection, "getresponse") as mock_resp:
                mock_http_resp = MagicMock()
                mock_http_resp.status = 302
                mock_resp.return_value = mock_http_resp

                with pytest.raises(ProviderResponseError, match="Redirects are disallowed"):
                    transport.request("GET", "http://127.0.0.1:11434/api/chat")
