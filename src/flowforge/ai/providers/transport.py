"""Transport interface and DNS-pinned, SSRF-hardened stdlib HTTP implementation."""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import urllib.parse
from typing import Protocol

from flowforge.ai.providers.exceptions import (
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from flowforge.ai.providers.types import MAX_RESPONSE_BYTES
from flowforge.ai.providers.validation import is_ip_disallowed


class HTTPTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, bytes]:
        """Send HTTP request and return status code and response body bytes."""
        ...


def resolve_and_validate_pinned_ip(host: str, port: int, *, allow_loopback: bool = False) -> str:
    """Resolve hostname to IP address, validate against SSRF policy, and return pinned IP string."""
    clean_host = re.sub(r"^\[|\]$", "", host)

    # Check if host is already an IP address
    try:
        ip_obj = ipaddress.ip_address(clean_host)
        if allow_loopback and ip_obj.is_loopback:
            return clean_host
        if is_ip_disallowed(ip_obj):
            raise ProviderConnectionError(
                f"Destination IP {clean_host} is not permitted for network connection"
            )
        return clean_host
    except ValueError:
        pass

    try:
        resolved_addrs = socket.getaddrinfo(clean_host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ProviderConnectionError(f"DNS resolution failure for host '{host}'") from None

    if not resolved_addrs:
        raise ProviderConnectionError(f"No IP addresses resolved for host '{host}'")

    permitted_ips: list[str] = []
    for _family, _, _, _, sockaddr in resolved_addrs:
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        if allow_loopback and ip_obj.is_loopback:
            permitted_ips.append(ip_str)
            continue

        if is_ip_disallowed(ip_obj):
            raise ProviderConnectionError(
                f"Resolved destination IP {ip_str} for host '{host}' is not permitted"
            )

        permitted_ips.append(ip_str)

    if not permitted_ips:
        raise ProviderConnectionError(f"No valid IP addresses available for host '{host}'")

    # Return the first permitted IP for pinned socket connection
    return permitted_ips[0]


class PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection connecting directly to an already-resolved pinned IP."""

    def __init__(self, pinned_ip: str, host: str, port: int, timeout: float = 30.0) -> None:
        super().__init__(host=host, port=port, timeout=timeout)
        self.pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self.pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection connecting directly to pinned IP while preserving SNI & cert verification."""

    def __init__(
        self,
        pinned_ip: str,
        host: str,
        port: int,
        timeout: float = 30.0,
        context: ssl.SSLContext | None = None,
    ) -> None:
        super().__init__(host=host, port=port, timeout=timeout, context=context)
        self.pinned_ip = pinned_ip

    def connect(self) -> None:
        # Create raw TCP socket to the pinned IP
        raw_sock = socket.create_connection(
            (self.pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )

        server_hostname = self.host
        if self._context is None:
            self._context = ssl.create_default_context()

        # Wrap socket preserving SNI (server_hostname) and certificate verification against host
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=server_hostname)


class StdlibHTTPTransport:
    """Deterministic, zero-dependency, DNS-pinned, SSRF-hardened HTTP transport."""

    def __init__(self, *, allow_loopback: bool = False) -> None:
        self.allow_loopback = allow_loopback

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, bytes]:
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise ProviderConnectionError(f"Unsupported scheme: {scheme}")

        host = parsed.hostname or ""
        if not host:
            raise ProviderConnectionError("Missing hostname in request URL")

        port = parsed.port or (443 if scheme == "https" else 80)

        # 1. Resolve and validate IP, strictly pinning the destination IP
        pinned_ip = resolve_and_validate_pinned_ip(host, port, allow_loopback=self.allow_loopback)

        # 2. Build direct pinned connection
        req_headers = headers.copy() if headers else {}
        # Ensure Host header is preserved
        if "Host" not in req_headers:
            req_headers["Host"] = f"{host}:{port}" if parsed.port else host

        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        conn: http.client.HTTPConnection | http.client.HTTPSConnection
        if scheme == "https":
            conn = PinnedHTTPSConnection(
                pinned_ip=pinned_ip,
                host=host,
                port=port,
                timeout=timeout,
            )
        else:
            conn = PinnedHTTPConnection(
                pinned_ip=pinned_ip,
                host=host,
                port=port,
                timeout=timeout,
            )

        try:
            conn.request(
                method=method.upper(),
                url=path,
                body=body,
                headers=req_headers,
            )
            response = conn.getresponse()
            status_code = response.status

            # Deny redirects for security
            if 300 <= status_code < 400:
                raise ProviderResponseError("Redirects are disallowed for security")

            content = bytearray()
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > MAX_RESPONSE_BYTES:
                    raise ProviderResponseError(
                        f"Response size exceeds maximum allowed ({MAX_RESPONSE_BYTES} bytes)"
                    )

            return status_code, bytes(content)

        except TimeoutError:
            raise ProviderTimeoutError(f"HTTP request timed out after {timeout} seconds") from None
        except ProviderResponseError:
            raise
        except ProviderConnectionError:
            raise
        except Exception as err:
            err_msg = str(err).lower()
            if "timed out" in err_msg:
                raise ProviderTimeoutError(
                    f"HTTP request timed out after {timeout} seconds"
                ) from None
            raise ProviderConnectionError(f"HTTP connection failed: {err}") from None
        finally:
            conn.close()
