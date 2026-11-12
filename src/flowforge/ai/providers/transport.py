"""Transport interface and hardened stdlib urllib implementation."""

from __future__ import annotations

import io
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
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


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects for security to prevent SSRF bypasses."""

    def redirect_request(
        self,
        req: urllib.request.Request,  # noqa: ARG002
        fp: io.BufferedIOBase,  # noqa: ARG002
        code: int,  # noqa: ARG002
        msg: str,  # noqa: ARG002
        hdrs: urllib.response.addinfourl,  # noqa: ARG002
        newurl: str,  # noqa: ARG002
    ) -> None:
        raise ProviderResponseError("Redirects are disallowed for security")


def _validate_destination_ip(host: str, port: int, allow_loopback: bool = False) -> None:
    """Resolve destination hostname and ensure resolved IP is permitted."""
    clean_host = re.sub(r"^\[|\]$", "", host)
    try:
        resolved_addrs = socket.getaddrinfo(clean_host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ProviderConnectionError(f"DNS resolution failure for host '{host}'") from None

    if not resolved_addrs:
        raise ProviderConnectionError(f"No IP addresses resolved for host '{host}'")

    for _family, _, _, _, sockaddr in resolved_addrs:
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        if allow_loopback and ip_obj.is_loopback:
            continue

        if is_ip_disallowed(ip_obj):
            raise ProviderConnectionError(
                f"Resolved destination IP {ip_str} is not permitted for network connection"
            )


class StdlibHTTPTransport:
    """Deterministic, zero-dependency, SSRF-hardened HTTP transport utilizing Python stdlib urllib."""

    def __init__(self, *, allow_loopback: bool = False) -> None:
        self.allow_loopback = allow_loopback
        # Explicit ProxyHandler({}) ignores environment proxies (http_proxy/https_proxy) to prevent proxy bypasses
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

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
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        # Enforce destination validation at connection time (DNS resolved IPs)
        _validate_destination_ip(host, port, allow_loopback=self.allow_loopback)

        req_headers = headers.copy() if headers else {}
        req = urllib.request.Request(
            url=url,
            data=body,
            headers=req_headers,
            method=method.upper(),
        )

        try:
            with self.opener.open(req, timeout=timeout) as response:
                status_code = response.status
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

        except urllib.error.HTTPError as err:
            status_code = err.code
            try:
                error_body = err.read(65536)
            except Exception:
                error_body = b""
            return status_code, error_body

        except urllib.error.URLError as err:
            reason = str(err.reason) if hasattr(err, "reason") else str(err)
            if "timed out" in reason.lower():
                raise ProviderTimeoutError(
                    f"HTTP request timed out after {timeout} seconds"
                ) from None
            raise ProviderConnectionError("HTTP connection failed") from None

        except TimeoutError:
            raise ProviderTimeoutError(f"HTTP request timed out after {timeout} seconds") from None

        except ProviderResponseError:
            raise

        except Exception:
            raise ProviderResponseError("HTTP request failed") from None
