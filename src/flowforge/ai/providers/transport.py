"""Transport interface and minimal stdlib urllib-based implementation."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from typing import Protocol

from flowforge.ai.providers.exceptions import (
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from flowforge.ai.providers.types import MAX_RESPONSE_BYTES


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


class StdlibHTTPTransport:
    """Deterministic, zero-dependency HTTP transport utilizing Python stdlib urllib."""

    def __init__(self) -> None:
        self.opener = urllib.request.build_opener(_NoRedirectHandler)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, bytes]:
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
                ) from err
            raise ProviderConnectionError(f"HTTP connection error: {reason}") from err

        except TimeoutError as err:
            raise ProviderTimeoutError(f"HTTP request timed out after {timeout} seconds") from err

        except ProviderResponseError:
            raise

        except Exception as err:
            raise ProviderResponseError(f"HTTP request failed: {err}") from err
