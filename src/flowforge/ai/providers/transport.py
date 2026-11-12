"""Transport interface and requests-based implementation."""

from __future__ import annotations

from typing import Protocol

import requests

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


class RequestsHTTPTransport:
    """Deterministic HTTP transport utilizing requests with streaming byte caps."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, bytes]:
        try:
            resp = self.session.request(
                method=method,
                url=url,
                headers=headers,
                data=body,
                timeout=timeout,
                allow_redirects=False,  # Refuse HTTP redirects to prevent SSRF hopping
                stream=True,
            )

            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                raise ProviderResponseError("Redirects are disallowed for security")

            content = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    content.extend(chunk)
                    if len(content) > MAX_RESPONSE_BYTES:
                        raise ProviderResponseError(
                            f"Response size exceeds maximum allowed ({MAX_RESPONSE_BYTES} bytes)"
                        )

            return resp.status_code, bytes(content)

        except requests.exceptions.Timeout as err:
            raise ProviderTimeoutError(f"HTTP request timed out after {timeout} seconds") from err
        except requests.exceptions.ConnectionError as err:
            raise ProviderConnectionError(f"HTTP connection error: {err}") from err
        except requests.exceptions.RequestException as err:
            raise ProviderResponseError(f"HTTP request failed: {err}") from err
