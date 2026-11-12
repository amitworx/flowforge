"""Endpoint validation and SSRF protection policy."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from flowforge.ai.providers.exceptions import ProviderConfigError

FORBIDDEN_HOSTS = {
    "169.254.169.254",  # AWS/GCP/Azure link-local instance metadata
    "metadata.google.internal",
    "instance-data",
}


def validate_endpoint_url(
    url: str,
    *,
    allow_custom_host: bool = False,
    default_loopback_only: bool = False,
) -> str:
    """Validate endpoint URL strictly against SSRF and malformed targets."""
    if not url or not isinstance(url, str):
        raise ProviderConfigError("Endpoint URL must be a non-empty string")

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ProviderConfigError(
            f"Invalid scheme '{parsed.scheme}': only http and https are allowed"
        )

    if parsed.username or parsed.password:
        raise ProviderConfigError("URL credentials not allowed")

    if parsed.query:
        raise ProviderConfigError("URL queries are not allowed in base provider endpoint")

    if parsed.fragment:
        raise ProviderConfigError("URL fragments are not allowed in base provider endpoint")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ProviderConfigError("Invalid endpoint URL: missing host")

    if hostname in FORBIDDEN_HOSTS:
        raise ProviderConfigError(f"Forbidden host '{hostname}' (cloud metadata or dangerous host)")

    # If default_loopback_only is True (e.g. Ollama default), ensure it is loopback unless allow_custom_host is True
    if default_loopback_only:
        is_loopback = False
        if hostname in ("127.0.0.1", "localhost", "::1"):
            is_loopback = True
        else:
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_loopback:
                    is_loopback = True
            except ValueError:
                pass

        if not is_loopback and not allow_custom_host:
            raise ProviderConfigError(
                f"Custom Ollama host requires explicit allow_custom_host=True (received '{hostname}')"
            )

    return url.rstrip("/")
