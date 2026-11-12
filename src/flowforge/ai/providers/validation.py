"""Endpoint validation and SSRF protection policy."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

from flowforge.ai.providers.exceptions import ProviderConfigError

FORBIDDEN_HOSTNAMES = {
    "169.254.169.254",  # AWS/GCP/Azure link-local instance metadata
    "metadata.google.internal",
    "instance-data",
    "localhost",
}


def is_ip_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if IP address is private, loopback, link-local, multicast, or reserved."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_endpoint_url(
    url: str,
    *,
    allow_custom_host: bool = False,
    default_loopback_only: bool = False,
) -> str:
    """Validate endpoint URL strictly against SSRF, dangerous IP ranges, and malformed targets."""
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

    # Cloud metadata check
    if hostname in FORBIDDEN_HOSTNAMES and hostname != "localhost":
        raise ProviderConfigError(f"Forbidden host '{hostname}' (cloud metadata or dangerous host)")

    # Check if host is raw IP or literal
    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        # Strip brackets for IPv6 if present
        clean_ip = re.sub(r"^\[|\]$", "", hostname)
        ip_obj = ipaddress.ip_address(clean_ip)
    except ValueError:
        ip_obj = None

    if default_loopback_only:
        # Loopback required (e.g. default Ollama)
        is_loopback = False
        if hostname == "localhost" or ip_obj and ip_obj.is_loopback:
            is_loopback = True

        if not is_loopback and not allow_custom_host:
            raise ProviderConfigError(
                f"Custom Ollama host requires explicit allow_custom_host=True (disallowed non-loopback '{hostname}')"
            )

        if not allow_custom_host and not is_loopback:
            raise ProviderConfigError(f"Disallowed non-loopback Ollama host '{hostname}'")

    else:
        # Hosted provider (OpenAI-compatible)
        # By default, hosted providers MUST NOT point to loopback, private, link-local or reserved IPs
        if ip_obj and is_ip_disallowed(ip_obj):
            raise ProviderConfigError(
                f"Disallowed private/internal IP address for hosted provider: {hostname}"
            )
        if hostname == "localhost":
            raise ProviderConfigError(
                "Disallowed private/internal IP address for hosted provider: localhost"
            )

    return url.rstrip("/")
