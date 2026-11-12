"""Deterministic log and exception redaction primitives."""

import re
from typing import Any

# Match standard API keys (e.g. OpenAI sk-..., sk-proj-..., anthropic sk-ant-..., generic api keys)
_API_KEY_REGEX = re.compile(
    r"\b(sk-[a-zA-Z0-9_-]{16,}|sk-proj-[a-zA-Z0-9_-]{16,}|sk-ant-[a-zA-Z0-9_-]{16,}|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36})\b",
    re.IGNORECASE,
)

# Match Authorization headers
_AUTH_HEADER_REGEX = re.compile(
    r"(Bearer\s+)[a-zA-Z0-9\._\-]+",
    re.IGNORECASE,
)

# Match credentials embedded in URLs: http(s)://user:password@host
_URL_CREDENTIALS_REGEX = re.compile(
    r"(https?://)([^:]+):([^@]+)@",
    re.IGNORECASE,
)

# Sensitive dictionary keys to redact automatically
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "secret",
    "token",
    "password",
    "passwd",
    "access_token",
    "refresh_token",
    "client_secret",
}


def redact_secrets_from_text(text: str) -> str:
    """Deterministically redact API keys, Bearer tokens, and URL credentials from text."""
    if not isinstance(text, str):
        return text

    # Redact Authorization: Bearer <token>
    redacted = _AUTH_HEADER_REGEX.sub(r"\1[REDACTED]", text)

    # Redact URL credentials: user:pass@host -> user:[REDACTED]@host
    redacted = _URL_CREDENTIALS_REGEX.sub(r"\1\2:[REDACTED]@", redacted)

    # Redact standard known API keys
    redacted = _API_KEY_REGEX.sub("[REDACTED]", redacted)

    return redacted


def redact_secrets_from_dict(data: Any) -> Any:
    """Recursively and deterministically redact sensitive values in structured data."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            lower_key = str(key).lower()
            if any(sensitive_word in lower_key for sensitive_word in _SENSITIVE_KEYS):
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_secrets_from_dict(value)
        return result
    elif isinstance(data, list):
        return [redact_secrets_from_dict(item) for item in data]
    elif isinstance(data, str):
        return redact_secrets_from_text(data)
    else:
        return data
