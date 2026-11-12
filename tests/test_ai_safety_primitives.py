"""Tests for pure AI execution safety contracts (M0-018A).

Tests cover:
- Typed validation limits (prompt characters, UTF-8 bytes, response bytes, output tokens, workflow call count, timeout).
- Boundary and off-by-one cases, Unicode byte counting.
- Safe error categories and exceptions (fail-closed, sanitized, no credential leakage).
- Deterministic log and metadata redaction (API keys, Bearer headers, URL credentials, structured metadata).
- Deterministic retry decision policy (idempotent operations only, status code evaluation, bounded exponential backoff with deterministic jitter).
- In-memory request budget tracker (single execution isolation, concurrency safe, no global mutable state, strict budget exhaustion).
"""

import pytest

from src.flowforge.ai.safety.budget import ExecutionBudgetTracker
from src.flowforge.ai.safety.exceptions import (
    AIConfigurationError,
    AIPolicyLimitError,
)
from src.flowforge.ai.safety.limits import (
    SafetyLimits,
    validate_prompt_input,
    validate_response_output,
)
from src.flowforge.ai.safety.redaction import redact_secrets_from_dict, redact_secrets_from_text
from src.flowforge.ai.safety.retry import RetryPolicy, evaluate_retry


def test_safety_limits_default_and_boundary_checks():
    limits = SafetyLimits(
        max_prompt_chars=10,
        max_prompt_bytes=20,
        max_response_bytes=50,
        max_output_tokens=100,
        max_workflow_ai_calls=5,
        timeout_seconds=30.0,
    )

    # Valid exact boundary
    validate_prompt_input("1234567890", limits)  # 10 chars, 10 bytes

    # Off-by-one char violation
    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_prompt_input("12345678901", limits)
    assert "Prompt length (11 chars) exceeds maximum allowed limit (10 chars)" in str(
        exc_info.value
    )
    assert exc_info.value.limit_name == "max_prompt_chars"

    # Unicode byte violation (each emoji is 4 bytes: 3 * 4 = 12 bytes <= 20, but 6 * 4 = 24 > 20)
    validate_prompt_input("🔒🔒🔒", limits)  # 3 chars, 12 bytes -> valid
    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_prompt_input("🔒🔒🔒🔒🔒🔒", limits)  # 6 chars (<=10), 24 bytes (>20)
    assert "Prompt UTF-8 size (24 bytes) exceeds maximum allowed limit (20 bytes)" in str(
        exc_info.value
    )
    assert exc_info.value.limit_name == "max_prompt_bytes"


def test_safety_limits_invalid_configuration():
    with pytest.raises(AIConfigurationError):
        SafetyLimits(max_prompt_chars=0)
    with pytest.raises(AIConfigurationError):
        SafetyLimits(timeout_seconds=-1.0)
    with pytest.raises(AIConfigurationError):
        SafetyLimits(timeout_seconds=0.0)


def test_response_output_validation():
    limits = SafetyLimits(max_response_bytes=10, max_output_tokens=5)

    validate_response_output(b"1234567890", output_tokens=5, limits=limits)

    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_response_output(b"12345678901", output_tokens=5, limits=limits)
    assert "Response size (11 bytes) exceeds maximum allowed limit (10 bytes)" in str(
        exc_info.value
    )

    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_response_output(b"1234", output_tokens=6, limits=limits)
    assert "Output token count (6) exceeds maximum allowed limit (5)" in str(exc_info.value)


def test_sanitized_exceptions_do_not_leak_secrets():
    secret_key = "sk-proj-supersecret1234567890abcdef"
    err = AIPolicyLimitError(
        f"Validation failed for secret token {secret_key}",
        limit_name="max_prompt_chars",
    )
    # The string representation or error message should sanitize sensitive strings if passed
    sanitized_str = err.sanitized_message()
    assert secret_key not in sanitized_str
    assert "[REDACTED]" in sanitized_str


def test_deterministic_text_redaction():
    secret_key = "sk-proj-123456789012345678901234"
    openai_key = "sk-abcdefghijklmnopqrstuvwxyz123456"
    bearer_header = "Authorization: Bearer my-secret-token-12345"
    url_with_auth = "https://user:mypassword@example.com/v1/models"

    text = (
        f"Calling with {secret_key} and {openai_key}. Header: {bearer_header}. Url: {url_with_auth}"
    )
    redacted = redact_secrets_from_text(text)

    assert secret_key not in redacted
    assert openai_key not in redacted
    assert "my-secret-token-12345" not in redacted
    assert "mypassword" not in redacted
    assert (
        "user:***@example.com" in redacted
        or "user:[REDACTED]@example.com" in redacted
        or "[REDACTED]" in redacted
    )


def test_deterministic_dict_redaction():
    data = {
        "api_key": "sk-123456789012345678901234567890",
        "nested": {
            "token": "secret_token_abc",
            "url": "https://api_key:secret_val@api.openai.com/v1",
            "normal_field": "public_data",
        },
        "list_items": [
            {"Authorization": "Bearer secret_jwt_xyz"},
            "sk-proj-999999999999999999999999",
        ],
    }

    redacted = redact_secrets_from_dict(data)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert "secret_val" not in redacted["nested"]["url"]
    assert redacted["nested"]["normal_field"] == "public_data"
    assert "secret_jwt_xyz" not in str(redacted["list_items"])
    assert "sk-proj-999999999999999999999999" not in str(redacted["list_items"])


def test_retry_policy_decision():
    policy = RetryPolicy(
        max_attempts=3,
        initial_interval_seconds=1.0,
        backoff_multiplier=2.0,
        max_interval_seconds=10.0,
        retryable_status_codes=(429, 502, 503, 504),
    )

    # Non-idempotent operations should NEVER be retried regardless of error
    decision = evaluate_retry(
        status_code=429,
        attempt=1,
        is_idempotent=False,
        policy=policy,
        pseudo_seed=0.5,
    )
    assert decision.should_retry is False
    assert decision.reason == "Operation is not idempotent"

    # Status codes 429, 502, 503, 504 should be retryable on idempotent operations
    for code in [429, 502, 503, 504]:
        decision = evaluate_retry(
            status_code=code,
            attempt=1,
            is_idempotent=True,
            policy=policy,
            pseudo_seed=0.0,
        )
        assert decision.should_retry is True
        assert decision.delay_seconds >= 1.0

    # Non-retryable codes (e.g., 400, 401, 403, 404, 500)
    for code in [400, 401, 403, 404, 500]:
        decision = evaluate_retry(
            status_code=code,
            attempt=1,
            is_idempotent=True,
            policy=policy,
            pseudo_seed=0.0,
        )
        assert decision.should_retry is False
        assert "Non-retryable status code" in decision.reason

    # Max attempts exceeded
    decision = evaluate_retry(
        status_code=429,
        attempt=3,
        is_idempotent=True,
        policy=policy,
        pseudo_seed=0.0,
    )
    assert decision.should_retry is False
    assert "Max retry attempts exceeded" in decision.reason


def test_retry_deterministic_backoff_calculation():
    policy = RetryPolicy(
        max_attempts=5,
        initial_interval_seconds=2.0,
        backoff_multiplier=2.0,
        max_interval_seconds=10.0,
        retryable_status_codes=(429,),
    )

    # Attempt 1 base delay: 2.0 * (2^0) = 2.0. With pseudo_seed=0.5, jitter policy provides deterministic value
    d1_a = evaluate_retry(
        status_code=429, attempt=1, is_idempotent=True, policy=policy, pseudo_seed=0.5
    )
    d1_b = evaluate_retry(
        status_code=429, attempt=1, is_idempotent=True, policy=policy, pseudo_seed=0.5
    )
    assert d1_a.delay_seconds == d1_b.delay_seconds
    assert 2.0 <= d1_a.delay_seconds <= 3.0

    # Attempt 3 base delay: 2.0 * (2^2) = 8.0
    d3 = evaluate_retry(
        status_code=429, attempt=3, is_idempotent=True, policy=policy, pseudo_seed=0.0
    )
    assert d3.delay_seconds == 8.0

    # Attempt 4 base delay: 2.0 * (2^3) = 16.0 -> bounded by max_interval_seconds (10.0)
    d4 = evaluate_retry(
        status_code=429, attempt=4, is_idempotent=True, policy=policy, pseudo_seed=0.0
    )
    assert d4.delay_seconds == 10.0


def test_execution_budget_tracker_single_execution():
    tracker = ExecutionBudgetTracker(max_ai_calls=2, max_total_tokens=100)

    # 1st call
    tracker.record_call(tokens_used=40)
    assert tracker.calls_count == 1
    assert tracker.tokens_count == 40

    # 2nd call
    tracker.record_call(tokens_used=50)
    assert tracker.calls_count == 2
    assert tracker.tokens_count == 90

    # Exceeding max calls on 3rd attempt
    with pytest.raises(AIPolicyLimitError) as exc_info:
        tracker.record_call(tokens_used=5)
    assert "Workflow AI call limit (2 calls) exceeded" in str(exc_info.value)
    assert exc_info.value.limit_name == "max_workflow_ai_calls"


def test_execution_budget_tracker_token_exhaustion():
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=50)
    tracker.record_call(tokens_used=30)

    with pytest.raises(AIPolicyLimitError) as exc_info:
        tracker.record_call(tokens_used=30)
    assert "Workflow AI total token limit (50 tokens) exceeded" in str(exc_info.value)
    assert exc_info.value.limit_name == "max_total_tokens"


def test_execution_budget_isolation_no_global_mutable_state():
    tracker_a = ExecutionBudgetTracker(max_ai_calls=1, max_total_tokens=100)
    tracker_b = ExecutionBudgetTracker(max_ai_calls=1, max_total_tokens=100)

    tracker_a.record_call(tokens_used=50)
    assert tracker_a.calls_count == 1
    assert tracker_b.calls_count == 0  # isolated!
