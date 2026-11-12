"""Tests for pure AI execution safety contracts (M0-018A-1).

Tests cover:
- Public package import correctness (`from flowforge.ai.safety...`).
- Typed validation limits with fail-closed numeric constraints:
  - Rejection of bool, float for integer fields, negative, zero, NaN, Inf, extreme floats.
  - Safe prompt UTF-8 byte and character limits.
  - Safe response byte and token limits.
- Deterministic log, text, and structured metadata redaction.
- Safe error categories and exceptions:
  - AISafetyError string representation and details safe by default (never leaks raw secrets, URL credentials, bearer tokens, or nested dictionary secrets).
- Deterministic retry decision policy:
  - Rejection of invalid status codes, negative attempts, non-ints, invalid float jitter/intervals.
  - Idempotent operations only.
  - Deterministic bounded exponential backoff.
- In-memory request budget tracker:
  - Thread safety, isolation per execution.
  - Rejection of negative token additions, non-ints, bools.
  - Guarantee that rejected calls do NOT mutate budget counters.
"""

import threading

import pytest

from flowforge.ai.safety import (
    AIConfigurationError,
    AIMalformedResponseError,
    AINonRetryableProviderError,
    AIPolicyLimitError,
    AIRetryableTransportError,
    AISafetyError,
    ExecutionBudgetTracker,
    RetryPolicy,
    SafetyLimits,
    evaluate_retry,
    redact_secrets_from_dict,
    redact_secrets_from_text,
    validate_prompt_input,
    validate_response_output,
)


def test_public_package_imports_and_types():
    """Verify package import identity and exports from flowforge.ai.safety."""
    assert issubclass(AIConfigurationError, AISafetyError)
    assert issubclass(AIPolicyLimitError, AISafetyError)
    assert issubclass(AIRetryableTransportError, AISafetyError)
    assert issubclass(AINonRetryableProviderError, AISafetyError)
    assert issubclass(AIMalformedResponseError, AISafetyError)


def test_safety_limits_fail_closed_numeric_constraints():
    """SafetyLimits must strictly reject booleans, non-integers, negatives, zero, NaN, and Inf."""
    # Valid limits
    limits = SafetyLimits(
        max_prompt_chars=100,
        max_prompt_bytes=200,
        max_response_bytes=500,
        max_output_tokens=50,
        max_workflow_ai_calls=10,
        timeout_seconds=30.0,
    )
    assert limits.max_prompt_chars == 100

    # Reject bool (bool is a subclass of int in Python)
    with pytest.raises(AIConfigurationError, match="must be an integer, not bool"):
        SafetyLimits(max_prompt_chars=True)  # type: ignore
    with pytest.raises(AIConfigurationError, match="must be an integer, not bool"):
        SafetyLimits(max_output_tokens=False)  # type: ignore

    # Reject float for integer fields
    with pytest.raises(AIConfigurationError, match="must be an integer"):
        SafetyLimits(max_prompt_chars=100.5)  # type: ignore

    # Reject zero and negative integers
    with pytest.raises(AIConfigurationError, match="must be a strictly positive integer"):
        SafetyLimits(max_prompt_chars=0)
    with pytest.raises(AIConfigurationError, match="must be a strictly positive integer"):
        SafetyLimits(max_prompt_bytes=-10)
    with pytest.raises(AIConfigurationError, match="must be a strictly positive integer"):
        SafetyLimits(max_response_bytes=0)
    with pytest.raises(AIConfigurationError, match="must be a strictly positive integer"):
        SafetyLimits(max_output_tokens=-1)
    with pytest.raises(AIConfigurationError, match="must be a strictly positive integer"):
        SafetyLimits(max_workflow_ai_calls=0)

    # Reject timeout_seconds: bool, <= 0, NaN, Inf
    with pytest.raises(AIConfigurationError, match="must be a real number, not bool"):
        SafetyLimits(timeout_seconds=True)  # type: ignore
    with pytest.raises(AIConfigurationError, match="must be a strictly positive finite float"):
        SafetyLimits(timeout_seconds=0.0)
    with pytest.raises(AIConfigurationError, match="must be a strictly positive finite float"):
        SafetyLimits(timeout_seconds=-5.0)
    with pytest.raises(AIConfigurationError, match="must be a strictly positive finite float"):
        SafetyLimits(timeout_seconds=float("nan"))
    with pytest.raises(AIConfigurationError, match="must be a strictly positive finite float"):
        SafetyLimits(timeout_seconds=float("inf"))
    with pytest.raises(AIConfigurationError, match="must be a strictly positive finite float"):
        SafetyLimits(timeout_seconds=float("-inf"))


def test_prompt_input_boundary_and_validation():
    limits = SafetyLimits(max_prompt_chars=10, max_prompt_bytes=20)

    # Non-string prompt rejected
    with pytest.raises(AIPolicyLimitError, match="Prompt must be a string"):
        validate_prompt_input(12345, limits)  # type: ignore

    # Exact boundary
    validate_prompt_input("1234567890", limits)

    # Character limit violation
    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_prompt_input("12345678901", limits)
    assert exc_info.value.limit_name == "max_prompt_chars"
    assert "Prompt length (11 chars) exceeds maximum allowed limit (10 chars)" in str(
        exc_info.value
    )

    # Unicode UTF-8 multi-byte boundary and violation
    validate_prompt_input("🔒🔒🔒", limits)  # 3 chars, 12 bytes <= 20
    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_prompt_input("🔒🔒🔒🔒🔒🔒", limits)  # 6 chars (<=10), 24 bytes (>20)
    assert exc_info.value.limit_name == "max_prompt_bytes"
    assert "Prompt UTF-8 size (24 bytes) exceeds maximum allowed limit (20 bytes)" in str(
        exc_info.value
    )


def test_response_output_fail_closed_validation():
    limits = SafetyLimits(max_response_bytes=10, max_output_tokens=5)

    # Non-bytes rejected
    with pytest.raises(AIPolicyLimitError, match="Response must be bytes"):
        validate_response_output("not bytes", output_tokens=1, limits=limits)  # type: ignore

    # Reject non-int, bool, or negative output_tokens
    with pytest.raises(AIPolicyLimitError, match="Output tokens must be an integer, not bool"):
        validate_response_output(b"abc", output_tokens=True, limits=limits)  # type: ignore
    with pytest.raises(AIPolicyLimitError, match="Output tokens must be a non-negative integer"):
        validate_response_output(b"abc", output_tokens=-1, limits=limits)
    with pytest.raises(AIPolicyLimitError, match="Output tokens must be a non-negative integer"):
        validate_response_output(b"abc", output_tokens=2.5, limits=limits)  # type: ignore

    # Exact boundary
    validate_response_output(b"1234567890", output_tokens=5, limits=limits)

    # Exceed byte length
    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_response_output(b"12345678901", output_tokens=5, limits=limits)
    assert exc_info.value.limit_name == "max_response_bytes"

    # Exceed tokens
    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_response_output(b"1234", output_tokens=6, limits=limits)
    assert exc_info.value.limit_name == "max_output_tokens"


def test_exception_string_and_details_safe_by_default():
    """AISafetyError __str__, __repr__, and details must never echo sensitive tokens."""
    secret_key = "sk-proj-supersecret1234567890abcdef"
    bearer_token = "Bearer secret_jwt_token_12345"
    url_cred = "https://user:password123@api.openai.com/v1"

    err = AISafetyError(
        f"Failed call with key {secret_key} and header {bearer_token} to {url_cred}",
        details={
            "api_key": secret_key,
            "nested": {"auth": bearer_token, "url": url_cred, "safe": "ok"},
        },
    )

    err_str = str(err)
    err_repr = repr(err)

    # Verify str(err) does NOT leak secrets
    assert secret_key not in err_str
    assert "secret_jwt_token_12345" not in err_str
    assert "password123" not in err_str
    assert "[REDACTED]" in err_str

    # Verify repr(err) does NOT leak secrets
    assert secret_key not in err_repr
    assert "secret_jwt_token_12345" not in err_repr
    assert "password123" not in err_repr

    # Verify sanitized details property
    sanitized_details = err.sanitized_details()
    assert sanitized_details["api_key"] == "[REDACTED]"
    assert sanitized_details["nested"]["auth"] == "[REDACTED]"
    assert "password123" not in sanitized_details["nested"]["url"]
    assert sanitized_details["nested"]["safe"] == "ok"


def test_deterministic_text_and_dict_redaction():
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

    data = {
        "api_key": secret_key,
        "token": "secret_token_123",
        "nested": {
            "password": "pass",
            "url": url_with_auth,
            "normal": 1234,
        },
        "list_items": [{"secret": "s1"}, "sk-proj-999999999999999999999999"],
    }
    redacted_data = redact_secrets_from_dict(data)
    assert redacted_data["api_key"] == "[REDACTED]"
    assert redacted_data["token"] == "[REDACTED]"
    assert redacted_data["nested"]["password"] == "[REDACTED]"
    assert "mypassword" not in redacted_data["nested"]["url"]
    assert redacted_data["nested"]["normal"] == 1234
    assert redacted_data["list_items"][0]["secret"] == "[REDACTED]"
    assert "999999999999999999999999" not in str(redacted_data["list_items"])


def test_retry_policy_fail_closed_validation():
    # Reject bool for max_attempts
    with pytest.raises(AIConfigurationError, match="must be an integer, not bool"):
        RetryPolicy(max_attempts=True)  # type: ignore

    # Reject non-positive max_attempts
    with pytest.raises(
        AIConfigurationError, match="max_attempts must be a strictly positive integer"
    ):
        RetryPolicy(max_attempts=0)

    # Reject invalid intervals: bool, <= 0, NaN, Inf
    with pytest.raises(
        AIConfigurationError,
        match="initial_interval_seconds must be a strictly positive finite float",
    ):
        RetryPolicy(initial_interval_seconds=0.0)
    with pytest.raises(
        AIConfigurationError,
        match="initial_interval_seconds must be a strictly positive finite float",
    ):
        RetryPolicy(initial_interval_seconds=float("nan"))
    with pytest.raises(
        AIConfigurationError, match="backoff_multiplier must be >= 1.0 finite float"
    ):
        RetryPolicy(backoff_multiplier=0.5)
    with pytest.raises(
        AIConfigurationError, match="backoff_multiplier must be >= 1.0 finite float"
    ):
        RetryPolicy(backoff_multiplier=float("inf"))
    with pytest.raises(
        AIConfigurationError, match="max_interval_seconds must be >= initial_interval_seconds"
    ):
        RetryPolicy(initial_interval_seconds=10.0, max_interval_seconds=5.0)


def test_evaluate_retry_validation_and_decisions():
    policy = RetryPolicy(
        max_attempts=3,
        initial_interval_seconds=1.0,
        backoff_multiplier=2.0,
        max_interval_seconds=10.0,
        retryable_status_codes=(429, 502, 503, 504),
    )

    # Validation: status_code must be int (not bool)
    with pytest.raises(AIConfigurationError, match="status_code must be an integer, not bool"):
        evaluate_retry(status_code=True, attempt=1, is_idempotent=True, policy=policy)  # type: ignore

    # Validation: attempt must be >= 1 int (not bool)
    with pytest.raises(AIConfigurationError, match="attempt must be an integer, not bool"):
        evaluate_retry(status_code=429, attempt=False, is_idempotent=True, policy=policy)  # type: ignore
    with pytest.raises(AIConfigurationError, match="attempt must be >= 1"):
        evaluate_retry(status_code=429, attempt=0, is_idempotent=True, policy=policy)

    # Validation: pseudo_seed must be finite float/int
    with pytest.raises(AIConfigurationError, match="pseudo_seed must be a finite float"):
        evaluate_retry(
            status_code=429, attempt=1, is_idempotent=True, policy=policy, pseudo_seed=float("nan")
        )

    # Non-idempotent never retried
    decision = evaluate_retry(status_code=429, attempt=1, is_idempotent=False, policy=policy)
    assert decision.should_retry is False
    assert decision.reason == "Operation is not idempotent"

    # Retryable codes
    for code in [429, 502, 503, 504]:
        dec = evaluate_retry(
            status_code=code, attempt=1, is_idempotent=True, policy=policy, pseudo_seed=0.5
        )
        assert dec.should_retry is True
        assert 1.0 <= dec.delay_seconds <= 1.5

    # Non-retryable codes
    for code in [400, 401, 403, 404, 500]:
        dec = evaluate_retry(status_code=code, attempt=1, is_idempotent=True, policy=policy)
        assert dec.should_retry is False

    # Max attempts reached
    dec = evaluate_retry(status_code=429, attempt=3, is_idempotent=True, policy=policy)
    assert dec.should_retry is False
    assert "Max retry attempts exceeded" in dec.reason

    # Bounded delay calculation despite large attempts
    policy_high = RetryPolicy(
        max_attempts=100,
        initial_interval_seconds=1.0,
        backoff_multiplier=2.0,
        max_interval_seconds=8.0,
    )
    dec = evaluate_retry(
        status_code=429, attempt=20, is_idempotent=True, policy=policy_high, pseudo_seed=0.5
    )
    assert dec.should_retry is True
    assert dec.delay_seconds == 8.0  # bounded by max_interval_seconds


def test_execution_budget_tracker_numeric_constraints_and_no_mutation_on_rejection():
    # Reject invalid budget tracker construction
    with pytest.raises(
        AIConfigurationError, match="max_ai_calls must be a strictly positive integer, not bool"
    ):
        ExecutionBudgetTracker(max_ai_calls=True)  # type: ignore
    with pytest.raises(
        AIConfigurationError, match="max_ai_calls must be a strictly positive integer"
    ):
        ExecutionBudgetTracker(max_ai_calls=0)
    with pytest.raises(
        AIConfigurationError, match="max_total_tokens must be a strictly positive integer"
    ):
        ExecutionBudgetTracker(max_total_tokens=-5)

    tracker = ExecutionBudgetTracker(max_ai_calls=2, max_total_tokens=100)

    # Reject negative tokens
    with pytest.raises(AIConfigurationError, match="tokens_used must be a non-negative integer"):
        tracker.record_call(tokens_used=-10)
    # Ensure counters did NOT mutate
    assert tracker.calls_count == 0
    assert tracker.tokens_count == 0

    # Reject bool for tokens_used
    with pytest.raises(AIConfigurationError, match="tokens_used must be an integer, not bool"):
        tracker.record_call(tokens_used=True)  # type: ignore
    assert tracker.calls_count == 0
    assert tracker.tokens_count == 0

    # Valid call 1
    tracker.record_call(tokens_used=40)
    assert tracker.calls_count == 1
    assert tracker.tokens_count == 40

    # Token limit violation: attempting to consume 70 tokens (40 + 70 = 110 > 100)
    with pytest.raises(AIPolicyLimitError) as exc_info:
        tracker.record_call(tokens_used=70)
    assert exc_info.value.limit_name == "max_total_tokens"
    # Ensure NO counter mutation on policy rejection
    assert tracker.calls_count == 1
    assert tracker.tokens_count == 40

    # Valid call 2
    tracker.record_call(tokens_used=50)
    assert tracker.calls_count == 2
    assert tracker.tokens_count == 90

    # Call limit violation: attempting 3rd call (> max_ai_calls 2)
    with pytest.raises(AIPolicyLimitError) as exc_info:
        tracker.record_call(tokens_used=5)
    assert exc_info.value.limit_name == "max_workflow_ai_calls"
    # Ensure NO counter mutation on call limit rejection
    assert tracker.calls_count == 2
    assert tracker.tokens_count == 90


def test_execution_budget_tracker_concurrency():
    tracker = ExecutionBudgetTracker(max_ai_calls=100, max_total_tokens=10_000)

    def worker():
        for _ in range(10):
            tracker.record_call(tokens_used=5)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.calls_count == 100
    assert tracker.tokens_count == 500
