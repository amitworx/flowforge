"""Tests for pure AI execution safety contracts (M0-018A-2).

Tests cover:
- Public package import correctness (`from flowforge.ai.safety...`).
- Typed validation limits with fail-closed numeric constraints:
  - Rejection of bool, float for integer fields, negative, zero, NaN, Inf, extreme floats.
  - Safe prompt UTF-8 byte and character limits.
  - Safe response byte and token limits.
- Deterministic log, text, and structured metadata redaction.
- Safe error categories and exceptions (fail-closed at construction):
  - AISafetyError stores sanitized values in `self.args`, `self.message`, and `self.details` at `__init__`.
  - Serializers and loggers inspecting `exc.args`, `exc.message`, `exc.details`, `str(exc)`, `repr(exc)` NEVER see raw secrets.
  - Support detached causes on sensitive failures.
- Deterministic retry decision policy:
  - Strict validation on status_code, attempt, is_idempotent, pseudo_seed, and retryable_status_codes.
  - Overflow-proof exponential backoff: extreme valid configurations (e.g. attempt=1000, multiplier=100.0) never raise OverflowError and always clamp safely to <= max_interval_seconds.
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
    limits = SafetyLimits(
        max_prompt_chars=100,
        max_prompt_bytes=200,
        max_response_bytes=500,
        max_output_tokens=50,
        max_workflow_ai_calls=10,
        timeout_seconds=30.0,
    )
    assert limits.max_prompt_chars == 100

    # Reject bool
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

    with pytest.raises(AIPolicyLimitError, match="Prompt must be a string"):
        validate_prompt_input(12345, limits)  # type: ignore

    validate_prompt_input("1234567890", limits)

    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_prompt_input("12345678901", limits)
    assert exc_info.value.limit_name == "max_prompt_chars"
    assert "Prompt length (11 chars) exceeds maximum allowed limit (10 chars)" in str(
        exc_info.value
    )

    validate_prompt_input("🔒🔒🔒", limits)
    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_prompt_input("🔒🔒🔒🔒🔒🔒", limits)
    assert exc_info.value.limit_name == "max_prompt_bytes"
    assert "Prompt UTF-8 size (24 bytes) exceeds maximum allowed limit (20 bytes)" in str(
        exc_info.value
    )


def test_response_output_fail_closed_validation():
    limits = SafetyLimits(max_response_bytes=10, max_output_tokens=5)

    with pytest.raises(AIPolicyLimitError, match="Response must be bytes"):
        validate_response_output("not bytes", output_tokens=1, limits=limits)  # type: ignore

    with pytest.raises(AIPolicyLimitError, match="Output tokens must be an integer, not bool"):
        validate_response_output(b"abc", output_tokens=True, limits=limits)  # type: ignore
    with pytest.raises(AIPolicyLimitError, match="Output tokens must be a non-negative integer"):
        validate_response_output(b"abc", output_tokens=-1, limits=limits)
    with pytest.raises(AIPolicyLimitError, match="Output tokens must be a non-negative integer"):
        validate_response_output(b"abc", output_tokens=2.5, limits=limits)  # type: ignore

    validate_response_output(b"1234567890", output_tokens=5, limits=limits)

    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_response_output(b"12345678901", output_tokens=5, limits=limits)
    assert exc_info.value.limit_name == "max_response_bytes"

    with pytest.raises(AIPolicyLimitError) as exc_info:
        validate_response_output(b"1234", output_tokens=6, limits=limits)
    assert exc_info.value.limit_name == "max_output_tokens"


def test_exception_construction_sanitization_args_message_details():
    """AISafetyError must sanitize raw secrets at construction time into args, message, and details."""
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

    # 1. Check self.args
    assert len(err.args) == 1
    args_str = str(err.args[0])
    assert secret_key not in args_str
    assert "secret_jwt_token_12345" not in args_str
    assert "password123" not in args_str
    assert "[REDACTED]" in args_str

    # 2. Check self.message
    assert secret_key not in err.message
    assert "secret_jwt_token_12345" not in err.message
    assert "password123" not in err.message

    # 3. Check self.details directly (not just via a helper method)
    assert err.details["api_key"] == "[REDACTED]"
    assert err.details["nested"]["auth"] == "[REDACTED]"
    assert "password123" not in err.details["nested"]["url"]
    assert err.details["nested"]["safe"] == "ok"

    # 4. Check str() and repr()
    assert secret_key not in str(err)
    assert secret_key not in repr(err)


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
    with pytest.raises(AIConfigurationError, match="must be an integer, not bool"):
        RetryPolicy(max_attempts=True)  # type: ignore

    with pytest.raises(
        AIConfigurationError, match="max_attempts must be a strictly positive integer"
    ):
        RetryPolicy(max_attempts=0)

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

    # Validate retryable_status_codes entries
    with pytest.raises(
        AIConfigurationError,
        match="status code in retryable_status_codes must be an integer, not bool",
    ):
        RetryPolicy(retryable_status_codes=(429, True))  # type: ignore
    with pytest.raises(
        AIConfigurationError,
        match="status code in retryable_status_codes must be between 100 and 599",
    ):
        RetryPolicy(retryable_status_codes=(429, 999))


def test_evaluate_retry_validation_and_decisions():
    policy = RetryPolicy(
        max_attempts=3,
        initial_interval_seconds=1.0,
        backoff_multiplier=2.0,
        max_interval_seconds=10.0,
        retryable_status_codes=(429, 502, 503, 504),
    )

    with pytest.raises(AIConfigurationError, match="status_code must be an integer, not bool"):
        evaluate_retry(status_code=True, attempt=1, is_idempotent=True, policy=policy)  # type: ignore

    with pytest.raises(AIConfigurationError, match="attempt must be an integer, not bool"):
        evaluate_retry(status_code=429, attempt=False, is_idempotent=True, policy=policy)  # type: ignore
    with pytest.raises(AIConfigurationError, match="attempt must be >= 1"):
        evaluate_retry(status_code=429, attempt=0, is_idempotent=True, policy=policy)

    with pytest.raises(AIConfigurationError, match="is_idempotent must be a boolean"):
        evaluate_retry(status_code=429, attempt=1, is_idempotent=1, policy=policy)  # type: ignore

    with pytest.raises(AIConfigurationError, match="pseudo_seed must be a finite float, not bool"):
        evaluate_retry(
            status_code=429, attempt=1, is_idempotent=True, policy=policy, pseudo_seed=True
        )  # type: ignore
    with pytest.raises(AIConfigurationError, match="pseudo_seed must be a finite float"):
        evaluate_retry(
            status_code=429, attempt=1, is_idempotent=True, policy=policy, pseudo_seed=float("nan")
        )

    decision = evaluate_retry(status_code=429, attempt=1, is_idempotent=False, policy=policy)
    assert decision.should_retry is False
    assert decision.reason == "Operation is not idempotent"

    for code in [429, 502, 503, 504]:
        dec = evaluate_retry(
            status_code=code, attempt=1, is_idempotent=True, policy=policy, pseudo_seed=0.5
        )
        assert dec.should_retry is True
        assert 1.0 <= dec.delay_seconds <= 1.5

    for code in [400, 401, 403, 404, 500]:
        dec = evaluate_retry(status_code=code, attempt=1, is_idempotent=True, policy=policy)
        assert dec.should_retry is False

    dec = evaluate_retry(status_code=429, attempt=3, is_idempotent=True, policy=policy)
    assert dec.should_retry is False
    assert "Max retry attempts exceeded" in dec.reason


def test_evaluate_retry_overflow_safety_with_extreme_values():
    """Verify that extreme configurations (e.g. attempt=10000, multiplier=100.0) NEVER cause OverflowError."""
    policy = RetryPolicy(
        max_attempts=100_000,
        initial_interval_seconds=1.0,
        backoff_multiplier=100.0,
        max_interval_seconds=60.0,
        retryable_status_codes=(429,),
    )

    # Attempt 5000 with multiplier 100 would normally raise OverflowError (100^5000)
    decision = evaluate_retry(
        status_code=429,
        attempt=5000,
        is_idempotent=True,
        policy=policy,
        pseudo_seed=0.5,
    )
    assert decision.should_retry is True
    assert decision.delay_seconds <= policy.max_interval_seconds
    assert decision.delay_seconds == 60.0


def test_execution_budget_tracker_numeric_constraints_and_no_mutation_on_rejection():
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

    with pytest.raises(AIConfigurationError, match="tokens_used must be a non-negative integer"):
        tracker.record_call(tokens_used=-10)
    assert tracker.calls_count == 0
    assert tracker.tokens_count == 0

    with pytest.raises(AIConfigurationError, match="tokens_used must be an integer, not bool"):
        tracker.record_call(tokens_used=True)  # type: ignore
    assert tracker.calls_count == 0
    assert tracker.tokens_count == 0

    tracker.record_call(tokens_used=40)
    assert tracker.calls_count == 1
    assert tracker.tokens_count == 40

    with pytest.raises(AIPolicyLimitError) as exc_info:
        tracker.record_call(tokens_used=70)
    assert exc_info.value.limit_name == "max_total_tokens"
    assert tracker.calls_count == 1
    assert tracker.tokens_count == 40

    tracker.record_call(tokens_used=50)
    assert tracker.calls_count == 2
    assert tracker.tokens_count == 90

    with pytest.raises(AIPolicyLimitError) as exc_info:
        tracker.record_call(tokens_used=5)
    assert exc_info.value.limit_name == "max_workflow_ai_calls"
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
