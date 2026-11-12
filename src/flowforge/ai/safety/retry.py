"""Pure retry policy and deterministic backoff decision logic.

No actual network calls, sleep commands, or side effects are performed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry evaluation."""

    max_attempts: int = 3
    initial_interval_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_interval_seconds: float = 10.0
    retryable_status_codes: tuple[int, ...] = (429, 502, 503, 504)


@dataclass(frozen=True)
class RetryDecision:
    """Deterministic result of retry evaluation."""

    should_retry: bool
    delay_seconds: float
    reason: str


def evaluate_retry(
    status_code: int,
    attempt: int,
    is_idempotent: bool,
    policy: RetryPolicy,
    pseudo_seed: float = 0.0,
) -> RetryDecision:
    """Evaluate whether an operation is eligible for retry deterministically.

    Args:
        status_code: HTTP response status code.
        attempt: The 1-based attempt count already executed.
        is_idempotent: Whether the request is safe to repeat.
        policy: The RetryPolicy rules.
        pseudo_seed: A deterministic float in [0.0, 1.0) used for bounded jitter calculations.
    """
    if not is_idempotent:
        return RetryDecision(
            should_retry=False,
            delay_seconds=0.0,
            reason="Operation is not idempotent",
        )

    if attempt >= policy.max_attempts:
        return RetryDecision(
            should_retry=False,
            delay_seconds=0.0,
            reason=f"Max retry attempts exceeded ({attempt}/{policy.max_attempts})",
        )

    if status_code not in policy.retryable_status_codes:
        return RetryDecision(
            should_retry=False,
            delay_seconds=0.0,
            reason=f"Non-retryable status code: {status_code}",
        )

    # Calculate bounded exponential backoff
    # Base delay = initial * (multiplier ^ (attempt - 1))
    exponent = max(0, attempt - 1)
    base_delay = policy.initial_interval_seconds * (policy.backoff_multiplier**exponent)
    bounded_delay = min(base_delay, policy.max_interval_seconds)

    # Apply deterministic jitter: jitter factor in [0, 1) scaled to 0.5 * bounded_delay
    clamped_seed = max(0.0, min(1.0, pseudo_seed))
    jitter = clamped_seed * (bounded_delay * 0.5)
    total_delay = min(bounded_delay + jitter, policy.max_interval_seconds)

    return RetryDecision(
        should_retry=True,
        delay_seconds=round(total_delay, 4),
        reason=f"Retryable status code {status_code} on attempt {attempt}",
    )
