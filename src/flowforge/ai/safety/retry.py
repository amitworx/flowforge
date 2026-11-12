"""Pure retry policy and deterministic backoff decision logic.

No actual network calls, sleep commands, or side effects are performed.
"""

import math
from dataclasses import dataclass

from flowforge.ai.safety.exceptions import AIConfigurationError


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry evaluation."""

    max_attempts: int = 3
    initial_interval_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_interval_seconds: float = 10.0
    retryable_status_codes: tuple[int, ...] = (429, 502, 503, 504)

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise AIConfigurationError("max_attempts must be an integer, not bool")
        if self.max_attempts <= 0:
            raise AIConfigurationError("max_attempts must be a strictly positive integer")

        for val, name in [
            (self.initial_interval_seconds, "initial_interval_seconds"),
            (self.max_interval_seconds, "max_interval_seconds"),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise AIConfigurationError(f"{name} must be a real number, not bool")
            if math.isnan(val) or math.isinf(val) or val <= 0.0:
                raise AIConfigurationError(f"{name} must be a strictly positive finite float")

        if isinstance(self.backoff_multiplier, bool) or not isinstance(
            self.backoff_multiplier, (int, float)
        ):
            raise AIConfigurationError("backoff_multiplier must be a real number, not bool")
        if (
            math.isnan(self.backoff_multiplier)
            or math.isinf(self.backoff_multiplier)
            or self.backoff_multiplier < 1.0
        ):
            raise AIConfigurationError("backoff_multiplier must be >= 1.0 finite float")

        if self.max_interval_seconds < self.initial_interval_seconds:
            raise AIConfigurationError("max_interval_seconds must be >= initial_interval_seconds")

        if not isinstance(self.retryable_status_codes, (tuple, list)):
            raise AIConfigurationError("retryable_status_codes must be a tuple or list of integers")

        for code in self.retryable_status_codes:
            if isinstance(code, bool) or not isinstance(code, int):
                raise AIConfigurationError(
                    "status code in retryable_status_codes must be an integer, not bool"
                )
            if code < 100 or code > 599:
                raise AIConfigurationError(
                    "status code in retryable_status_codes must be between 100 and 599"
                )


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
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        raise AIConfigurationError("status_code must be an integer, not bool")

    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise AIConfigurationError("attempt must be an integer, not bool")
    if attempt < 1:
        raise AIConfigurationError("attempt must be >= 1")

    if not isinstance(is_idempotent, bool):
        raise AIConfigurationError("is_idempotent must be a boolean")

    if isinstance(pseudo_seed, bool) or not isinstance(pseudo_seed, (int, float)):
        raise AIConfigurationError("pseudo_seed must be a finite float, not bool")
    if math.isnan(pseudo_seed) or math.isinf(pseudo_seed):
        raise AIConfigurationError("pseudo_seed must be a finite float")

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

    # Calculate bounded exponential backoff safely without overflow
    exponent = max(0, attempt - 1)
    if policy.backoff_multiplier == 1.0 or exponent == 0:
        base_delay = policy.initial_interval_seconds
    else:
        # Check if exponent would exceed log2 threshold to avoid OverflowError
        max_ratio = policy.max_interval_seconds / policy.initial_interval_seconds
        try:
            # If exponent * log(multiplier) >= log(max_ratio), it reaches max_interval_seconds
            log_mult = math.log(policy.backoff_multiplier)
            log_ratio = math.log(max_ratio) if max_ratio > 0 else 0.0
            if exponent * log_mult >= log_ratio:
                base_delay = policy.max_interval_seconds
            else:
                base_delay = policy.initial_interval_seconds * (policy.backoff_multiplier**exponent)
        except (OverflowError, ValueError):
            base_delay = policy.max_interval_seconds

    bounded_delay = min(base_delay, policy.max_interval_seconds)

    # Apply deterministic jitter: jitter factor in [0, 1) scaled to 0.5 * bounded_delay
    clamped_seed = max(0.0, min(1.0, float(pseudo_seed)))
    jitter = clamped_seed * (bounded_delay * 0.5)
    total_delay = min(bounded_delay + jitter, policy.max_interval_seconds)

    return RetryDecision(
        should_retry=True,
        delay_seconds=round(total_delay, 4),
        reason=f"Retryable status code {status_code} on attempt {attempt}",
    )
