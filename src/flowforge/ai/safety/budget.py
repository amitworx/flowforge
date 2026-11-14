"""In-memory deterministic request budget tracker for single executions.

No global mutable state, thread-safe, and zero persistence/database dependencies.
"""

import threading

from flowforge.ai.safety.exceptions import AIConfigurationError, AIPolicyLimitError


class ExecutionBudgetTracker:
    """Thread-safe, isolated, in-memory budget tracker for a single workflow or request execution."""

    def __init__(self, max_ai_calls: int = 50, max_total_tokens: int = 100_000) -> None:
        if isinstance(max_ai_calls, bool) or not isinstance(max_ai_calls, int):
            raise AIConfigurationError("max_ai_calls must be a strictly positive integer, not bool")
        if max_ai_calls <= 0:
            raise AIConfigurationError("max_ai_calls must be a strictly positive integer")

        if isinstance(max_total_tokens, bool) or not isinstance(max_total_tokens, int):
            raise AIConfigurationError(
                "max_total_tokens must be a strictly positive integer, not bool"
            )
        if max_total_tokens <= 0:
            raise AIConfigurationError("max_total_tokens must be a strictly positive integer")

        self.max_ai_calls = max_ai_calls
        self.max_total_tokens = max_total_tokens
        self._lock = threading.Lock()
        self._calls_count = 0
        self._tokens_count = 0
        self._reserved_tokens = 0
        self._reservations: dict[str, int] = {}

    @property
    def calls_count(self) -> int:
        with self._lock:
            return self._calls_count

    @property
    def tokens_count(self) -> int:
        with self._lock:
            return self._tokens_count

    @property
    def reserved_tokens(self) -> int:
        with self._lock:
            return self._reserved_tokens

    @property
    def has_overrun(self) -> bool:
        """Returns True if recorded token usage has exceeded max_total_tokens."""
        with self._lock:
            return self._tokens_count > self.max_total_tokens

    @property
    def tokens_overrun(self) -> int:
        """Returns the number of tokens recorded beyond max_total_tokens, or 0 if within limit."""
        with self._lock:
            return max(0, self._tokens_count - self.max_total_tokens)

    @property
    def effective_tokens_count(self) -> int:
        with self._lock:
            return self._tokens_count + self._reserved_tokens

    def reserve_call(self, estimated_tokens: int = 0) -> str:
        """Atomically check and reserve one AI call and estimated token allowance under lock.

        Returns a unique reservation ID string.
        Raises AIConfigurationError or AIPolicyLimitError fail-closed if limits are reached.
        """
        import uuid

        if isinstance(estimated_tokens, bool) or not isinstance(estimated_tokens, int):
            raise AIConfigurationError("estimated_tokens must be an integer, not bool")
        if estimated_tokens < 0:
            raise AIConfigurationError("estimated_tokens must be a non-negative integer")

        with self._lock:
            if self._calls_count + 1 > self.max_ai_calls:
                raise AIPolicyLimitError(
                    f"Workflow AI call limit ({self.max_ai_calls} calls) exceeded",
                    limit_name="max_workflow_ai_calls",
                    actual_value=self._calls_count + 1,
                    limit_value=self.max_ai_calls,
                )

            if (
                self._tokens_count + self._reserved_tokens + estimated_tokens
                > self.max_total_tokens
            ):
                raise AIPolicyLimitError(
                    f"Workflow AI total token limit ({self.max_total_tokens} tokens) exceeded by estimated request ({estimated_tokens} tokens)",
                    limit_name="max_total_tokens",
                    actual_value=self._tokens_count + self._reserved_tokens + estimated_tokens,
                    limit_value=self.max_total_tokens,
                )

            res_id = str(uuid.uuid4())
            self._calls_count += 1
            self._reserved_tokens += estimated_tokens
            self._reservations[res_id] = estimated_tokens
            return res_id

    def complete_call(
        self,
        reservation_id: str,
        actual_tokens: int | None = None,
        *,
        failed: bool = False,
    ) -> None:
        """Complete a reserved AI call, reconciling actual tokens used.

        - If reservation_id is not in active reservations, raises AIConfigurationError fail-closed.
        - If adding final tokens exceeds max_total_tokens, raises AIPolicyLimitError fail-closed and preserves limit.
        - If failed=True and actual_tokens is None or 0, retains 0 tokens if not sent, or retains conservative usage.
        - Actual tokens replace the reserved tokens under lock without double counting calls or allowing negative decrements.
        """
        if actual_tokens is not None:
            if isinstance(actual_tokens, bool) or not isinstance(actual_tokens, int):
                raise AIConfigurationError("actual_tokens must be an integer, not bool")
            if actual_tokens < 0:
                raise AIConfigurationError("actual_tokens must be a non-negative integer")

        with self._lock:
            if reservation_id not in self._reservations:
                raise AIConfigurationError(
                    f"Unknown or duplicate reservation_id '{reservation_id}'"
                )

            reserved = self._reservations.pop(reservation_id)
            self._reserved_tokens = max(0, self._reserved_tokens - reserved)

            if failed and (actual_tokens is None or actual_tokens == 0):
                # Call failed without tokens used; release reservation
                return

            # Determine final tokens to record
            final_tokens = actual_tokens if actual_tokens is not None else reserved
            # Accurately record actual token usage to prevent silently underreporting external spend
            self._tokens_count += final_tokens

            if self._tokens_count > self.max_total_tokens:
                raise AIPolicyLimitError(
                    f"Workflow AI total token limit ({self.max_total_tokens} tokens) exceeded on completion with {final_tokens} tokens (total used: {self._tokens_count})",
                    limit_name="max_total_tokens",
                    actual_value=self._tokens_count,
                    limit_value=self.max_total_tokens,
                )

    def record_call(self, tokens_used: int = 0) -> None:
        """Record an AI call and its token consumption against the budget.

        Raises:
            AIConfigurationError: If input values are invalid (negative, non-int, bool).
            AIPolicyLimitError: If call count or token budget is exceeded.
                Counters are guaranteed NOT to mutate when a call is rejected.
        """
        if isinstance(tokens_used, bool):
            raise AIConfigurationError("tokens_used must be an integer, not bool")
        if not isinstance(tokens_used, int):
            raise AIConfigurationError("tokens_used must be an integer")
        if tokens_used < 0:
            raise AIConfigurationError("tokens_used must be a non-negative integer")

        with self._lock:
            if self._calls_count + 1 > self.max_ai_calls:
                raise AIPolicyLimitError(
                    f"Workflow AI call limit ({self.max_ai_calls} calls) exceeded",
                    limit_name="max_workflow_ai_calls",
                    actual_value=self._calls_count + 1,
                    limit_value=self.max_ai_calls,
                )

            if self._tokens_count + self._reserved_tokens + tokens_used > self.max_total_tokens:
                raise AIPolicyLimitError(
                    f"Workflow AI total token limit ({self.max_total_tokens} tokens) exceeded",
                    limit_name="max_total_tokens",
                    actual_value=self._tokens_count + self._reserved_tokens + tokens_used,
                    limit_value=self.max_total_tokens,
                )

            self._calls_count += 1
            self._tokens_count += tokens_used
