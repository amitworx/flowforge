"""In-memory deterministic request budget tracker for single executions.

No global mutable state, thread-safe, and zero persistence/database dependencies.
"""

import threading

from src.flowforge.ai.safety.exceptions import AIPolicyLimitError


class ExecutionBudgetTracker:
    """Thread-safe, isolated, in-memory budget tracker for a single workflow or request execution."""

    def __init__(self, max_ai_calls: int = 50, max_total_tokens: int = 100_000) -> None:
        self.max_ai_calls = max_ai_calls
        self.max_total_tokens = max_total_tokens
        self._lock = threading.Lock()
        self._calls_count = 0
        self._tokens_count = 0

    @property
    def calls_count(self) -> int:
        with self._lock:
            return self._calls_count

    @property
    def tokens_count(self) -> int:
        with self._lock:
            return self._tokens_count

    def record_call(self, tokens_used: int = 0) -> None:
        """Record an AI call and its token consumption against the budget.

        Raises:
            AIPolicyLimitError: If call count or token budget is exceeded.
        """
        with self._lock:
            if self._calls_count + 1 > self.max_ai_calls:
                raise AIPolicyLimitError(
                    f"Workflow AI call limit ({self.max_ai_calls} calls) exceeded",
                    limit_name="max_workflow_ai_calls",
                    actual_value=self._calls_count + 1,
                    limit_value=self.max_ai_calls,
                )

            if self._tokens_count + tokens_used > self.max_total_tokens:
                raise AIPolicyLimitError(
                    f"Workflow AI total token limit ({self.max_total_tokens} tokens) exceeded",
                    limit_name="max_total_tokens",
                    actual_value=self._tokens_count + tokens_used,
                    limit_value=self.max_total_tokens,
                )

            self._calls_count += 1
            self._tokens_count += tokens_used
