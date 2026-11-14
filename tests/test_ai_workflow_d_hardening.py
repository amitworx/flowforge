"""RED tests for Milestone M0-014D: Final Budget Reservation Invariants, Rejection & Cost Safety."""

import concurrent.futures
import threading

import pytest

from flowforge.ai.providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderConnectionError,
    TokenUsage,
)
from flowforge.ai.safety import ExecutionBudgetTracker
from flowforge.ai.safety.exceptions import AIConfigurationError, AIPolicyLimitError
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    _estimate_prompt_tokens,
    execute_step,
)


class MockLLMProvider(LLMProvider):
    def __init__(self, responses=None, exception_to_raise=None):
        super().__init__(base_url="http://127.0.0.1:11434", allow_loopback=True)
        self.responses = list(responses or ["default output"])
        self.exception_to_raise = exception_to_raise
        self.calls = []
        self._lock = threading.Lock()

    def generate(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.calls.append(request)
        if self.exception_to_raise:
            raise self.exception_to_raise
        with self._lock:
            content = self.responses.pop(0) if self.responses else "default output"
        return LLMResponse(
            text=content,
            model=request.model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        )


# --- 1. ExecutionBudgetTracker Invariants on Complete & Record ---


def test_complete_call_fails_closed_when_actual_tokens_exceed_max_total():
    """If actual_tokens > reserved and total exceeds max_total_tokens, must raise AIPolicyLimitError and not overcommit."""
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=100)
    res_id = tracker.reserve_call(estimated_tokens=10)

    # Actual tokens reported = 120, which exceeds max_total_tokens (100)
    with pytest.raises(AIPolicyLimitError, match="total token limit"):
        tracker.complete_call(res_id, actual_tokens=120)

    # Tracker state must not record tokens beyond limit
    assert tracker.tokens_count <= 100


def test_complete_call_rejects_unknown_or_duplicate_reservation_id():
    """Attempting to complete with an unknown or already-completed reservation ID must fail without altering state."""
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=100)
    res_id = tracker.reserve_call(estimated_tokens=20)
    tracker.complete_call(res_id, actual_tokens=15)

    initial_tokens = tracker.tokens_count
    initial_calls = tracker.calls_count

    # Duplicate completion with same ID
    with pytest.raises(AIConfigurationError, match="Unknown or duplicate reservation_id"):
        tracker.complete_call(res_id, actual_tokens=15)

    # Nonexistent reservation ID
    with pytest.raises(AIConfigurationError, match="Unknown or duplicate reservation_id"):
        tracker.complete_call("non-existent-uuid-12345", actual_tokens=10)

    assert tracker.tokens_count == initial_tokens
    assert tracker.calls_count == initial_calls


def test_record_call_accounts_for_reserved_tokens_to_prevent_overcommit():
    """record_call must verify current tokens + reserved_tokens + tokens_used <= max_total_tokens."""
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=100)
    # Reserve 60 tokens
    _res_id = tracker.reserve_call(estimated_tokens=60)

    # record_call attempting to add 50 tokens (60 reserved + 50 = 110 > 100) must fail
    with pytest.raises(AIPolicyLimitError, match="total token limit"):
        tracker.record_call(tokens_used=50)

    assert tracker.tokens_count == 0
    assert tracker.reserved_tokens == 60


def test_concurrent_racing_reserve_and_complete():
    """Concurrent threads racing reserve_call, complete_call, and record_call must never exceed limits."""
    tracker = ExecutionBudgetTracker(max_ai_calls=10, max_total_tokens=500)
    successes = 0
    rejections = 0
    lock = threading.Lock()

    def worker(i):
        nonlocal successes, rejections
        try:
            if i % 2 == 0:
                res = tracker.reserve_call(estimated_tokens=50)
                tracker.complete_call(res, actual_tokens=45)
            else:
                tracker.record_call(tokens_used=45)
            with lock:
                successes += 1
        except AIPolicyLimitError:
            with lock:
                rejections += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(25)]
        concurrent.futures.wait(futures)

    assert tracker.calls_count <= 10
    assert tracker.tokens_count <= 500
    assert successes <= 10
    assert successes + rejections == 25


# --- 2. Conservative Token Estimation Policy ---


def test_estimate_prompt_tokens_includes_utf8_bytes_and_system_overhead():
    """_estimate_prompt_tokens must be conservative: account for multi-byte UTF-8 and system instruction."""
    # Plain ASCII
    ascii_prompt = "Hello world"
    ascii_est = _estimate_prompt_tokens(ascii_prompt, system_instruction="System prompt")

    # Multi-byte UTF-8 characters (e.g. CJK or emojis take 3-4 bytes each and tokenize to multiple tokens)
    unicode_prompt = "こんにちは世界！🚀"
    unicode_est = _estimate_prompt_tokens(unicode_prompt, system_instruction="System prompt")

    # The estimate must be strictly >= length / 2 and account for multi-byte UTF-8 bytes
    assert ascii_est >= len(ascii_prompt) // 2
    assert unicode_est >= len(unicode_prompt.encode("utf-8")) // 2


# --- 3. Failed Provider Call Accounting ('may have been sent') ---


def test_failed_provider_call_retains_conservative_token_exposure():
    """When a provider request fails during or after generate (network/server error), conservative prompt tokens are retained."""
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=1000)
    provider = MockLLMProvider(exception_to_raise=ProviderConnectionError("Remote failure"))

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "max_output_tokens": 100,
    }
    data = [{"name": "User 1"}]

    with pytest.raises(WorkflowExecutionError):
        execute_step(data, step, llm_provider=provider, budget_tracker=tracker)

    # 1 call was attempted and failed
    assert tracker.calls_count == 1
    # Tokens count must NOT be silently reset to 0: remote server may have processed prompt
    assert tracker.tokens_count > 0
    assert tracker.reserved_tokens == 0
