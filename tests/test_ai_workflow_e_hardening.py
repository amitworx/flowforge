"""RED tests for Milestone M0-014E: Token-Budget Truthfulness and Completion Safety."""

import contextlib
import threading

import pytest

from flowforge.ai.providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderTimeoutError,
    TokenUsage,
)
from flowforge.ai.safety import ExecutionBudgetTracker
from flowforge.ai.safety.exceptions import AIPolicyLimitError
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


# --- 1. Token Estimation Truthfulness & Heuristic Clarity ---


def test_estimate_prompt_tokens_adversarial_utf8_and_framing():
    """_estimate_prompt_tokens is an internal conservative heuristic, accounts for user + system framing.

    Verifies that adversarial UTF-8 (such as 4-byte astral characters, emojis, combining diacritics)
    and system framing overhead are accounted for in the estimate.
    """
    system_prompt = "You are a helpful assistant."
    adversarial_prompt = "🔥" * 50 + "a\u0300\u0301\u0302" * 20  # Emojis and combining characters

    est = _estimate_prompt_tokens(adversarial_prompt, system_instruction=system_prompt)

    # Must include system framing overhead (>= 10 tokens) and account for bytes/chars
    assert est > len(adversarial_prompt.encode("utf-8")) // 2
    # Framing is present
    est_without_sys = _estimate_prompt_tokens(adversarial_prompt)
    assert est > est_without_sys


# --- 2. Observable Overrun Accounting in ExecutionBudgetTracker ---


def test_complete_call_accurate_overrun_tracking():
    """complete_call records true actual usage (never silently clipping spend) and exposes overrun status.

    Test scenario: reserved=10, actual=120, limit=100.
    1. Must raise AIPolicyLimitError on overrun.
    2. Tracker must accurately reflect actual observable usage (120 tokens, not clipped to 100).
    3. Tracker must flag has_overrun as True and tokens_overrun = 20.
    4. Any subsequent reserve_call or record_call must be blocked fail-closed.
    """
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=100)
    res_id = tracker.reserve_call(estimated_tokens=10)

    # Actual tokens reported = 120, which overruns the limit of 100
    with pytest.raises(AIPolicyLimitError, match="total token limit"):
        tracker.complete_call(res_id, actual_tokens=120)

    # Observable actual usage must be truthfully recorded as 120, NOT silently capped at 100!
    assert tracker.tokens_count == 120
    assert tracker.has_overrun is True
    assert tracker.tokens_overrun == 20

    # Subsequent reserve_call must fail closed immediately
    with pytest.raises(AIPolicyLimitError):
        tracker.reserve_call(estimated_tokens=5)

    # Subsequent record_call must fail closed immediately
    with pytest.raises(AIPolicyLimitError):
        tracker.record_call(tokens_used=5)


def test_concurrent_overrun_safety_under_race():
    """Racing multiple completions where one or more cause an overrun maintains accurate tracking."""
    tracker = ExecutionBudgetTracker(max_ai_calls=10, max_total_tokens=200)
    res_ids = [tracker.reserve_call(estimated_tokens=20) for _ in range(5)]

    def complete_worker(res_id, actual):
        with contextlib.suppress(AIPolicyLimitError):
            tracker.complete_call(res_id, actual_tokens=actual)

    # Race completions: 4 with 30 tokens (120 total), 1 with 150 tokens (overrun)
    threads = []
    for i, res_id in enumerate(res_ids):
        act = 150 if i == 0 else 30
        t = threading.Thread(target=complete_worker, args=(res_id, act))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Total tokens count must equal sum of actual tokens completed (150 + 4*30 = 270)
    assert tracker.tokens_count == 270
    assert tracker.has_overrun is True
    assert tracker.tokens_overrun == 70
    assert tracker.reserved_tokens == 0


# --- 3. Failed Sent Requests & Token Exposure Accounting ---


def test_failed_sent_request_retains_full_reserved_exposure_when_usage_unknown():
    """When a dispatched call fails with network/provider error, keep full reserved exposure charged.

    Because the remote provider may have generated output (consuming both prompt and completion tokens)
    before the connection dropped or timed out, the tracker must retain the full reserved tokens
    to prevent underreporting external spend.
    """
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=1000)
    provider = MockLLMProvider(exception_to_raise=ProviderTimeoutError("Gateway timeout"))

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Analyze: {{input_text}}",
        "output_column": "analysis",
        "max_output_tokens": 150,
    }
    data = [{"input_text": "sample data"}]

    with pytest.raises(WorkflowExecutionError, match="LLM generation failed"):
        execute_step(
            data,
            step,
            schema=["input_text"],
            llm_provider=provider,
            budget_tracker=tracker,
        )

    # Call attempt was consumed
    assert tracker.calls_count == 1
    # Full reserved exposure (estimated prompt tokens + max_output_tokens) must be retained
    # because remote provider may have processed prompt and output before timeout!
    assert tracker.tokens_count > 150
    assert tracker.reserved_tokens == 0


def test_pre_network_rejection_does_not_consume_tokens():
    """If a step is rejected before dispatch (e.g. prompt safety limit violation), no tokens are charged."""
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=1000)
    provider = MockLLMProvider()

    # Very large prompt exceeding prompt char limit
    large_prompt = "x" * 200_000
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "{{input_text}}",
        "output_column": "analysis",
        "max_output_tokens": 100,
    }
    data = [{"input_text": large_prompt}]

    with pytest.raises(WorkflowExecutionError, match="Prompt length"):
        execute_step(
            data,
            step,
            schema=["input_text"],
            llm_provider=provider,
            budget_tracker=tracker,
        )

    # Rejected pre-network: 0 calls made, 0 tokens charged, 0 reserved
    assert tracker.calls_count == 0
    assert tracker.tokens_count == 0
    assert tracker.reserved_tokens == 0
