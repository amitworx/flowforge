"""RED tests for Milestone M0-014F: Final Reservation Lifecycle Fix."""

import threading

import pytest

from flowforge.ai.providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from flowforge.ai.safety import ExecutionBudgetTracker
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_step,
)

_DEFAULT_TOKEN_USAGE = TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25)


class MockLLMProviderWithUsage(LLMProvider):
    def __init__(
        self,
        responses=None,
        usage=_DEFAULT_TOKEN_USAGE,
        exception_to_raise=None,
    ):
        super().__init__(base_url="http://127.0.0.1:11434", allow_loopback=True)
        self.responses = list(responses or ["default output"])
        self.usage = usage
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
            usage=self.usage,
        )


def test_complete_call_overrun_in_execute_step_preserves_policy_limit_error():
    """When actual usage > limit in execute_step, WorkflowExecutionError must preserve AIPolicyLimitError,
    and NOT be masked by AIConfigurationError (duplicate reservation).

    Test setup:
    reserved = 10 + 50 = 60
    actual = 120 (total_tokens in usage)
    max_total_tokens = 100
    """
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=100)
    provider = MockLLMProviderWithUsage(
        usage=TokenUsage(prompt_tokens=20, completion_tokens=100, total_tokens=120)
    )

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "max_output_tokens": 50,
    }
    data = [{"name": "World"}]

    # Must raise WorkflowExecutionError indicating budget/policy limit, NOT duplicate reservation!
    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step(
            data,
            step,
            schema=["name"],
            llm_provider=provider,
            budget_tracker=tracker,
        )

    # Verify that duplicate reservation error was NOT raised / did not mask the overrun
    assert "Unknown or duplicate reservation_id" not in str(exc_info.value)
    assert "AIConfigurationError" not in str(exc_info.value)
    # The error should properly identify the budget overrun / policy violation
    assert (
        "AIPolicyLimitError" in str(exc_info.value)
        or "budget exceeded" in str(exc_info.value).lower()
    )

    # Budget tracker must truthfully reflect the 120 tokens and flag overrun
    assert tracker.tokens_count == 120
    assert tracker.has_overrun is True
    assert tracker.tokens_overrun == 20
    assert tracker.reserved_tokens == 0


def test_output_validation_error_after_provider_response_charges_actual_usage():
    """If provider succeeds and reports usage, but response output validation fails (e.g. max_output_chars exceeded),
    the reservation must be finalized with actual reported usage, NOT called twice and NOT discarded.
    """
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=1000)
    # Provider returns 50 tokens usage, but output is 20,000 chars (exceeding MAX_AI_OUTPUT_CHARS = 10,000)
    provider = MockLLMProviderWithUsage(
        responses=["A" * 20_000],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=40, total_tokens=50),
    )

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "max_output_tokens": 20,  # response tokens (40) will also violate safety limit output tokens!
    }
    data = [{"name": "World"}]

    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step(
            data,
            step,
            schema=["name"],
            llm_provider=provider,
            budget_tracker=tracker,
        )

    # Must NOT fail with duplicate reservation ID
    assert "Unknown or duplicate reservation_id" not in str(exc_info.value)
    # Reservation is finalized exactly once
    assert tracker.reserved_tokens == 0
    assert tracker.calls_count == 1
    # Actual known usage (50) was charged!
    assert tracker.tokens_count == 50


def test_missing_usage_on_success_finalizes_with_estimated_exposure():
    """If provider succeeds but does not provide usage info (None), reservation is finalized exactly once
    with estimated conservative exposure.
    """
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=1000)
    # Mock provider returning usage=None
    provider = MockLLMProviderWithUsage(
        responses=["Valid response"],
        usage=None,
    )

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "max_output_tokens": 50,
    }
    data = [{"name": "World"}]

    new_data = execute_step(
        data,
        step,
        schema=["name"],
        llm_provider=provider,
        budget_tracker=tracker,
    )

    assert len(new_data) == 1
    assert tracker.reserved_tokens == 0
    assert tracker.calls_count == 1
    # Charged estimated tokens (> 50 because prompt tokens + 50 max_output_tokens)
    assert tracker.tokens_count > 50
