"""RED tests for Milestone M0-014C: True Atomic Budget Reservation, Concurrency & Celery Task Secrecy."""

import concurrent.futures
import threading
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from flowforge.ai.providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from flowforge.ai.safety import ExecutionBudgetTracker
from flowforge.ai.safety.exceptions import AIPolicyLimitError
from flowforge.processing.models import ProcessingJob
from flowforge.workflows.engine import (
    execute_step,
)
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
from flowforge.workflows.tasks import execute_workflow_run_task

User = get_user_model()


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


# --- 1. True Atomic Pre-Call Budget Reservation API ---


def test_budget_tracker_public_atomic_reservation_and_completion_contract():
    """ExecutionBudgetTracker must have public thread-safe reserve_call and complete_call methods."""
    tracker = ExecutionBudgetTracker(max_ai_calls=2, max_total_tokens=100)

    # 1. Successful reservation
    res_id = tracker.reserve_call(estimated_tokens=40)
    assert res_id is not None
    assert tracker.calls_count == 1
    assert tracker.reserved_tokens == 40
    assert tracker.effective_tokens_count == 40

    # 2. Complete call with actual usage lower than reserved
    tracker.complete_call(res_id, actual_tokens=25)
    assert tracker.calls_count == 1
    assert tracker.reserved_tokens == 0
    assert tracker.tokens_count == 25
    assert tracker.effective_tokens_count == 25

    # 3. Complete failed call (actual_tokens=0, failed=True) - call attempt remains consumed, reserved released
    res_id2 = tracker.reserve_call(estimated_tokens=40)
    tracker.complete_call(res_id2, actual_tokens=0, failed=True)
    assert tracker.calls_count == 2
    assert tracker.reserved_tokens == 0
    assert tracker.tokens_count == 25

    # 4. Further reservation must fail atomically because call limit (2) is reached
    with pytest.raises(AIPolicyLimitError, match="call limit"):
        tracker.reserve_call(estimated_tokens=10)


def test_budget_tracker_concurrent_threads_competing_under_lock():
    """Multiple threads competing for the last available call/token budget must never exceed budget."""
    tracker = ExecutionBudgetTracker(max_ai_calls=5, max_total_tokens=200)

    success_count = 0
    rejected_count = 0
    lock = threading.Lock()

    def worker():
        nonlocal success_count, rejected_count
        try:
            res_id = tracker.reserve_call(estimated_tokens=40)
            # Simulate slight work
            tracker.complete_call(res_id, actual_tokens=40)
            with lock:
                success_count += 1
        except AIPolicyLimitError:
            with lock:
                rejected_count += 1

    # 20 concurrent threads trying to grab a budget that only supports 5 calls (5 * 40 = 200 tokens)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(20)]
        concurrent.futures.wait(futures)

    assert success_count == 5
    assert rejected_count == 15
    assert tracker.calls_count == 5
    assert tracker.tokens_count == 200
    assert tracker.reserved_tokens == 0


def test_engine_does_not_access_private_tracker_attributes():
    """Engine must interact with budget_tracker only via public methods, never accessing _lock or _calls_count."""
    import src.flowforge.workflows.engine as engine_module

    with open(engine_module.__file__, encoding="utf-8") as f:
        content = f.read()

    # Must NOT touch private fields on budget_tracker
    assert "budget_tracker._lock" not in content
    assert "budget_tracker._calls_count" not in content
    assert "budget_tracker._tokens_count" not in content


# --- 2. Conservative Token Bounds & Missing Usage Handling ---


def test_engine_handles_missing_provider_usage_conservatively():
    """If provider returns no token usage in response, conservative reservation must be retained without crashing or decrementing negative."""

    class NoUsageMockProvider(LLMProvider):
        def __init__(self):
            super().__init__(base_url="http://127.0.0.1:11434", allow_loopback=True)

        def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                text="output with no usage reported", model=request.model, usage=None
            )

    tracker = ExecutionBudgetTracker(max_ai_calls=2, max_total_tokens=10_000)
    provider = NoUsageMockProvider()

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "max_output_tokens": 100,
    }
    data = [{"name": "User 1"}]

    execute_step(data, step, llm_provider=provider, budget_tracker=tracker)
    assert tracker.calls_count == 1
    # Should conservatively charge at least the estimated tokens rather than 0
    assert tracker.tokens_count > 0


# --- 3. Celery Task Exception Secrecy ---


@pytest.mark.django_db
def test_worker_unexpected_exception_logging_never_interpolates_str_exc(caplog):
    """Unexpected exceptions in Celery worker must NEVER log str(exc) with unknown format secrets."""
    user = User.objects.create_user(username="celery_secrecy_user", password="password")
    job = ProcessingJob.objects.create(
        owner=user,
        state=ProcessingJob.State.COMPLETED,
        output_data=[{"name": "Alice"}],
        column_names=["name"],
    )
    wf_def = WorkflowDefinition.objects.create(
        owner=user,
        name="Task Secrecy Workflow",
        steps=[
            {
                "op": "llm_generate",
                "provider": "ollama",
                "model": "llama3.2",
                "prompt_template": "Hello {{name}}",
                "output_column": "greeting",
            }
        ],
    )
    run = WorkflowRun.objects.create(
        owner=user,
        workflow_definition=wf_def,
        source_job=job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=wf_def.steps,
    )

    arbitrary_unknown_secret = "XYZZY_TOPSECRET_TOKEN_99999"
    internal_crash = RuntimeError(
        f"Database crash on {arbitrary_unknown_secret} with payload Alice"
    )

    with (
        patch("flowforge.workflows.tasks.execute_workflow_pipeline", side_effect=internal_crash),
    ):
        execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED

    # Check error_summary in DB
    assert arbitrary_unknown_secret not in run.error_summary
    assert "Internal workflow execution error occurred." in run.error_summary

    # Check Celery worker logs
    logged_text = caplog.text
    assert arbitrary_unknown_secret not in logged_text
