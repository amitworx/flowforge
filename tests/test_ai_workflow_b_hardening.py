"""RED tests for Milestone M0-014B: Budget Reservation, Exception Secrecy, Provider Matching, and Audits."""

import os
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from flowforge.ai.providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderConnectionError,
    ProviderError,
    TokenUsage,
)
from flowforge.ai.safety import ExecutionBudgetTracker
from flowforge.processing.models import ProcessingJob
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_step,
    execute_workflow_pipeline_with_schema,
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

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.exception_to_raise:
            raise self.exception_to_raise
        content = self.responses.pop(0) if self.responses else "default output"
        return LLMResponse(
            text=content,
            model=request.model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        )


# --- 1. Atomic Pre-Call Budget Reservation ---


def test_budget_reservation_token_limit_exhausted_before_network_call():
    """If estimated tokens exceed budget before call, provider.generate must NOT be invoked."""
    # max_total_tokens = 50. Prompt estimate + max_output_tokens (e.g. 100) > 50 -> must reject before call
    tracker = ExecutionBudgetTracker(max_ai_calls=10, max_total_tokens=50)
    provider = MockLLMProvider(responses=["Resp 1"])

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "max_output_tokens": 100,  # Conservative allocation 100 > 50 tokens available
    }
    data = [{"name": "User 1"}]

    with pytest.raises(WorkflowExecutionError, match="budget"):
        execute_step(data, step, llm_provider=provider, budget_tracker=tracker)

    # Provider network call must NEVER have been called!
    assert len(provider.calls) == 0


def test_budget_reservation_failed_calls_count_towards_call_budget():
    """A failed call must still consume the reserved call attempt so exhausted budget stops further calls."""
    tracker = ExecutionBudgetTracker(max_ai_calls=1, max_total_tokens=10_000)
    failing_provider = MockLLMProvider(
        exception_to_raise=ProviderConnectionError("Connection failed")
    )

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
    }
    data = [{"name": "User 1"}, {"name": "User 2"}]

    with pytest.raises(WorkflowExecutionError):
        execute_step(data, step, llm_provider=failing_provider, budget_tracker=tracker)

    # 1 call made, failed, but 1 call attempt reserved/consumed
    assert len(failing_provider.calls) == 1
    assert tracker.calls_count == 1


# --- 2. Sensitive Provider Exception Secrecy ---


def test_provider_exception_uses_generic_allowlisted_reason_no_raw_str():
    """Raw str(exc) with arbitrary secret shape and user prompt must NEVER leak into WorkflowExecutionError."""
    leak_secret = "my-custom-unformatted-secret-99998888"
    raw_prompt_leak = "SecretInternalData:Alice"
    leaking_exc = ProviderError(
        f"Remote failure on {leak_secret} while executing {raw_prompt_leak}"
    )

    provider = MockLLMProvider(exception_to_raise=leaking_exc)
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
    }
    data = [{"name": "Alice"}]

    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step(data, step, llm_provider=provider)

    err_msg = str(exc_info.value)
    assert leak_secret not in err_msg
    assert raw_prompt_leak not in err_msg
    # Must use generic allowlisted message
    assert "LLM generation failed" in err_msg


@pytest.mark.django_db
def test_worker_logging_and_audits_never_contain_raw_provider_exception(caplog):
    user = User.objects.create_user(username="test_secrecy_user", password="password")
    job = ProcessingJob.objects.create(
        owner=user,
        state=ProcessingJob.State.COMPLETED,
        output_data=[{"name": "Alice"}],
        column_names=["name"],
    )
    wf_def = WorkflowDefinition.objects.create(
        owner=user,
        name="Secrecy Workflow",
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

    unformatted_secret = "custom_secret_key_abcdef_987654"
    raw_url = "http://internal-secret-service.local:9999/v1/models"
    raw_body = "payload={'secret': '12345'}"
    malicious_exc = ProviderError(
        f"HTTP error connecting to {raw_url}: {raw_body} with {unformatted_secret}"
    )

    with (
        patch.dict(os.environ, {"FLOWFORGE_ENABLE_AI_WORKFLOW_STEPS": "true"}),
        patch("flowforge.ai.providers.OllamaProvider.generate", side_effect=malicious_exc),
    ):
        execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED

    # Check error_summary in DB
    assert unformatted_secret not in run.error_summary
    assert raw_url not in run.error_summary
    assert raw_body not in run.error_summary

    # Check Celery worker logs
    logged_text = caplog.text
    assert unformatted_secret not in logged_text
    assert raw_url not in logged_text
    assert raw_body not in logged_text


# --- 3. Provider Fallback Fix ---


def test_reject_silent_provider_fallback_when_map_passed():
    """If llm_providers is passed and provider is missing from map, must NOT silently fall back to llm_provider."""
    step = {
        "op": "llm_generate",
        "provider": "openai",
        "model": "gpt-4o",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
    }
    data = [{"name": "Alice"}]
    ollama_provider = MockLLMProvider()
    fallback_provider = MockLLMProvider()

    # llm_providers only provides 'ollama', but step requests 'openai'
    # Even if llm_provider=fallback_provider is provided, it must FAIL, not silently run openai on fallback
    with pytest.raises(WorkflowExecutionError, match="No provider configured"):
        execute_step(
            data,
            step,
            llm_providers={"ollama": ollama_provider},
            llm_provider=fallback_provider,
        )


# --- 4. Cumulative AI Row Cap Step Audit ---


def test_cumulative_ai_row_cap_records_failed_step_audit():
    """When cumulative AI row cap is exceeded, WorkflowExecutionError must preserve completed and failed step audits."""
    steps = [
        {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3.2",
            "prompt_template": "Step 1: {{name}}",
            "output_column": "col1",
        },
        {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3.2",
            "prompt_template": "Step 2: {{col1}}",
            "output_column": "col2",
        },
    ]
    # 30 rows in step 1 = 30 AI rows (passes <= 50).
    # 30 rows in step 2 = 60 AI rows (fails > 50).
    data = [{"name": f"User {i}"} for i in range(30)]
    provider = MockLLMProvider(responses=[f"Resp {i}" for i in range(30)])

    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_workflow_pipeline_with_schema(data, steps, llm_provider=provider)

    err = exc_info.value
    assert err.step_index == 1
    assert len(err.completed_step_audit) == 1
    assert err.completed_step_audit[0]["status"] == "completed"
    assert err.failed_step_audit is not None
    assert err.failed_step_audit["step_index"] == 1
    assert err.failed_step_audit["status"] == "failed"
    assert "exceeds max allowed limit" in err.failed_step_audit["error_summary"]
