"""Tests for M0-014A QA Corrections."""

import os
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from flowforge.ai.providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    TokenUsage,
)
from flowforge.ai.safety import ExecutionBudgetTracker
from flowforge.processing.models import ProcessingJob
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_step,
    execute_workflow_pipeline_with_schema,
    validate_step_definitions,
)
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
from flowforge.workflows.tasks import (
    execute_workflow_run_task,
    resolve_workflow_llm_providers,
)

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


# --- 1. Schema & Validation Hardening (RED) ---


def test_temperature_rejects_nan_and_inf():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "temperature": float("nan"),
    }
    with pytest.raises(WorkflowExecutionError, match="temperature"):
        validate_step_definitions([step])

    step_inf = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
        "temperature": float("inf"),
    }
    with pytest.raises(WorkflowExecutionError, match="temperature"):
        validate_step_definitions([step_inf])


def test_template_detects_unmatched_or_malformed_braces():
    malformed_templates = [
        "Hello {{name",
        "Hello name}}",
        "Hello {{}}",
        "Hello {{   }}",
        "Hello {{ bad-name }}",
        "Hello {{ name }} and {{ broken",
    ]
    for tmpl in malformed_templates:
        step = {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3.2",
            "prompt_template": tmpl,
            "output_column": "summary",
        }
        with pytest.raises(WorkflowExecutionError, match="template"):
            validate_step_definitions([step])


def test_pipeline_rejects_mixed_providers_if_unsupported():
    steps = [
        {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3.2",
            "prompt_template": "Hello {{name}}",
            "output_column": "step1",
        },
        {
            "op": "llm_generate",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_template": "Analyze {{step1}}",
            "output_column": "step2",
        },
    ]
    # At definition validation time, or worker resolution time
    validated = validate_step_definitions(steps)
    assert len(validated) == 2


# --- 2. Budget Tracking Hardening (RED) ---


def test_budget_tracker_records_exactly_one_call_per_invocation():
    tracker = ExecutionBudgetTracker(max_ai_calls=2, max_total_tokens=10_000)
    provider = MockLLMProvider(responses=["Resp 1", "Resp 2"])

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
    }
    data = [{"name": "User 1"}, {"name": "User 2"}]

    # Exactly 2 rows and max_ai_calls=2: MUST succeed without double count
    execute_step(data, step, llm_provider=provider, budget_tracker=tracker)
    assert tracker.calls_count == 2
    assert tracker.tokens_count == 50  # 25 tokens * 2


def test_budget_tracker_stops_before_extra_network_call():
    tracker = ExecutionBudgetTracker(max_ai_calls=1, max_total_tokens=10_000)
    provider = MockLLMProvider(responses=["Resp 1", "Resp 2"])

    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3.2",
        "prompt_template": "Hello {{name}}",
        "output_column": "summary",
    }
    data = [{"name": "User 1"}, {"name": "User 2"}]

    # Second row must fail on budget limit BEFORE provider generate is invoked
    with pytest.raises(WorkflowExecutionError, match="call limit"):
        execute_step(data, step, llm_provider=provider, budget_tracker=tracker)

    assert len(provider.calls) == 1
    assert tracker.calls_count == 1


def test_budget_enforced_across_multiple_steps_in_pipeline():
    tracker = ExecutionBudgetTracker(max_ai_calls=3, max_total_tokens=10_000)
    provider = MockLLMProvider(responses=["A1", "A2", "B1"])

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
    data = [{"name": "User 1"}, {"name": "User 2"}]

    # Step 1 uses 2 calls. Step 2 row 1 uses 1 call (total 3). Step 2 row 2 exceeds budget (call 4)
    with pytest.raises(WorkflowExecutionError, match="call limit"):
        execute_workflow_pipeline_with_schema(
            data, steps, llm_provider=provider, budget_tracker=tracker
        )

    assert len(provider.calls) == 3


def test_cumulative_ai_rows_across_steps_bounded():
    # If workflow has two AI steps, total rows across all AI operations cannot exceed MAX_AI_ROWS_PER_WORKFLOW
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
    data = [{"name": f"User {i}"} for i in range(30)]  # 30 * 2 = 60 calls > 50
    provider = MockLLMProvider()

    with pytest.raises(WorkflowExecutionError, match="exceeds max allowed"):
        execute_workflow_pipeline_with_schema(data, steps, llm_provider=provider)


# --- 3. Multi-Provider & Hosted Billing Safeguards (RED) ---


def test_resolve_providers_requires_hosted_opt_in():
    steps = [
        {
            "op": "llm_generate",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_template": "Hello {{name}}",
            "output_column": "summary",
        }
    ]
    with patch.dict(
        os.environ,
        {
            "FLOWFORGE_ENABLE_AI_WORKFLOW_STEPS": "true",
            "FLOWFORGE_OPENAI_API_KEY": "sk-realapikey12345678",
            "FLOWFORGE_ALLOW_HOSTED_AI_BILLING": "false",  # Hosted billing NOT opted in
        },
    ):
        providers = resolve_workflow_llm_providers(steps)
        assert providers.get("openai") is None  # Must fail closed without explicit hosted opt-in


def test_multi_provider_workflow_binds_per_step():
    steps = [
        {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3.2",
            "prompt_template": "Hello {{name}}",
            "output_column": "col1",
        },
        {
            "op": "llm_generate",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_template": "Analyze {{col1}}",
            "output_column": "col2",
        },
    ]
    ollama_mock = MockLLMProvider(responses=["Ollama Resp"])
    openai_mock = MockLLMProvider(responses=["OpenAI Resp"])

    providers_map = {"ollama": ollama_mock, "openai": openai_mock}

    data = [{"name": "Alice"}]
    output, audits, schema = execute_workflow_pipeline_with_schema(
        data, steps, llm_providers=providers_map
    )

    assert len(ollama_mock.calls) == 1
    assert len(openai_mock.calls) == 1
    assert output[0]["col1"] == "Ollama Resp"
    assert output[0]["col2"] == "OpenAI Resp"


# --- 4. Worker Logging & Error Summary Redaction (RED) ---


@pytest.mark.django_db
def test_worker_redacts_malicious_provider_exception_with_causes(caplog):
    user = User.objects.create_user(username="malicious_user", password="password")
    job = ProcessingJob.objects.create(
        owner=user,
        state=ProcessingJob.State.COMPLETED,
        output_data=[{"name": "Alice"}],
        column_names=["name"],
    )
    wf_def = WorkflowDefinition.objects.create(
        owner=user,
        name="Error Redaction Workflow",
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

    cause_exc = Exception(
        "Underlying socket error to http://admin:supersecret@10.0.0.1:11434 with key sk-ant-secret12345678"
    )
    malicious_exc = ProviderError("External provider failure")
    malicious_exc.__cause__ = cause_exc

    with (
        patch.dict(os.environ, {"FLOWFORGE_ENABLE_AI_WORKFLOW_STEPS": "true"}),
        patch("flowforge.ai.providers.OllamaProvider.generate", side_effect=malicious_exc),
    ):
        execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.FAILED

    # Check error_summary in DB
    assert "supersecret" not in run.error_summary
    assert "sk-ant" not in run.error_summary
    assert "10.0.0.1" not in run.error_summary

    # Check Celery worker logs
    logged_text = caplog.text
    assert "supersecret" not in logged_text
    assert "sk-ant" not in logged_text
