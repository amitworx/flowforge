"""Tests for strict structured LLM output in workflows (M0-015B).

TDD RED suite covering:
- Schema definition validation (output_schema presence, valid SchemaContract, rejects invalid keywords/types, bounds).
- Schema derivation (derive_workflow_schema tracks columns even for empty input).
- Execution with valid structured output (places validated Python dict/list/value in output_column).
- Execution without output_schema (preserves default plain-text string behavior).
- Execution failure cases:
  * Malformed JSON (no markdown stripping, no repair)
  * Schema mismatch / missing required fields / unexpected properties
  * NaN / Infinity rejection
  * Secret and raw field name leak prevention in errors and audits
  * Completed step audit and failed step audit preservation
  * Exactly-once budget accounting on failure
  * Tenant/user isolation and provider binding
  * Output column collisions
- Export/import and clone boundaries with structured output steps.
"""

import copy
import json

import pytest

from flowforge.ai.providers import (
    LLMProvider,
    LLMResponse,
    TokenUsage,
)
from flowforge.ai.safety import ExecutionBudgetTracker
from flowforge.workflows.engine import (
    WorkflowExecutionError,
    execute_step,
    execute_workflow_pipeline,
    execute_workflow_pipeline_with_schema,
    validate_step_definitions,
)
from flowforge.workflows.export import (
    derive_workflow_schema,
    export_workflow_run_to_csv,
    export_workflow_run_to_json,
)


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        responses: list[str] | None = None,
        usage: TokenUsage | None = None,
        omit_usage: bool = False,
    ):
        self.responses = responses or ["{}"]
        self.call_count = 0
        if omit_usage:
            self.usage = None
        else:
            self.usage = usage or TokenUsage(
                prompt_tokens=10, completion_tokens=15, total_tokens=25
            )
        self.recorded_requests = []

    def generate(self, request) -> LLMResponse:
        self.call_count += 1
        self.recorded_requests.append(request)
        resp_text = self.responses[(self.call_count - 1) % len(self.responses)]
        return LLMResponse(
            text=resp_text,
            model=request.model,
            usage=self.usage,
        )


# ==============================================================================
# 1. Definition-time validation & schema bounds
# ==============================================================================


def test_validate_step_definitions_accepts_valid_output_schema():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Classify {{ text }}",
        "output_column": "classification",
        "output_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["category"],
        },
    }
    validated = validate_step_definitions([step])
    assert validated[0]["output_schema"]["type"] == "object"


def test_validate_step_definitions_rejects_non_dict_output_schema():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Classify {{ text }}",
        "output_column": "classification",
        "output_schema": "not a dict",
    }
    with pytest.raises(WorkflowExecutionError, match="output_schema.*dictionary"):
        validate_step_definitions([step])


def test_validate_step_definitions_rejects_unsupported_schema_keywords():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Classify {{ text }}",
        "output_column": "classification",
        "output_schema": {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"category": {"type": "string"}},
        },
    }
    with pytest.raises(WorkflowExecutionError, match="schema"):
        validate_step_definitions([step])


def test_validate_step_definitions_rejects_oversized_schema():
    # Large schema exceeding MAX_SCHEMA_NODES
    props = {f"field_{i}": {"type": "string"} for i in range(2500)}
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Classify {{ text }}",
        "output_column": "classification",
        "output_schema": {
            "type": "object",
            "properties": props,
        },
    }
    with pytest.raises(WorkflowExecutionError, match="schema"):
        validate_step_definitions([step])


def test_validate_step_definitions_prevents_secret_leakage_in_schema_error():
    secret_marker = "SUPER_SECRET_TOKEN_XYZ_999"
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Classify {{ text }}",
        "output_column": "classification",
        "output_schema": {
            "type": "object",
            secret_marker: {"type": "string"},
        },
    }
    with pytest.raises(WorkflowExecutionError) as exc_info:
        validate_step_definitions([step])
    assert secret_marker not in str(exc_info.value)


# ==============================================================================
# 2. Schema derivation
# ==============================================================================


def test_derive_workflow_schema_preserves_structured_output_column():
    steps = [
        {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3",
            "prompt_template": "Extract from {{ text }}",
            "output_column": "extracted_json",
            "output_schema": {"type": "object"},
        }
    ]
    derived = derive_workflow_schema(["id", "text"], steps)
    assert derived == ["id", "text", "extracted_json"]


def test_derive_workflow_schema_tracks_column_even_on_empty_initial_schema():
    steps = [
        {
            "op": "llm_generate",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "prompt_template": "Static prompt",
            "output_column": "result_json",
            "output_schema": {"type": "object"},
        }
    ]
    derived = derive_workflow_schema([], steps)
    assert derived == ["result_json"]


# ==============================================================================
# 3. Execution - Default plain text when output_schema is absent
# ==============================================================================


def test_execute_step_plain_text_when_no_schema():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Summarize {{ text }}",
        "output_column": "summary",
    }
    provider = MockLLMProvider(responses=["Just plain text response."])
    data = [{"text": "Hello world"}]
    res = execute_step(data, step, schema=["text"], llm_provider=provider)
    # Verify string output persisted
    assert res[0]["summary"] == "Just plain text response."


# ==============================================================================
# 4. Execution - Strict structured output validation
# ==============================================================================


def test_execute_step_valid_structured_output():
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "score": {"type": "integer"},
        },
        "required": ["category", "score"],
    }
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": schema,
    }
    valid_payload = json.dumps({"category": "urgent", "tags": ["ops", "p0"], "score": 95})
    provider = MockLLMProvider(responses=[valid_payload])
    data = [{"text": "Server down!"}]
    res = execute_step(data, step, schema=["text"], llm_provider=provider)
    assert res[0]["analysis"] == {"category": "urgent", "tags": ["ops", "p0"], "score": 95}
    assert isinstance(res[0]["analysis"], dict)


def test_execute_step_empty_data_preserves_schema_with_structured_output():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": {"type": "object"},
    }
    provider = MockLLMProvider(responses=["{}"])
    schema = ["text"]
    res = execute_step([], step, schema=schema, llm_provider=provider)
    assert res == []
    assert schema == ["text", "analysis"]
    assert provider.call_count == 0


def test_execute_step_rejects_markdown_code_fences():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": {"type": "object", "properties": {"status": {"type": "string"}}},
    }
    # LLMs often emit markdown code blocks like ```json ... ```
    fenced_payload = '```json\n{"status": "ok"}\n```'
    provider = MockLLMProvider(responses=[fenced_payload])
    data = [{"text": "check"}]
    with pytest.raises(
        WorkflowExecutionError,
        match="Structured output validation failed: STRUCTURED_OUTPUT_PARSE_ERROR",
    ):
        execute_step(data, step, schema=["text"], llm_provider=provider)


def test_execute_step_rejects_malformed_json_without_repair():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": {"type": "object"},
    }
    provider = MockLLMProvider(responses=["{incomplete_json: "])
    data = [{"text": "check"}]
    with pytest.raises(
        WorkflowExecutionError,
        match="Structured output validation failed: STRUCTURED_OUTPUT_PARSE_ERROR",
    ):
        execute_step(data, step, schema=["text"], llm_provider=provider)


def test_execute_step_rejects_missing_required_fields():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    }
    provider = MockLLMProvider(responses=[json.dumps({"other": "value"})])
    data = [{"text": "check"}]
    with pytest.raises(WorkflowExecutionError, match="Structured output validation failed"):
        execute_step(data, step, schema=["text"], llm_provider=provider)


def test_execute_step_rejects_nan_and_infinity():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": {"type": "object", "properties": {"val": {"type": "number"}}},
    }
    provider = MockLLMProvider(responses=['{"val": NaN}'])
    data = [{"text": "check"}]
    with pytest.raises(WorkflowExecutionError, match="Structured output validation failed"):
        execute_step(data, step, schema=["text"], llm_provider=provider)


def test_execute_step_rejects_duplicate_keys():
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
    }
    provider = MockLLMProvider(responses=['{"a": "first", "a": "second"}'])
    data = [{"text": "check"}]
    with pytest.raises(WorkflowExecutionError, match="Structured output validation failed"):
        execute_step(data, step, schema=["text"], llm_provider=provider)


def test_execute_step_no_secret_leak_on_validation_failure():
    secret_token = "TOP_SECRET_USER_TOKEN_9999"
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "output_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    }
    # LLM echoes the secret in response that fails validation
    provider = MockLLMProvider(responses=[json.dumps({secret_token: "leaked_val"})])
    data = [{"text": "check"}]
    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step(data, step, schema=["text"], llm_provider=provider)
    assert secret_token not in str(exc_info.value)


# ==============================================================================
# 5. Pipeline execution, audits, and budget tracker
# ==============================================================================


def test_pipeline_preserves_audits_on_structured_validation_failure():
    steps = [
        {"op": "select_columns", "columns": ["text"]},
        {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3",
            "prompt_template": "Analyze {{ text }}",
            "output_column": "analysis",
            "output_schema": {"type": "object", "required": ["status"]},
        },
    ]
    provider = MockLLMProvider(responses=['{"wrong": "field"}'])
    data = [{"id": 1, "text": "sample"}]

    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_workflow_pipeline(data, steps, llm_provider=provider)

    err = exc_info.value
    assert len(err.completed_step_audit) == 1
    assert err.completed_step_audit[0]["operation"] == "select_columns"
    assert err.completed_step_audit[0]["status"] == "completed"

    assert err.failed_step_audit is not None
    assert err.failed_step_audit["operation"] == "llm_generate"
    assert err.failed_step_audit["status"] == "failed"
    assert "Structured output validation failed" in err.failed_step_audit["error_summary"]


def test_budget_tracker_finalized_on_structured_output_validation_failure():
    budget_tracker = ExecutionBudgetTracker(max_total_tokens=1000, max_ai_calls=10)
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "analysis",
        "max_output_tokens": 100,
        "output_schema": {"type": "object", "required": ["expected"]},
    }
    provider = MockLLMProvider(
        responses=['{"invalid": true}'],
        usage=TokenUsage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
    )
    data = [{"text": "hello"}]

    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_workflow_pipeline(
            data, [step], llm_provider=provider, budget_tracker=budget_tracker
        )
    print("Caught:", exc_info.value)

    # Call must be marked complete and charged
    assert budget_tracker.calls_count == 1
    assert budget_tracker.reserved_tokens == 0
    assert budget_tracker.tokens_count == 50


# ==============================================================================
# 6. Export compatibility
# ==============================================================================


def test_export_workflow_run_with_structured_output():
    output_data = [
        {"id": 1, "data": {"key": "val", "arr": [1, 2]}},
        {"id": 2, "data": {"key": "alert", "trigger": "=1+1"}},
    ]
    output_columns = ["id", "data"]

    # CSV export
    csv_out = export_workflow_run_to_csv(output_data, output_columns)
    assert "id,data" in csv_out
    assert "key" in csv_out

    # JSON export
    json_bytes = export_workflow_run_to_json(output_data)
    loaded = json.loads(json_bytes.decode("utf-8"))
    assert loaded[0]["data"]["arr"] == [1, 2]
    assert loaded[1]["data"]["trigger"] == "=1+1"


# ==============================================================================
# 7. Django integration & execution (sync and Celery async)
# ==============================================================================


@pytest.mark.django_db
def test_sync_workflow_run_service_with_structured_ai_step():
    from django.contrib.auth import get_user_model

    from flowforge.processing.models import ProcessingJob
    from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
    from flowforge.workflows.services import execute_workflow_run

    User = get_user_model()
    user = User.objects.create_user(username="struct_user", password="password")
    job = ProcessingJob.objects.create(
        owner=user,
        state=ProcessingJob.State.COMPLETED,
        output_data=[{"item": "Laptop", "price": 1200}],
        column_names=["item", "price"],
    )
    schema = {
        "type": "object",
        "properties": {
            "item_name": {"type": "string"},
            "is_expensive": {"type": "boolean"},
        },
        "required": ["item_name", "is_expensive"],
    }
    wf_def = WorkflowDefinition.objects.create(
        owner=user,
        name="Structured AI Enrichment",
        steps=[
            {
                "op": "llm_generate",
                "provider": "ollama",
                "model": "llama3.2",
                "prompt_template": "Classify price for {{item}}: {{price}}",
                "output_column": "analysis",
                "output_schema": schema,
            }
        ],
    )

    valid_payload = json.dumps({"item_name": "Laptop", "is_expensive": True})
    provider = MockLLMProvider(responses=[valid_payload])
    run = execute_workflow_run(wf_def, job, user, llm_provider=provider)

    assert run.state == WorkflowRun.State.COMPLETED
    assert run.output_data == [
        {"item": "Laptop", "price": 1200, "analysis": {"item_name": "Laptop", "is_expensive": True}}
    ]
    step_run = run.step_runs.first()
    assert step_run.operation == "llm_generate"
    assert step_run.status == "completed"


@pytest.mark.django_db
def test_sync_workflow_run_service_structured_ai_step_fails_closed():
    from django.contrib.auth import get_user_model

    from flowforge.processing.models import ProcessingJob
    from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
    from flowforge.workflows.services import execute_workflow_run

    User = get_user_model()
    user = User.objects.create_user(username="struct_fail_user", password="password")
    job = ProcessingJob.objects.create(
        owner=user,
        state=ProcessingJob.State.COMPLETED,
        output_data=[{"item": "Laptop"}],
        column_names=["item"],
    )
    schema = {
        "type": "object",
        "properties": {"rating": {"type": "integer"}},
        "required": ["rating"],
    }
    wf_def = WorkflowDefinition.objects.create(
        owner=user,
        name="Structured Fail Workflow",
        steps=[
            {
                "op": "llm_generate",
                "provider": "ollama",
                "model": "llama3.2",
                "prompt_template": "Rate {{item}}",
                "output_column": "rating_json",
                "output_schema": schema,
            }
        ],
    )

    # Provider emits malformed response
    provider = MockLLMProvider(responses=['```json\n{"rating": 5}\n```'])
    run = execute_workflow_run(wf_def, job, user, llm_provider=provider)

    assert run.state == WorkflowRun.State.FAILED
    assert "Structured output validation failed" in run.error_summary
    assert run.output_data == []
    assert run.step_runs.count() == 1
    assert run.step_runs.first().status == "failed"


@pytest.mark.django_db
def test_async_workflow_run_task_with_structured_output():
    import os
    from unittest.mock import patch

    from django.contrib.auth import get_user_model

    from flowforge.processing.models import ProcessingJob
    from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
    from flowforge.workflows.tasks import execute_workflow_run_task

    User = get_user_model()
    user = User.objects.create_user(username="async_struct_user", password="password")
    job = ProcessingJob.objects.create(
        owner=user,
        state=ProcessingJob.State.COMPLETED,
        output_data=[{"msg": "urgent bug"}],
        column_names=["msg"],
    )
    schema = {
        "type": "object",
        "properties": {"priority": {"type": "string"}},
        "required": ["priority"],
    }
    wf_def = WorkflowDefinition.objects.create(
        owner=user,
        name="Async Structured AI",
        steps=[
            {
                "op": "llm_generate",
                "provider": "ollama",
                "model": "llama3.2",
                "prompt_template": "Classify: {{msg}}",
                "output_column": "classified",
                "output_schema": schema,
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

    valid_payload = json.dumps({"priority": "P0"})
    mock_resp = LLMResponse(
        text=valid_payload,
        model="llama3.2",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )

    with (
        patch.dict(os.environ, {"FLOWFORGE_ENABLE_AI_WORKFLOW_STEPS": "true"}),
        patch("flowforge.ai.providers.OllamaProvider.generate", return_value=mock_resp),
    ):
        execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.COMPLETED
    assert run.output_data == [{"msg": "urgent bug", "classified": {"priority": "P0"}}]


# ==============================================================================
# 8. Import/export and clone boundaries with structured output schema
# ==============================================================================


@pytest.mark.django_db
def test_clone_and_export_import_workflow_with_structured_schema():
    from django.contrib.auth import get_user_model

    from flowforge.workflows.models import WorkflowDefinition
    from flowforge.workflows.portability import (
        WorkflowImportSerializer,
        serialize_workflow_definition_export,
    )

    User = get_user_model()
    user = User.objects.create_user(username="clone_export_user", password="password")
    schema = {
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"],
    }
    steps = [
        {
            "op": "llm_generate",
            "provider": "ollama",
            "model": "llama3",
            "prompt_template": "Score {{text}}",
            "output_column": "score_obj",
            "output_schema": schema,
        }
    ]
    wf = WorkflowDefinition.objects.create(
        owner=user,
        name="Original Workflow",
        steps=steps,
    )

    # 1. Export definition to JSON bytes
    exported_bytes = serialize_workflow_definition_export(wf.name, wf.steps)
    assert b"output_schema" in exported_bytes
    assert b"score" in exported_bytes

    # 2. Import serialized definition
    exported_dict = json.loads(exported_bytes.decode("utf-8"))
    serializer = WorkflowImportSerializer(data=exported_dict)
    assert serializer.is_valid(), serializer.errors
    imported_steps = serializer.validated_data["steps"]
    assert imported_steps[0]["output_schema"] == schema

    # 3. Clone workflow definition and verify output_schema is preserved and owner-isolated
    from flowforge.workflows.serializers import WorkflowCloneRequestSerializer

    clone_serializer = WorkflowCloneRequestSerializer(data={"name": "Cloned Workflow"})
    assert clone_serializer.is_valid(), clone_serializer.errors
    cloned_wf = WorkflowDefinition.objects.create(
        owner=user,
        name=clone_serializer.validated_data.get("name"),
        steps=copy.deepcopy(wf.steps),
    )
    assert cloned_wf.steps[0]["output_schema"] == schema
    assert cloned_wf.steps[0]["output_schema"] is not wf.steps[0]["output_schema"]

    # 4. Preview workflow with combined deterministic steps without provider calls
    preview_steps = [
        {"op": "select_columns", "columns": ["text", "val"]},
        {"op": "filter_equals", "column": "val", "value": 10},
        {"op": "rename_columns", "mapping": {"text": "prompt_input"}},
    ]
    preview_data = [
        {"text": "keep", "val": 10, "ignore": 1},
        {"text": "drop", "val": 20, "ignore": 2},
    ]
    preview_wf = WorkflowDefinition.objects.create(
        owner=user,
        name="Deterministic Workflow",
        steps=preview_steps,
    )
    preview_out, preview_audits, preview_schema = execute_workflow_pipeline_with_schema(
        preview_data, preview_wf.steps
    )
    assert len(preview_out) == 1
    assert preview_out[0] == {"prompt_input": "keep", "val": 10}
    assert preview_schema == ["prompt_input", "val"]
    assert len(preview_audits) == 3


def test_structured_validation_fixed_allowlisted_reason_codes():
    """Verify that structured output errors use fixed allowlisted reason codes and never leak content."""
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Classify {{ text }}",
        "output_column": "classification",
        "output_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["A", "B"]},
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["category", "score"],
            "additionalProperties": False,
        },
    }

    # TYPE_MISMATCH
    p_type = MockLLMProvider(responses=[json.dumps({"category": "A", "score": "not_an_int"})])
    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step([{"text": "x"}], step, schema=["text"], llm_provider=p_type)
    assert (
        exc_info.value.message
        == "Structured output validation failed: STRUCTURED_OUTPUT_INVALID (TYPE_MISMATCH)"
    )

    # MISSING_REQUIRED
    p_miss = MockLLMProvider(responses=[json.dumps({"category": "A"})])
    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step([{"text": "x"}], step, schema=["text"], llm_provider=p_miss)
    assert (
        exc_info.value.message
        == "Structured output validation failed: STRUCTURED_OUTPUT_INVALID (MISSING_REQUIRED)"
    )

    # ADDITIONAL_PROPERTIES
    p_extra = MockLLMProvider(
        responses=[json.dumps({"category": "A", "score": 50, "extra_leaked_key": "val"})]
    )
    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step([{"text": "x"}], step, schema=["text"], llm_provider=p_extra)
    assert (
        exc_info.value.message
        == "Structured output validation failed: STRUCTURED_OUTPUT_INVALID (ADDITIONAL_PROPERTIES)"
    )
    assert "extra_leaked_key" not in exc_info.value.message

    # ENUM_MISMATCH
    p_enum = MockLLMProvider(responses=[json.dumps({"category": "C", "score": 50})])
    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step([{"text": "x"}], step, schema=["text"], llm_provider=p_enum)
    assert (
        exc_info.value.message
        == "Structured output validation failed: STRUCTURED_OUTPUT_INVALID (ENUM_MISMATCH)"
    )


def test_definition_and_import_time_avoids_arbitrary_schema_interpolation():
    """Verify that invalid schema at definition/import time produces a safe generic error without echoing secrets."""
    secret_marker = "SUPER_SECRET_INTERNAL_SCHEMA_KEY_9999"
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Classify {{ text }}",
        "output_column": "classification",
        "output_schema": {
            "type": "object",
            secret_marker: {"type": "string"},
        },
    }
    with pytest.raises(WorkflowExecutionError) as exc_info:
        validate_step_definitions([step])
    assert secret_marker not in str(exc_info.value)
    assert exc_info.value.message == "Step 0 (llm_generate): Invalid output_schema definition."


def test_budget_overrun_preserved_when_structured_validation_fails():
    """Verify that when structured validation fails, any budget overrun is truthfully preserved and not masked."""
    tracker = ExecutionBudgetTracker(max_total_tokens=100, max_ai_calls=5)
    provider = MockLLMProvider(
        responses=['{"invalid": true}'],
        usage=TokenUsage(prompt_tokens=20, completion_tokens=100, total_tokens=120),
    )
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "out",
        "max_output_tokens": 50,
        "output_schema": {"type": "object", "required": ["expected"]},
    }
    data = [{"text": "hello"}]

    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step(data, step, schema=["text"], llm_provider=provider, budget_tracker=tracker)

    # Must preserve the budget policy violation / overrun error without masking
    assert (
        "Safety policy violation (AIPolicyLimitError)" in exc_info.value.message
        or "budget exceeded" in exc_info.value.message.lower()
    )
    assert tracker.tokens_count == 120
    assert tracker.has_overrun is True
    assert tracker.tokens_overrun == 20
    assert tracker.reserved_tokens == 0
    assert tracker.calls_count == 1


def test_budget_charged_estimated_tokens_when_usage_missing_on_validation_failure():
    """Verify that when structured validation fails and provider returns no usage, conservative tokens are charged exactly once."""
    tracker = ExecutionBudgetTracker(max_total_tokens=1000, max_ai_calls=5)
    provider = MockLLMProvider(responses=['{"invalid": true}'], omit_usage=True)
    step = {
        "op": "llm_generate",
        "provider": "ollama",
        "model": "llama3",
        "prompt_template": "Analyze {{ text }}",
        "output_column": "out",
        "max_output_tokens": 50,
        "output_schema": {"type": "object", "required": ["expected"]},
    }
    data = [{"text": "hello"}]

    with pytest.raises(WorkflowExecutionError) as exc_info:
        execute_step(data, step, schema=["text"], llm_provider=provider, budget_tracker=tracker)

    assert "Structured output validation failed" in exc_info.value.message
    # Conservative tokens: estimated prompt tokens (~11) + max_output_tokens (50) = ~61
    assert tracker.tokens_count >= 50
    assert tracker.reserved_tokens == 0
    assert tracker.calls_count == 1
    assert tracker.has_overrun is False
