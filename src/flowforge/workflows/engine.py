"""Pure Python deterministic workflow execution engine."""

import math
import re
from typing import Any

from flowforge.ai.providers import (
    ChatMessage,
    ChatRole,
    LLMProvider,
    LLMRequest,
    ProviderError,
)
from flowforge.ai.safety import (
    AISafetyError,
    ExecutionBudgetTracker,
    SafetyLimits,
    redact_secrets_from_text,
    validate_prompt_input,
    validate_response_output,
)


class WorkflowExecutionError(Exception):
    def __init__(
        self,
        message: str,
        step_index: int | None = None,
        completed_step_audit: list[dict[str, Any]] | None = None,
        failed_step_audit: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.step_index = step_index
        self.completed_step_audit = completed_step_audit or []
        self.failed_step_audit = failed_step_audit


MAX_STEPS = 10
MAX_ROWS = 1000
MAX_NAME_LENGTH = 128
MAX_AI_ROWS_PER_WORKFLOW = 50
MAX_AI_OUTPUT_CHARS = 10_000

ALLOWED_OPERATIONS = {"select_columns", "rename_columns", "filter_equals", "llm_generate"}
STEP_ALLOWED_FIELDS = {
    "select_columns": {"op", "columns"},
    "rename_columns": {"op", "mapping"},
    "filter_equals": {"op", "column", "value"},
    "llm_generate": {
        "op",
        "provider",
        "model",
        "prompt_template",
        "output_column",
        "system_instruction",
        "max_output_tokens",
        "temperature",
    },
}

ALLOWED_LLM_PROVIDERS = {"ollama", "openai"}

# Template variable regex: {{ column_name }} with strict identifier rules
_TEMPLATE_VAR_REGEX = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def _estimate_prompt_tokens(prompt: str) -> int:
    """Defensible conservative prompt token bound (at least 1 token per 2 chars, minimum 1)."""
    return max(1, math.ceil(len(prompt) / 2))


def _classify_provider_error_reason(exc: Exception) -> str:
    """Map external exceptions to safe, generic, allowlisted error reasons.

    Never echoes raw str(exc), prompts, bodies, URLs, or unredacted keys.
    """
    if isinstance(exc, AISafetyError):
        return f"Safety policy violation ({type(exc).__name__})"

    if isinstance(exc, ProviderError):
        exc_type = type(exc).__name__
        if "Auth" in exc_type:
            return "Authentication or credentials error with upstream AI provider"
        if "Timeout" in exc_type:
            return "Upstream AI provider request timed out"
        if "Connection" in exc_type:
            return "Connection error reaching upstream AI provider"
        if "Response" in exc_type or "Invalid" in exc_type:
            return "Invalid response or request payload for AI provider"
        return "Upstream AI provider error occurred"

    if isinstance(exc, TimeoutError):
        return "Upstream AI provider request timed out"

    return "Upstream AI provider failure"


def _is_valid_scalar(val: Any) -> bool:
    """Scalar types: str, int, float, bool, None."""
    return isinstance(val, (str, int, float, bool)) or val is None


def _is_exact_scalar_equal(a: Any, b: Any) -> bool:
    """
    Type-sensitive scalar comparison:
    - If either value is a boolean, both must be boolean and equal.
      (Prevents Python's True == 1 and False == 0 trap).
    - If both are numbers (int or float), numeric equality applies (e.g. 1 == 1.0).
    - If both are strings, exact string equality applies (preserving leading zeros).
    - None matches only None.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return False


def validate_step_definitions(steps: Any) -> list[dict[str, Any]]:
    """Validate definition-time step schemas."""
    if not isinstance(steps, list):
        raise WorkflowExecutionError("Steps must be a list of step definitions.")
    if not steps:
        raise WorkflowExecutionError("Workflow must contain at least one step.")
    if len(steps) > MAX_STEPS:
        raise WorkflowExecutionError(f"Workflow cannot exceed maximum {MAX_STEPS} steps.")

    validated = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise WorkflowExecutionError(f"Step {idx} must be a dictionary.")

        op = step.get("op")
        if not op or not isinstance(op, str) or op not in ALLOWED_OPERATIONS:
            raise WorkflowExecutionError(
                f"Step {idx}: Unsupported operation '{op}'. Supported operations are: {sorted(ALLOWED_OPERATIONS)}"
            )

        allowed_fields = STEP_ALLOWED_FIELDS[op]
        extra_fields = set(step.keys()) - allowed_fields
        if extra_fields:
            raise WorkflowExecutionError(
                f"Step {idx} ({op}): Unexpected fields {sorted(extra_fields)}. Allowed fields: {sorted(allowed_fields)}."
            )

        if op == "select_columns":
            columns = step.get("columns")
            if not isinstance(columns, list) or not columns:
                raise WorkflowExecutionError(
                    f"Step {idx} (select_columns): 'columns' must be a non-empty list of column names."
                )
            for c in columns:
                if not isinstance(c, str) or not c.strip() or len(c) > MAX_NAME_LENGTH:
                    raise WorkflowExecutionError(
                        f"Step {idx} (select_columns): Invalid column name '{c}'."
                    )
            if len(set(columns)) != len(columns):
                raise WorkflowExecutionError(
                    f"Step {idx} (select_columns): Duplicate column names in selection."
                )

        elif op == "rename_columns":
            mapping = step.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise WorkflowExecutionError(
                    f"Step {idx} (rename_columns): 'mapping' must be a non-empty dictionary."
                )
            for k, v in mapping.items():
                if (
                    not isinstance(k, str)
                    or not k.strip()
                    or not isinstance(v, str)
                    or not v.strip()
                    or len(k) > MAX_NAME_LENGTH
                    or len(v) > MAX_NAME_LENGTH
                ):
                    raise WorkflowExecutionError(
                        f"Step {idx} (rename_columns): Invalid column mapping pair '{k}': '{v}'."
                    )
            # Check target collisions
            targets = list(mapping.values())
            if len(set(targets)) != len(targets):
                raise WorkflowExecutionError(
                    f"Step {idx} (rename_columns): Mapping contains duplicate target column collision."
                )

        elif op == "filter_equals":
            col = step.get("column")
            if not isinstance(col, str) or not col.strip() or len(col) > MAX_NAME_LENGTH:
                raise WorkflowExecutionError(
                    f"Step {idx} (filter_equals): 'column' must be a valid non-empty string."
                )
            if "value" not in step:
                raise WorkflowExecutionError(
                    f"Step {idx} (filter_equals): Missing required 'value' field."
                )
            val = step.get("value")
            if not _is_valid_scalar(val):
                raise WorkflowExecutionError(
                    f"Step {idx} (filter_equals): 'value' must be a scalar (string, number, boolean, or null)."
                )

        elif op == "llm_generate":
            provider = step.get("provider")
            if (
                not provider
                or not isinstance(provider, str)
                or provider not in ALLOWED_LLM_PROVIDERS
            ):
                raise WorkflowExecutionError(
                    f"Step {idx} (llm_generate): Unsupported or unapproved LLM provider '{provider}'. "
                    f"Allowed providers: {sorted(ALLOWED_LLM_PROVIDERS)}."
                )

            model = step.get("model")
            if (
                not model
                or not isinstance(model, str)
                or not model.strip()
                or len(model) > MAX_NAME_LENGTH
            ):
                raise WorkflowExecutionError(
                    f"Step {idx} (llm_generate): 'model' must be a non-empty string <= {MAX_NAME_LENGTH} chars."
                )

            prompt_tmpl = step.get("prompt_template")
            if not prompt_tmpl or not isinstance(prompt_tmpl, str) or not prompt_tmpl.strip():
                raise WorkflowExecutionError(
                    f"Step {idx} (llm_generate): 'prompt_template' must be a non-empty string."
                )

            # Validate that all {{...}} patterns match allowable scalar column identifiers,
            # and reject any malformed braces (e.g. {{, }}, {{}}, invalid characters)
            braces_pattern = re.compile(r"\{\{([^}]*)\}\}")
            found_braces = braces_pattern.findall(prompt_tmpl)
            valid_vars = _TEMPLATE_VAR_REGEX.findall(prompt_tmpl)
            if prompt_tmpl.count("{{") != len(found_braces) or prompt_tmpl.count("}}") != len(
                found_braces
            ):
                raise WorkflowExecutionError(
                    f"Step {idx} (llm_generate): Invalid template placeholder syntax in prompt_template."
                )
            if len(found_braces) != len(valid_vars):
                raise WorkflowExecutionError(
                    f"Step {idx} (llm_generate): Invalid template placeholder identifier in prompt_template."
                )
            for var in valid_vars:
                if not var[0].isalpha() and var[0] != "_":
                    raise WorkflowExecutionError(
                        f"Step {idx} (llm_generate): Invalid template placeholder identifier '{var}'."
                    )

            output_col = step.get("output_column")
            if (
                not output_col
                or not isinstance(output_col, str)
                or not output_col.strip()
                or len(output_col) > MAX_NAME_LENGTH
            ):
                raise WorkflowExecutionError(
                    f"Step {idx} (llm_generate): 'output_column' must be a valid non-empty string <= {MAX_NAME_LENGTH} chars."
                )

            if "system_instruction" in step:
                sys_inst = step.get("system_instruction")
                if sys_inst is not None and (not isinstance(sys_inst, str) or len(sys_inst) > 5000):
                    raise WorkflowExecutionError(
                        f"Step {idx} (llm_generate): 'system_instruction' must be a string up to 5000 chars."
                    )

            if "max_output_tokens" in step:
                max_tokens = step.get("max_output_tokens")
                if (
                    isinstance(max_tokens, bool)
                    or not isinstance(max_tokens, int)
                    or max_tokens <= 0
                    or max_tokens > 8192
                ):
                    raise WorkflowExecutionError(
                        f"Step {idx} (llm_generate): 'max_output_tokens' must be a positive integer <= 8192."
                    )

            if "temperature" in step:
                temp = step.get("temperature")
                if (
                    isinstance(temp, bool)
                    or not isinstance(temp, (int, float))
                    or not math.isfinite(temp)
                    or temp < 0.0
                    or temp > 2.0
                ):
                    raise WorkflowExecutionError(
                        f"Step {idx} (llm_generate): 'temperature' must be a float between 0.0 and 2.0."
                    )

        validated.append(step)
    return validated


def execute_step(
    data: list[dict[str, Any]],
    step: dict[str, Any],
    schema: list[str] | None = None,
    llm_provider: LLMProvider | None = None,
    llm_providers: dict[str, LLMProvider] | None = None,
    budget_tracker: ExecutionBudgetTracker | None = None,
) -> list[dict[str, Any]]:
    """
    Execute a single deterministic step on data, propagating and enforcing schema.
    Returns transformed data as a list of dicts.
    If schema is provided, modifies schema in-place (or caller uses execute_step_with_schema).
    """
    transformed_data, new_schema = execute_step_with_schema(
        data,
        step,
        schema,
        llm_provider=llm_provider,
        llm_providers=llm_providers,
        budget_tracker=budget_tracker,
    )
    if schema is not None:
        schema.clear()
        schema.extend(new_schema)
    return transformed_data


def execute_step_with_schema(
    data: list[dict[str, Any]],
    step: dict[str, Any],
    schema: list[str] | None = None,
    llm_provider: LLMProvider | None = None,
    llm_providers: dict[str, LLMProvider] | None = None,
    budget_tracker: ExecutionBudgetTracker | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Execute a single step on data with explicit schema tracking.
    Returns (transformed_data, new_schema).
    """
    op = step["op"]

    # Check input row count
    if len(data) > MAX_ROWS:
        raise WorkflowExecutionError(f"Input data exceeds maximum limit of {MAX_ROWS} rows.")

    # Determine active columns
    if schema is not None:
        existing_keys = list(schema)
    elif data:
        existing_keys = list(data[0].keys())
    else:
        existing_keys = []

    if op == "select_columns":
        columns = step["columns"]
        # Validate that all requested columns exist in active schema
        for col in columns:
            if col not in existing_keys:
                raise WorkflowExecutionError(
                    f"Column '{col}' requested in select_columns does not exist in input data."
                )
        new_data = [{col: row[col] for col in columns} for row in data]
        return new_data, list(columns)

    elif op == "rename_columns":
        mapping = step["mapping"]
        # Validate source columns exist in active schema
        for old_col in mapping:
            if old_col not in existing_keys:
                raise WorkflowExecutionError(
                    f"Column '{old_col}' to rename does not exist in input data."
                )

        # Check collision with existing unrenamed columns
        unrenamed_keys = set(existing_keys) - set(mapping.keys())
        for _old_col, new_col in mapping.items():
            if new_col in unrenamed_keys:
                raise WorkflowExecutionError(
                    f"Rename collision: target column '{new_col}' already exists in input data."
                )

        new_schema = [mapping.get(k, k) for k in existing_keys]

        new_data = []
        for row in data:
            new_row = {}
            for k, v in row.items():
                target_k = mapping.get(k, k)
                new_row[target_k] = v
            new_data.append(new_row)
        return new_data, new_schema

    elif op == "filter_equals":
        col = step["column"]
        target_val = step["value"]
        if col not in existing_keys:
            raise WorkflowExecutionError(f"Column '{col}' to filter does not exist in input data.")

        new_data = []
        for row in data:
            row_val = row.get(col)
            if _is_exact_scalar_equal(row_val, target_val):
                new_data.append(dict(row))
        return new_data, list(existing_keys)

    elif op == "llm_generate":
        output_col = step["output_column"]
        # Schema collision validation
        if output_col in existing_keys:
            raise WorkflowExecutionError(f"Output column '{output_col}' already exists in schema.")

        prompt_tmpl = step["prompt_template"]
        placeholders = _TEMPLATE_VAR_REGEX.findall(prompt_tmpl)
        for var in placeholders:
            if var not in existing_keys:
                raise WorkflowExecutionError(f"Placeholder '{var}' does not exist in schema.")

        new_schema = list(existing_keys) + [output_col]

        # Empty data: schema preserved without calling provider
        if not data:
            return [], new_schema

        provider_type = step["provider"]
        # Explicit provider resolution: if llm_providers mapping is provided,
        # require explicit mapping for provider_type (no silent fallback to other providers).
        if llm_providers is not None:
            if not llm_providers:
                raise WorkflowExecutionError(
                    "LLM execution is opt-in and requires an explicit, secure provider instance."
                )
            effective_provider = llm_providers.get(provider_type)
            if effective_provider is None:
                raise WorkflowExecutionError(
                    f"No provider configured for '{provider_type}' in multi-provider mapping."
                )
        else:
            effective_provider = llm_provider

        if effective_provider is None:
            raise WorkflowExecutionError(
                "LLM execution is opt-in and requires an explicit, secure provider instance."
            )

        if len(data) > MAX_AI_ROWS_PER_WORKFLOW:
            raise WorkflowExecutionError(
                f"AI step exceeds max allowed rows limit of {MAX_AI_ROWS_PER_WORKFLOW} (got {len(data)})."
            )

        model = step["model"]
        system_instruction = step.get("system_instruction")
        max_output_tokens = step.get("max_output_tokens", 1024)
        temperature = step.get("temperature", 0.7)

        safety_limits = SafetyLimits(
            max_output_tokens=max_output_tokens,
            max_workflow_ai_calls=MAX_AI_ROWS_PER_WORKFLOW,
        )

        new_data = []
        for row_idx, row in enumerate(data):
            # Substitute scalar columns deterministically
            def replace_var(match: re.Match[str], current_row=row, current_idx=row_idx) -> str:
                col_name = match.group(1)
                val = current_row.get(col_name)
                if not _is_valid_scalar(val):
                    raise WorkflowExecutionError(
                        f"Row {current_idx}: Column '{col_name}' value must be a scalar for prompt substitution."
                    )
                return "" if val is None else str(val)

            rendered_prompt = _TEMPLATE_VAR_REGEX.sub(replace_var, prompt_tmpl)

            # Validate input safety limits
            try:
                validate_prompt_input(rendered_prompt, safety_limits)
            except AISafetyError as exc:
                raise WorkflowExecutionError(f"Row {row_idx}: {exc.message}") from None

            messages = []
            if system_instruction:
                messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system_instruction))
            messages.append(ChatMessage(role=ChatRole.USER, content=rendered_prompt))

            req = LLMRequest(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
            )

            # True atomic pre-call budget reservation via public API:
            reservation_id = None
            if budget_tracker:
                estimated_prompt_tokens = _estimate_prompt_tokens(rendered_prompt)
                conservative_tokens = estimated_prompt_tokens + int(max_output_tokens)
                try:
                    reservation_id = budget_tracker.reserve_call(
                        estimated_tokens=conservative_tokens
                    )
                except AISafetyError as exc:
                    raise WorkflowExecutionError(
                        f"AI execution budget exceeded: {exc.message}"
                    ) from None

            try:
                resp = effective_provider.generate(req)
                resp_content = resp.text or ""
                # Redact any inadvertent secret shapes from generated output
                sanitized_output = redact_secrets_from_text(resp_content)

                if len(sanitized_output) > MAX_AI_OUTPUT_CHARS:
                    sanitized_output = sanitized_output[:MAX_AI_OUTPUT_CHARS]

                # Validate output size
                validate_response_output(
                    sanitized_output.encode("utf-8"),
                    resp.usage.completion_tokens if resp.usage else 0,
                    safety_limits,
                )

                if budget_tracker and reservation_id:
                    actual_tokens = (
                        resp.usage.total_tokens
                        if (resp.usage and resp.usage.total_tokens is not None)
                        else None
                    )
                    budget_tracker.complete_call(reservation_id, actual_tokens=actual_tokens)

            except (AISafetyError, ProviderError, Exception) as exc:
                if budget_tracker and reservation_id:
                    budget_tracker.complete_call(reservation_id, actual_tokens=0, failed=True)
                if isinstance(exc, WorkflowExecutionError):
                    raise
                reason = _classify_provider_error_reason(exc)
                raise WorkflowExecutionError(f"LLM generation failed: {reason}") from None

            new_row = dict(row)
            new_row[output_col] = sanitized_output
            new_data.append(new_row)

        return new_data, new_schema

    raise WorkflowExecutionError(f"Unsupported operation '{op}'")


def execute_workflow_pipeline_with_schema(
    data: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    schema: list[str] | None = None,
    llm_provider: LLMProvider | None = None,
    llm_providers: dict[str, LLMProvider] | None = None,
    budget_tracker: ExecutionBudgetTracker | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    Execute an ordered pipeline of steps against data.
    Propagates deterministic schema across all steps, even if intermediate data becomes empty.
    Returns (output_data, step_audits, final_schema).
    Raises WorkflowExecutionError on failure, preserving completed step audits.
    """
    validate_step_definitions(steps)

    current_data = [dict(row) for row in data]
    if schema is not None:
        current_schema = list(schema)
    elif current_data:
        current_schema = list(current_data[0].keys())
    else:
        current_schema = []

    completed_audits = []
    cumulative_ai_rows = 0

    for idx, step in enumerate(steps):
        input_count = len(current_data)
        op = step["op"]

        try:
            if op == "llm_generate":
                cumulative_ai_rows += input_count
                if cumulative_ai_rows > MAX_AI_ROWS_PER_WORKFLOW:
                    raise WorkflowExecutionError(
                        f"Cumulative AI rows across workflow ({cumulative_ai_rows}) exceeds max allowed limit of {MAX_AI_ROWS_PER_WORKFLOW}."
                    )

            current_data, current_schema = execute_step_with_schema(
                current_data,
                step,
                schema=current_schema,
                llm_provider=llm_provider,
                llm_providers=llm_providers,
                budget_tracker=budget_tracker,
            )
            output_count = len(current_data)
            completed_audits.append(
                {
                    "step_index": idx,
                    "operation": op,
                    "status": "completed",
                    "input_row_count": input_count,
                    "output_row_count": output_count,
                    "error_summary": None,
                }
            )
        except WorkflowExecutionError as exc:
            failed_audit = {
                "step_index": idx,
                "operation": op,
                "status": "failed",
                "input_row_count": input_count,
                "output_row_count": 0,
                "error_summary": exc.message,
            }
            raise WorkflowExecutionError(
                message=exc.message,
                step_index=idx,
                completed_step_audit=completed_audits,
                failed_step_audit=failed_audit,
            ) from exc

    return current_data, completed_audits, current_schema


def execute_workflow_pipeline(
    data: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    schema: list[str] | None = None,
    llm_provider: LLMProvider | None = None,
    llm_providers: dict[str, LLMProvider] | None = None,
    budget_tracker: ExecutionBudgetTracker | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Execute an ordered pipeline of steps against data.
    Preserves backward compatibility returning (output_data, step_audits).
    """
    output_data, audits, _schema = execute_workflow_pipeline_with_schema(
        data,
        steps,
        schema=schema,
        llm_provider=llm_provider,
        llm_providers=llm_providers,
        budget_tracker=budget_tracker,
    )
    return output_data, audits
