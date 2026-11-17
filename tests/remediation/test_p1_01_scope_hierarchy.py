"""RED regression tests for P1-01: Correct Provider Scope Hierarchy Enforcement."""

import datetime

import pytest

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import AIBudgetExhaustedError
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
)


@pytest.fixture(params=["in_memory", "postgres"])
def repo(request):
    if request.param == "in_memory":
        return InMemoryBudgetRepository()
    return PostgreSQLBudgetRepository()


@pytest.mark.django_db(transaction=True)
def test_p1_01_provider_scope_enforces_workflow_ancestor_budget(repo):
    """When reserving under a PROVIDER scope that has a workflow context, the workflow's budget cap must be enforced."""
    wf_scope = BudgetScope(
        scope_type=BudgetScopeType.WORKFLOW,
        tenant_id="tenant-p1",
        workflow_id="wf-1",
    )
    provider_scope = BudgetScope(
        scope_type=BudgetScopeType.PROVIDER,
        tenant_id="tenant-p1",
        workflow_id="wf-1",
        provider="ollama",
    )

    # Set workflow limit: max 1 call
    repo.set_budget_settings(
        BudgetSettings(
            scope=wf_scope,
            max_calls=1,
            max_total_tokens=10000,
        )
    )

    now = datetime.datetime.now(datetime.UTC)

    # Call 1 on provider_scope: must succeed
    repo.reserve_atomic(
        reservation_id="res-prov-1",
        idempotency_key="idemp-prov-1",
        scope=provider_scope,
        ancestor_scopes=provider_scope.get_ancestor_scopes(),
        provider="ollama",
        model="llama3",
        estimated_prompt_tokens=100,
        estimated_completion_tokens=200,
        estimated_total_tokens=300,
        estimated_cost_micros=5000,
        attempt_number=1,
        created_at=now,
    )

    # Call 2 on provider_scope (or another provider under same workflow): must exceed workflow limit of 1
    with pytest.raises(AIBudgetExhaustedError) as exc_info:
        repo.reserve_atomic(
            reservation_id="res-prov-2",
            idempotency_key="idemp-prov-2",
            scope=provider_scope,
            ancestor_scopes=provider_scope.get_ancestor_scopes(),
            provider="ollama",
            model="llama3",
            estimated_prompt_tokens=100,
            estimated_completion_tokens=200,
            estimated_total_tokens=300,
            estimated_cost_micros=5000,
            attempt_number=1,
            created_at=now,
        )
    assert exc_info.value.limit_name == "max_calls"
