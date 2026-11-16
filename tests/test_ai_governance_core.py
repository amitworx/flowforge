import pytest

from flowforge.ai.governance.exceptions import (
    AIGovernanceError,
)
from flowforge.ai.governance.memory_repository import InMemoryBudgetRepository
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
)
from flowforge.ai.governance.repository import BudgetRepository


def test_budget_settings_validation():
    # Valid budget settings
    scope = BudgetScope(
        scope_type=BudgetScopeType.TENANT,
        tenant_id="tenant-123",
    )
    settings = BudgetSettings(
        scope=scope,
        version=1,
        max_calls=10,
        max_total_tokens=1000,
        max_estimated_cost_micros=5000,
        allowed_providers=("anthropic", "openai"),
        denied_providers=(),
    )
    assert settings.scope.tenant_id == "tenant-123"
    assert settings.version == 1
    assert settings.max_calls == 10
    assert settings.max_total_tokens == 1000
    assert "anthropic" in settings.allowed_providers

    # Invalid limits: negative tokens
    with pytest.raises(AIGovernanceError):
        BudgetSettings(
            scope=scope,
            version=1,
            max_calls=10,
            max_total_tokens=-1,
        )


def test_in_memory_repository_implements_protocol():
    repo = InMemoryBudgetRepository()
    assert isinstance(repo, BudgetRepository)
