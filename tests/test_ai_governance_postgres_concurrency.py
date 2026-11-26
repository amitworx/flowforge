"""High-concurrency stress test suite verifying PostgreSQLBudgetRepository locking and atomicity."""

import concurrent.futures
import datetime

import pytest
from django.db import connection

from flowforge.ai.governance.db.repository import PostgreSQLBudgetRepository
from flowforge.ai.governance.exceptions import AIBudgetExhaustedError
from flowforge.ai.governance.models import (
    BudgetScope,
    BudgetScopeType,
    BudgetSettings,
    SettlementType,
)


@pytest.mark.django_db(transaction=True)
def test_postgres_concurrent_reservations_budget_limit():
    """Verify that 20 concurrent threads trying to reserve against a budget of 5 calls

    result in exactly 5 successful reservations and 15 budget exhausted errors without deadlocks.
    """
    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-stress-limit")

    # Set budget limit: max 5 calls
    settings = BudgetSettings(scope=scope, max_calls=5, max_total_tokens=10000)
    repo.set_budget_settings(settings)

    num_threads = 20
    successes = []
    exhausted = []
    errors = []

    def _worker(thread_idx: int):
        # Explicitly close old connection to force fresh per-thread connection in Django
        connection.close()
        now = datetime.datetime.now(datetime.UTC)
        try:
            res, is_replayed = repo.reserve_atomic(
                reservation_id=f"res-thread-{thread_idx}",
                idempotency_key=f"idemp-thread-{thread_idx}",
                scope=scope,
                ancestor_scopes=[],
                provider="ollama",
                model="llama3",
                estimated_prompt_tokens=10,
                estimated_completion_tokens=20,
                estimated_total_tokens=30,
                estimated_cost_micros=500,
                attempt_number=1,
                created_at=now,
            )
            successes.append(res.reservation_id)
        except AIBudgetExhaustedError:
            exhausted.append(thread_idx)
        except Exception as e:
            errors.append((thread_idx, type(e).__name__, str(e)))
        finally:
            connection.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Encountered unexpected errors during concurrency: {errors}"
    assert len(successes) == 5, f"Expected exactly 5 successful reservations, got {len(successes)}"
    assert len(exhausted) == 15, f"Expected 15 budget exhausted rejections, got {len(exhausted)}"

    # Authoritative counter verification
    active_calls, active_tokens, active_cost = repo.get_active_reserved_usage(scope)
    assert active_calls == 5
    assert active_tokens == 5 * 30
    assert active_cost == 5 * 500


@pytest.mark.django_db(transaction=True)
def test_postgres_concurrent_settlements_no_deadlock():
    """Verify that multiple concurrent reservations and settlements execute cleanly without deadlocks."""
    repo = PostgreSQLBudgetRepository()
    scope = BudgetScope(scope_type=BudgetScopeType.TENANT, tenant_id="tenant-stress-settle")
    settings = BudgetSettings(scope=scope, max_calls=50, max_total_tokens=100000)
    repo.set_budget_settings(settings)

    num_ops = 10
    now = datetime.datetime.now(datetime.UTC)

    # Pre-create reservations
    for i in range(num_ops):
        repo.reserve_atomic(
            reservation_id=f"res-concurr-{i}",
            idempotency_key=f"idemp-concurr-{i}",
            scope=scope,
            ancestor_scopes=[],
            provider="ollama",
            model="llama3",
            estimated_prompt_tokens=10,
            estimated_completion_tokens=20,
            estimated_total_tokens=30,
            estimated_cost_micros=500,
            attempt_number=1,
            created_at=now,
        )

    errors = []

    def _settle_worker(idx: int):
        connection.close()
        try:
            settle_now = datetime.datetime.now(datetime.UTC)
            repo.settle_reservation(
                settlement_id=f"stl-concurr-{idx}",
                reservation_id=f"res-concurr-{idx}",
                idempotency_key=f"idemp-stl-concurr-{idx}",
                settlement_type=SettlementType.EXACT,
                actual_prompt_tokens=10,
                actual_completion_tokens=20,
                actual_total_tokens=30,
                actual_cost_micros=500,
                attempt_number=1,
                settled_at=settle_now,
            )
        except Exception as e:
            errors.append((idx, type(e).__name__, str(e)))
        finally:
            connection.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_ops) as executor:
        futures = [executor.submit(_settle_worker, i) for i in range(num_ops)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Encountered unexpected settlement errors: {errors}"

    total_calls, total_tokens, total_cost = repo.get_aggregate_usage(scope)
    assert total_calls == num_ops
    assert total_tokens == num_ops * 30
    assert total_cost == num_ops * 500
