"""
PostgreSQL Concurrency Tests for Workflow Portability (M0-011).

Deterministic proofs using pg_locks (pg_blocking_pids / NOT granted) that:
1. When PATCH holds a row lock on a workflow definition, EXPORT blocks on that lock,
   preventing dirty/torn definition reads, and unblocks to output the updated definition.
2. When EXPORT holds a row lock on a workflow definition, PATCH blocks on that lock,
   guaranteeing the snapshot taken by EXPORT is consistent and completed before PATCH mutates the record.
"""

import threading
import time
from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from django.db import connection
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.workflows.models import WorkflowDefinition


@pytest.fixture
def user_a(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="user_portability_concurrency", password="password123")


@pytest.fixture
def sample_workflow(user_a: User) -> WorkflowDefinition:
    return WorkflowDefinition.objects.create(
        owner=user_a,
        name="Original Portability Workflow",
        steps=[
            {"op": "select_columns", "columns": ["id", "name"]},
        ],
    )


def wait_for_blocked_db_lock(timeout: float = 5.0) -> bool:
    """
    Polls PostgreSQL's pg_locks table to confirm deterministically that a transaction
    has reached and is actively blocked waiting on a database lock (NOT granted).
    Guarantees no timing assumptions or arbitrary sleeps are needed.
    """
    start = time.monotonic()
    with connection.cursor() as cursor:
        while time.monotonic() - start < timeout:
            cursor.execute("SELECT count(*) FROM pg_locks WHERE NOT granted")
            row = cursor.fetchone()
            if row and row[0] > 0:
                return True
            time.sleep(0.005)
    return False


@pytest.mark.django_db(transaction=True)
def test_concurrent_patch_then_export_blocks_and_exports_updated_definition(
    user_a, sample_workflow
):
    """
    Ordering Proof: PATCH -> EXPORT
    Test scenario: PATCH acquires the row lock to mutate the definition.
    EXPORT arrives and blocks on the PostgreSQL row lock.
    PATCH completes its mutation and commits.
    EXPORT unblocks, snapshots the newly committed definition, and exports it.
    """
    patch_in_lock = threading.Event()
    export_blocked = threading.Event()
    results = {}
    thread_errors = []

    orig_select_for_update = WorkflowDefinition.objects.select_for_update

    def intercept_select(*args, **kwargs):
        qs = orig_select_for_update(*args, **kwargs)
        orig_get = qs.get

        def intercept_get(*g_args, **g_kwargs):
            wf = orig_get(*g_args, **g_kwargs)
            patch_in_lock.set()
            if not export_blocked.wait(timeout=5):
                raise TimeoutError("export_blocked event was not set in time")
            return wf

        qs.get = intercept_get
        return qs

    def do_patch():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition.objects,
                "select_for_update",
                side_effect=intercept_select,
            ):
                res = client.patch(
                    f"/api/v1/workflows/{sample_workflow.id}/",
                    data={
                        "name": "Updated Portability Workflow",
                        "steps": [{"op": "select_columns", "columns": ["name"]}],
                    },
                    format="json",
                )
                results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_export():
        try:
            connection.close()
            if not patch_in_lock.wait(timeout=5):
                raise TimeoutError("patch_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.get(f"/api/v1/workflows/{sample_workflow.id}/export/")
            results["export_status"] = res.status_code
            results["export_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_export = threading.Thread(target=do_export)

    t_patch.start()
    t_export.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "EXPORT did not block on PostgreSQL lock"
    export_blocked.set()

    t_patch.join(timeout=10)
    t_export.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_export.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["export_status"] == status.HTTP_200_OK
    assert results["export_data"]["format"] == "flowforge.workflow"
    assert results["export_data"]["format_version"] == 1
    assert results["export_data"]["name"] == "Updated Portability Workflow"
    assert results["export_data"]["steps"] == [{"op": "select_columns", "columns": ["name"]}]


@pytest.mark.django_db(transaction=True)
def test_concurrent_export_then_patch_uses_pre_patch_snapshot(user_a, sample_workflow):
    """
    Ordering Proof: EXPORT -> PATCH
    Test scenario: EXPORT acquires the row lock to snapshot the active definition.
    PATCH arrives and blocks on the PostgreSQL row lock.
    EXPORT finishes taking its snapshot and commits its read-lock transaction.
    PATCH unblocks and commits the edit.
    The exported JSON reflects the pre-patch definition, without torn data.
    """
    export_in_lock = threading.Event()
    patch_blocked = threading.Event()
    results = {}
    thread_errors = []

    orig_select_for_update = WorkflowDefinition.objects.select_for_update

    def intercept_select(*args, **kwargs):
        qs = orig_select_for_update(*args, **kwargs)
        orig_get = qs.get

        def intercept_get(*g_args, **g_kwargs):
            wf = orig_get(*g_args, **g_kwargs)
            export_in_lock.set()
            if not patch_blocked.wait(timeout=5):
                raise TimeoutError("patch_blocked event was not set in time")
            return wf

        qs.get = intercept_get
        return qs

    def do_export():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition.objects,
                "select_for_update",
                side_effect=intercept_select,
            ):
                res = client.get(f"/api/v1/workflows/{sample_workflow.id}/export/")
                results["export_status"] = res.status_code
                results["export_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_patch():
        try:
            connection.close()
            if not export_in_lock.wait(timeout=5):
                raise TimeoutError("export_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.patch(
                f"/api/v1/workflows/{sample_workflow.id}/",
                data={
                    "name": "Updated Portability Workflow",
                    "steps": [{"op": "select_columns", "columns": ["name"]}],
                },
                format="json",
            )
            results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_export = threading.Thread(target=do_export)
    t_patch = threading.Thread(target=do_patch)

    t_export.start()
    t_patch.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "PATCH did not block on PostgreSQL lock"
    patch_blocked.set()

    t_export.join(timeout=10)
    t_patch.join(timeout=10)

    assert not t_export.is_alive()
    assert not t_patch.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["export_status"] == status.HTTP_200_OK
    assert results["patch_status"] == status.HTTP_200_OK

    # Exported data should reflect pre-patch state
    assert results["export_data"]["name"] == "Original Portability Workflow"
    assert results["export_data"]["steps"] == [{"op": "select_columns", "columns": ["id", "name"]}]

    # Workflow in DB has updated state
    sample_workflow.refresh_from_db()
    assert sample_workflow.name == "Updated Portability Workflow"
    assert sample_workflow.steps == [{"op": "select_columns", "columns": ["name"]}]
