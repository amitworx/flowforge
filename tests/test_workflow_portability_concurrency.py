"""
PostgreSQL Concurrency Tests for Workflow Portability (M0-011 / M0-011A).

Deterministic proofs using pg_blocking_pids that:
1. When PATCH holds a row lock on a workflow definition, EXPORT blocks specifically on PATCH's
   backend PID, preventing dirty/torn definition reads, and unblocks to output the updated definition.
2. When EXPORT holds a row lock on a workflow definition, PATCH blocks specifically on EXPORT's
   backend PID, guaranteeing the snapshot taken by EXPORT is consistent and completed before PATCH mutates the record.
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


def wait_for_blocked_backend(waiter_pid: int, blocker_pid: int, timeout: float = 5.0) -> bool:
    """
    Polls PostgreSQL's pg_blocking_pids to confirm deterministically that
    'waiter_pid' is actively blocked waiting on a database lock held specifically by 'blocker_pid'.
    Eliminates broad assertions and timing assumptions.
    """
    start = time.monotonic()
    with connection.cursor() as cursor:
        while time.monotonic() - start < timeout:
            cursor.execute("SELECT pg_blocking_pids(%s)", [waiter_pid])
            row = cursor.fetchone()
            if row and row[0] and blocker_pid in row[0]:
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
    EXPORT arrives and is proven blocked specifically on PATCH's backend PID via pg_blocking_pids.
    EXPORT cannot complete while PATCH holds the lock.
    PATCH completes its mutation and commits.
    EXPORT unblocks, snapshots the newly committed definition, and exports it.
    """
    connection.close()

    patch_in_lock = threading.Event()
    export_blocked_verified = threading.Event()
    pids = {}
    results = {}
    thread_errors = []

    orig_select_for_update = WorkflowDefinition.objects.select_for_update

    def intercept_select(*args, **kwargs):
        qs = orig_select_for_update(*args, **kwargs)
        orig_get = qs.get

        def intercept_get(*g_args, **g_kwargs):
            wf = orig_get(*g_args, **g_kwargs)
            # Record PATCH's backend PID while inside atomic transaction holding the lock
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["patch_pid"] = cursor.fetchone()[0]
            patch_in_lock.set()
            if not export_blocked_verified.wait(timeout=5):
                raise TimeoutError("export_blocked_verified event was not set in time")
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
            # Record EXPORT's backend PID right before triggering row lock acquisition
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["export_pid"] = cursor.fetchone()[0]
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

    # Wait for PATCH to acquire lock
    assert patch_in_lock.wait(timeout=5), "PATCH did not acquire lock in time"

    # Wait for EXPORT's PID to be registered
    start_wait = time.monotonic()
    while "export_pid" not in pids and time.monotonic() - start_wait < 5.0:
        time.sleep(0.005)
    assert "export_pid" in pids, "EXPORT PID was not registered"

    # Verify that EXPORT is actively blocked specifically by PATCH's backend PID
    assert wait_for_blocked_backend(
        waiter_pid=pids["export_pid"],
        blocker_pid=pids["patch_pid"],
        timeout=5.0,
    ), f"EXPORT (PID {pids['export_pid']}) did not block on PATCH (PID {pids['patch_pid']})"

    # Assert EXPORT has not completed while PATCH still holds the lock
    assert t_export.is_alive(), "EXPORT unexpectedly completed before PATCH released lock"

    # Release PATCH lock and join
    export_blocked_verified.set()

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
    PATCH arrives and is proven blocked specifically on EXPORT's backend PID via pg_blocking_pids.
    PATCH cannot complete while EXPORT holds the lock.
    EXPORT finishes taking its snapshot and commits its read-lock transaction.
    PATCH unblocks and commits the edit.
    The exported JSON reflects the pre-patch definition without torn data.
    """
    connection.close()

    export_in_lock = threading.Event()
    patch_blocked_verified = threading.Event()
    pids = {}
    results = {}
    thread_errors = []

    orig_select_for_update = WorkflowDefinition.objects.select_for_update

    def intercept_select(*args, **kwargs):
        qs = orig_select_for_update(*args, **kwargs)
        orig_get = qs.get

        def intercept_get(*g_args, **g_kwargs):
            wf = orig_get(*g_args, **g_kwargs)
            # Record EXPORT's backend PID while inside atomic transaction holding the lock
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["export_pid"] = cursor.fetchone()[0]
            export_in_lock.set()
            if not patch_blocked_verified.wait(timeout=5):
                raise TimeoutError("patch_blocked_verified event was not set in time")
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
            # Record PATCH's backend PID right before triggering row lock acquisition
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["patch_pid"] = cursor.fetchone()[0]
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

    # Wait for EXPORT to acquire lock
    assert export_in_lock.wait(timeout=5), "EXPORT did not acquire lock in time"

    # Wait for PATCH's PID to be registered
    start_wait = time.monotonic()
    while "patch_pid" not in pids and time.monotonic() - start_wait < 5.0:
        time.sleep(0.005)
    assert "patch_pid" in pids, "PATCH PID was not registered"

    # Verify that PATCH is actively blocked specifically by EXPORT's backend PID
    assert wait_for_blocked_backend(
        waiter_pid=pids["patch_pid"],
        blocker_pid=pids["export_pid"],
        timeout=5.0,
    ), f"PATCH (PID {pids['patch_pid']}) did not block on EXPORT (PID {pids['export_pid']})"

    # Assert PATCH has not completed while EXPORT still holds the lock
    assert t_patch.is_alive(), "PATCH unexpectedly completed before EXPORT released lock"

    # Release EXPORT lock and join
    patch_blocked_verified.set()

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
