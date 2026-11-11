import threading
import time

import pytest
from django.db import connection
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.workflows.models import WorkflowDefinition


@pytest.fixture
def user_concurrency(django_user_model):
    return django_user_model.objects.create_user(username="concurr_user", password="password123")


@pytest.fixture
def workflow_concurrency(user_concurrency):
    return WorkflowDefinition.objects.create(
        owner=user_concurrency,
        name="Concurrent Test Workflow",
        steps=[{"op": "select_columns", "columns": ["id", "name"]}],
    )


def wait_for_blocked_backend(waiter_pid: int, blocker_pid: int, timeout: float = 5.0) -> bool:
    """
    Polls PostgreSQL's pg_stat_activity using pg_blocking_pids to confirm
    deterministically that 'waiter_pid' is actively blocked waiting on a database lock
    held specifically by 'blocker_pid'.
    Eliminates broad assertions and timing assumptions.
    """
    start = time.monotonic()
    with connection.cursor() as cursor:
        while time.monotonic() - start < timeout:
            cursor.execute(
                "SELECT pg_blocking_pids(%s)",
                [waiter_pid],
            )
            row = cursor.fetchone()
            if row and row[0] and blocker_pid in row[0]:
                return True
            time.sleep(0.005)
    return False


@pytest.mark.django_db(transaction=True)
def test_concurrency_patch_holds_lock_blocks_clone_contention(
    user_concurrency, workflow_concurrency
):
    """
    Overlapping Concurrency Test 1:
    PATCH acquires select_for_update() row lock first and holds it open.
    CLONE arrives and is proven blocked specifically by PATCH's backend PID via pg_blocking_pids.
    CLONE cannot finish until PATCH commits.
    Once PATCH commits, CLONE unblocks and captures the FULL updated workflow definition.
    """
    connection.close()

    patch_in_lock = threading.Event()
    clone_blocked_verified = threading.Event()
    pids = {}
    results = {}
    thread_errors = []

    original_save = WorkflowDefinition.save

    def intercept_patch_save(self, *args, **kwargs):
        if self.id == workflow_concurrency.id and "update_fields" in kwargs:
            # We are holding the row lock inside PATCH's atomic transaction
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["patch_pid"] = cursor.fetchone()[0]
            patch_in_lock.set()
            # Wait until main thread verifies CLONE is blocked specifically on our lock
            if not clone_blocked_verified.wait(timeout=5):
                raise TimeoutError("clone_blocked_verified event was not set in time")
        return original_save(self, *args, **kwargs)

    def do_patch():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            from unittest.mock import patch

            with patch.object(
                WorkflowDefinition, "save", autospec=True, side_effect=intercept_patch_save
            ):
                res = client.patch(
                    f"/api/v1/workflows/{workflow_concurrency.id}/",
                    data={
                        "name": "Updated Workflow By Patch",
                        "steps": [
                            {"op": "select_columns", "columns": ["id", "name", "email"]},
                        ],
                    },
                    format="json",
                )
                results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_clone():
        try:
            connection.close()
            # Wait until PATCH holds the row lock
            if not patch_in_lock.wait(timeout=5):
                raise TimeoutError("patch_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            # Record our backend pid right before executing clone
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["clone_pid"] = cursor.fetchone()[0]
            # This POST will block on select_for_update() row lock held by do_patch
            res = client.post(
                f"/api/v1/workflows/{workflow_concurrency.id}/clone/",
                data={},
                format="json",
            )
            results["clone_status"] = res.status_code
            results["clone_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_clone = threading.Thread(target=do_clone)

    t_patch.start()
    t_clone.start()

    # Wait for PATCH to signal it holds the row lock
    assert patch_in_lock.wait(timeout=5), "PATCH did not acquire lock in time"

    # Wait for CLONE's PID to be registered
    start_wait = time.monotonic()
    while "clone_pid" not in pids and time.monotonic() - start_wait < 5.0:
        time.sleep(0.005)
    assert "clone_pid" in pids, "CLONE PID was not registered"

    # Verify that CLONE is actively blocked by PATCH's backend PID
    assert wait_for_blocked_backend(
        waiter_pid=pids["clone_pid"],
        blocker_pid=pids["patch_pid"],
        timeout=5.0,
    ), f"CLONE (PID {pids['clone_pid']}) did not block on PATCH (PID {pids['patch_pid']})"

    # Assert CLONE thread has NOT finished yet while PATCH holds the lock
    assert t_clone.is_alive(), "CLONE unexpectedly finished before PATCH committed"

    # Release PATCH to commit its transaction
    clone_blocked_verified.set()

    t_patch.join(timeout=10)
    t_clone.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_clone.is_alive()
    assert not thread_errors, f"Thread errors: {thread_errors}"

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["clone_status"] == status.HTTP_201_CREATED

    # Verify CLONE unblocked and received the full updated definition (not a mix)
    clone_data = results["clone_data"]
    assert clone_data["name"] == "Updated Workflow By Patch (copy)"
    assert clone_data["steps"] == [{"op": "select_columns", "columns": ["id", "name", "email"]}]


@pytest.mark.django_db(transaction=True)
def test_concurrency_clone_holds_lock_blocks_patch_contention(
    user_concurrency, workflow_concurrency
):
    """
    Overlapping Concurrency Test 2:
    CLONE acquires select_for_update() row lock first and holds it open.
    PATCH arrives and is proven blocked specifically by CLONE's backend PID via pg_blocking_pids.
    PATCH cannot finish until CLONE commits.
    Once CLONE commits, PATCH unblocks and applies its update.
    CLONE captures the FULL prior workflow definition.
    """
    connection.close()

    clone_in_lock = threading.Event()
    patch_blocked_verified = threading.Event()
    pids = {}
    results = {}
    thread_errors = []

    original_create = WorkflowDefinition.objects.create

    def intercept_clone_create(*args, **kwargs):
        # We are inside CLONE's atomic transaction holding the source row lock
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            pids["clone_pid"] = cursor.fetchone()[0]
        clone_in_lock.set()
        # Wait until main thread verifies PATCH is blocked specifically on our lock
        if not patch_blocked_verified.wait(timeout=5):
            raise TimeoutError("patch_blocked_verified event was not set in time")
        return original_create(*args, **kwargs)

    def do_clone():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            from unittest.mock import patch

            with patch.object(
                WorkflowDefinition.objects, "create", side_effect=intercept_clone_create
            ):
                res = client.post(
                    f"/api/v1/workflows/{workflow_concurrency.id}/clone/",
                    data={},
                    format="json",
                )
                results["clone_status"] = res.status_code
                results["clone_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_patch():
        try:
            connection.close()
            # Wait until CLONE holds the row lock
            if not clone_in_lock.wait(timeout=5):
                raise TimeoutError("clone_in_lock event timed out")
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            # Record our backend pid right before executing patch
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["patch_pid"] = cursor.fetchone()[0]
            # This PATCH will block on select_for_update() row lock held by do_clone
            res = client.patch(
                f"/api/v1/workflows/{workflow_concurrency.id}/",
                data={
                    "name": "Updated Workflow By Patch 2",
                    "steps": [{"op": "select_columns", "columns": ["id"]}],
                },
                format="json",
            )
            results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_clone = threading.Thread(target=do_clone)
    t_patch = threading.Thread(target=do_patch)

    t_clone.start()
    t_patch.start()

    # Wait for CLONE to signal it holds the row lock
    assert clone_in_lock.wait(timeout=5), "CLONE did not acquire lock in time"

    # Wait for PATCH's PID to be registered
    start_wait = time.monotonic()
    while "patch_pid" not in pids and time.monotonic() - start_wait < 5.0:
        time.sleep(0.005)
    assert "patch_pid" in pids, "PATCH PID was not registered"

    # Verify that PATCH is actively blocked by CLONE's backend PID
    assert wait_for_blocked_backend(
        waiter_pid=pids["patch_pid"],
        blocker_pid=pids["clone_pid"],
        timeout=5.0,
    ), f"PATCH (PID {pids['patch_pid']}) did not block on CLONE (PID {pids['clone_pid']})"

    # Assert PATCH thread has NOT finished yet while CLONE holds the lock
    assert t_patch.is_alive(), "PATCH unexpectedly finished before CLONE committed"

    # Release CLONE to commit its transaction
    patch_blocked_verified.set()

    t_clone.join(timeout=10)
    t_patch.join(timeout=10)

    assert not t_clone.is_alive()
    assert not t_patch.is_alive()
    assert not thread_errors, f"Thread errors: {thread_errors}"

    assert results["clone_status"] == status.HTTP_201_CREATED
    assert results["patch_status"] == status.HTTP_200_OK

    # Verify CLONE captured the full prior definition
    clone_data = results["clone_data"]
    assert clone_data["name"] == "Concurrent Test Workflow (copy)"
    assert clone_data["steps"] == [{"op": "select_columns", "columns": ["id", "name"]}]

    # Original now has the post-commit patched values
    workflow_concurrency.refresh_from_db()
    assert workflow_concurrency.name == "Updated Workflow By Patch 2"
    assert workflow_concurrency.steps == [{"op": "select_columns", "columns": ["id"]}]
