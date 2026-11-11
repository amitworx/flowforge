import threading

import pytest
from django.db import connection
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.workflows.models import WorkflowDefinition
from flowforge.workflows.serializers import (
    WorkflowCloneRequestSerializer,
    WorkflowDefinitionUpdateSerializer,
)


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


@pytest.mark.django_db(transaction=True)
def test_concurrency_patch_before_clone_serializes_cleanly(user_concurrency, workflow_concurrency):
    """
    Ordering Proof 1: PATCH commits before CLONE acquires lock.
    Clone MUST capture the full updated definition (new name and new steps).
    """
    connection.close()

    patch_committed = threading.Event()
    results = {}
    thread_errors = []

    def do_patch():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            res = client.patch(
                f"/api/v1/workflows/{workflow_concurrency.id}/",
                data={
                    "name": "Updated Workflow Name",
                    "steps": [
                        {"op": "select_columns", "columns": ["id", "name", "email"]},
                    ],
                },
                format="json",
            )
            results["patch_status"] = res.status_code
            patch_committed.set()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    orig_is_valid = WorkflowCloneRequestSerializer.is_valid

    def intercept_clone_is_valid(self, *args, **kwargs):
        res = orig_is_valid(self, *args, **kwargs)
        if not patch_committed.wait(timeout=5):
            raise TimeoutError("patch_committed event timed out")
        return res

    def do_clone():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(WorkflowCloneRequestSerializer, "is_valid", intercept_clone_is_valid)
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

    t_patch.join(timeout=10)
    t_clone.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_clone.is_alive()
    assert not thread_errors, f"Thread errors: {thread_errors}"

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["clone_status"] == status.HTTP_201_CREATED

    # Clone saw the newly committed definition
    clone_data = results["clone_data"]
    assert clone_data["name"] == "Updated Workflow Name (copy)"
    assert clone_data["steps"] == [{"op": "select_columns", "columns": ["id", "name", "email"]}]


@pytest.mark.django_db(transaction=True)
def test_concurrency_clone_before_patch_serializes_cleanly(user_concurrency, workflow_concurrency):
    """
    Ordering Proof 2: CLONE acquires lock and commits before PATCH applies updates.
    Clone MUST capture the full prior definition (original name and original steps).
    """
    connection.close()

    clone_committed = threading.Event()
    results = {}
    thread_errors = []

    def do_clone():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            res = client.post(
                f"/api/v1/workflows/{workflow_concurrency.id}/clone/",
                data={},
                format="json",
            )
            results["clone_status"] = res.status_code
            results["clone_data"] = res.json()
            clone_committed.set()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    orig_patch_is_valid = WorkflowDefinitionUpdateSerializer.is_valid

    def intercept_patch_is_valid(self, *args, **kwargs):
        res = orig_patch_is_valid(self, *args, **kwargs)
        if not clone_committed.wait(timeout=5):
            raise TimeoutError("clone_committed event timed out")
        return res

    def do_patch():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_concurrency)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(WorkflowDefinitionUpdateSerializer, "is_valid", intercept_patch_is_valid)
                res = client.patch(
                    f"/api/v1/workflows/{workflow_concurrency.id}/",
                    data={
                        "name": "Updated Workflow Name 2",
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

    t_clone.join(timeout=10)
    t_patch.join(timeout=10)

    assert not t_clone.is_alive()
    assert not t_patch.is_alive()
    assert not thread_errors, f"Thread errors: {thread_errors}"

    assert results["clone_status"] == status.HTTP_201_CREATED
    assert results["patch_status"] == status.HTTP_200_OK

    # Clone saw the prior definition
    clone_data = results["clone_data"]
    assert clone_data["name"] == "Concurrent Test Workflow (copy)"
    assert clone_data["steps"] == [{"op": "select_columns", "columns": ["id", "name"]}]

    # Now original workflow has the updated definition
    workflow_concurrency.refresh_from_db()
    assert workflow_concurrency.name == "Updated Workflow Name 2"
    assert workflow_concurrency.steps == [{"op": "select_columns", "columns": ["id"]}]
