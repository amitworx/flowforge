import threading
from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from django.db import connection
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun


@pytest.fixture
def user_a(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="user_a", password="password123")


@pytest.fixture
def sample_job(user_a: User) -> ProcessingJob:
    return ProcessingJob.objects.create(
        owner=user_a,
        file_type="csv",
        state=ProcessingJob.State.COMPLETED,
        original_filename="users.csv",
        input_row_count=3,
        output_row_count=3,
        column_names=["id", "name", "status"],
        output_data=[
            {"id": "001", "name": "Alice", "status": "active"},
            {"id": "002", "name": "Bob", "status": "inactive"},
            {"id": "003", "name": "Charlie", "status": "active"},
        ],
    )


@pytest.fixture
def sample_workflow(user_a: User) -> WorkflowDefinition:
    return WorkflowDefinition.objects.create(
        owner=user_a,
        name="Original Workflow",
        steps=[
            {"op": "select_columns", "columns": ["id", "name"]},
        ],
    )


# -------------------------------------------------------------------------
# Concurrency interleaving tests (Deterministic PostgreSQL)
# -------------------------------------------------------------------------
@pytest.mark.django_db(transaction=True)
def test_concurrent_archive_wins_over_patch_forces_409(user_a, sample_workflow):
    """
    Test scenario: PATCH reads an unarchived workflow. Prior to entering its lock,
    ARCHIVE acquires the row lock, transitions is_archived=True, and commits.
    When PATCH then locks the row and rechecks is_archived inside its transaction,
    it must return 409 Conflict without applying edits to the archived definition.
    """
    connection.close()

    patch_validated = threading.Event()
    archive_done = threading.Event()
    results = {}

    from flowforge.workflows.serializers import WorkflowDefinitionUpdateSerializer

    orig_is_valid = WorkflowDefinitionUpdateSerializer.is_valid

    def intercept_is_valid(self, *args, **kwargs):
        res = orig_is_valid(self, *args, **kwargs)
        # Signal that validation has completed, before atomic transaction
        patch_validated.set()
        # Wait for archive transaction to complete and commit
        assert archive_done.wait(timeout=5)
        return res

    def do_patch():
        connection.close()
        client = APIClient()
        client.force_authenticate(user=user_a)
        with patch.object(
            WorkflowDefinitionUpdateSerializer,
            "is_valid",
            side_effect=intercept_is_valid,
            autospec=True,
        ):
            res = client.patch(
                f"/api/v1/workflows/{sample_workflow.id}/",
                data={"name": "Post-Archive Patch"},
                format="json",
            )
            results["patch_status"] = res.status_code
            results["patch_data"] = res.json()
        connection.close()

    def do_archive():
        connection.close()
        assert patch_validated.wait(timeout=5)
        client = APIClient()
        client.force_authenticate(user=user_a)
        res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
        results["archive_status"] = res.status_code
        archive_done.set()
        connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_arch = threading.Thread(target=do_archive)

    t_patch.start()
    t_arch.start()

    t_patch.join(timeout=10)
    t_arch.join(timeout=10)

    assert results["archive_status"] == status.HTTP_200_OK
    assert results["patch_status"] == status.HTTP_409_CONFLICT
    assert "archived" in results["patch_data"]["detail"].lower()

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is True
    assert sample_workflow.name == "Original Workflow"


@pytest.mark.django_db(transaction=True)
def test_concurrent_patch_wins_over_archive_preserves_edit(user_a, sample_workflow):
    """
    Test scenario: PATCH acquires the row lock first and commits.
    ARCHIVE waits on the lock, and upon acquiring it, archives the updated definition
    preserving the patched changes.
    """
    connection.close()

    patch_in_lock = threading.Event()
    archive_attempted = threading.Event()
    results = {}

    original_save = WorkflowDefinition.save

    def intercept_patch_save(self, *args, **kwargs):
        if self.id == sample_workflow.id and "update_fields" in kwargs:
            # We are inside PATCH save holding row lock
            patch_in_lock.set()
            # Wait until archive is blocked attempting to lock the same row
            archive_attempted.wait(timeout=0.2)
        return original_save(self, *args, **kwargs)

    def do_patch():
        connection.close()
        client = APIClient()
        client.force_authenticate(user=user_a)
        with patch.object(
            WorkflowDefinition, "save", autospec=True, side_effect=intercept_patch_save
        ):
            res = client.patch(
                f"/api/v1/workflows/{sample_workflow.id}/",
                data={"name": "Pre-Archive Patched Name"},
                format="json",
            )
            results["patch_status"] = res.status_code
        connection.close()

    def do_archive():
        connection.close()
        assert patch_in_lock.wait(timeout=5)
        archive_attempted.set()
        client = APIClient()
        client.force_authenticate(user=user_a)
        res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
        results["archive_status"] = res.status_code
        connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_arch = threading.Thread(target=do_archive)

    t_patch.start()
    t_arch.start()

    t_patch.join(timeout=10)
    t_arch.join(timeout=10)

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["archive_status"] == status.HTTP_200_OK

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is True
    assert sample_workflow.name == "Pre-Archive Patched Name"


@pytest.mark.django_db(transaction=True)
def test_concurrent_archive_and_restore_linearizable(user_a, sample_workflow):
    """
    Test scenario: ARCHIVE and RESTORE invoked concurrently with barriers.
    Because both lock unconditionally and read fresh state inside the transaction,
    the final state of the database and the response of whichever committed last
    must be completely consistent and truthful.
    """
    connection.close()

    barrier = threading.Barrier(2)
    results = {}

    def do_archive():
        connection.close()
        client = APIClient()
        client.force_authenticate(user=user_a)
        barrier.wait()
        res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
        results["archive"] = res.json()
        connection.close()

    def do_restore():
        connection.close()
        client = APIClient()
        client.force_authenticate(user=user_a)
        barrier.wait()
        res = client.post(f"/api/v1/workflows/{sample_workflow.id}/restore/")
        results["restore"] = res.json()
        connection.close()

    t1 = threading.Thread(target=do_archive)
    t2 = threading.Thread(target=do_restore)

    t1.start()
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    sample_workflow.refresh_from_db()
    if sample_workflow.is_archived:
        assert sample_workflow.archived_at is not None
        assert results["archive"]["is_archived"] is True
    else:
        assert sample_workflow.archived_at is None
        assert results["restore"]["is_archived"] is False


@pytest.mark.django_db(transaction=True)
def test_concurrent_patch_and_run_takes_consistent_snapshot(user_a, sample_workflow, sample_job):
    """
    Test scenario: Concurrent PATCH and run creation.
    When run creation executes, it locks the workflow definition row to snapshot steps.
    The resulting WorkflowRun must have either the pre-patch or post-patch snapshot,
    never a partial or corrupted snapshot.
    """
    connection.close()

    barrier = threading.Barrier(2)
    results = {}

    def do_patch():
        connection.close()
        client = APIClient()
        client.force_authenticate(user=user_a)
        barrier.wait()
        res = client.patch(
            f"/api/v1/workflows/{sample_workflow.id}/",
            data={
                "name": "Step Modified WF",
                "steps": [{"op": "select_columns", "columns": ["id"]}],
            },
            format="json",
        )
        results["patch"] = res.status_code
        connection.close()

    def do_run():
        connection.close()
        client = APIClient()
        client.force_authenticate(user=user_a)
        barrier.wait()
        res = client.post(
            f"/api/v1/workflows/{sample_workflow.id}/runs/",
            data={"source_job_id": str(sample_job.id)},
            format="json",
        )
        results["run"] = res.status_code
        results["run_data"] = res.json()
        connection.close()

    t1 = threading.Thread(target=do_patch)
    t2 = threading.Thread(target=do_run)

    t1.start()
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results["patch"] == status.HTTP_200_OK
    assert results["run"] == status.HTTP_201_CREATED

    run = WorkflowRun.objects.get(id=results["run_data"]["id"])
    assert run.step_snapshot in (
        [{"op": "select_columns", "columns": ["id", "name"]}],
        [{"op": "select_columns", "columns": ["id"]}],
    )
    assert run.state == WorkflowRun.State.COMPLETED
