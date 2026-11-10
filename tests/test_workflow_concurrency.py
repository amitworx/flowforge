import threading
import time
from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from django.db import connection
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun
from flowforge.workflows.serializers import WorkflowDefinitionUpdateSerializer


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


# -------------------------------------------------------------------------
# Concurrency interleaving tests (Deterministic PostgreSQL proofs)
# -------------------------------------------------------------------------


@pytest.mark.django_db(transaction=True)
def test_concurrent_archive_wins_over_patch_forces_409(user_a, sample_workflow):
    """
    Ordering Proof: ARCHIVE -> PATCH
    Test scenario: PATCH reads an unarchived workflow. Prior to entering its lock,
    ARCHIVE acquires the row lock, transitions is_archived=True, and commits.
    When PATCH then locks the row and rechecks is_archived inside its transaction,
    it must return 409 Conflict without applying edits to the archived definition.
    """
    connection.close()

    patch_validated = threading.Event()
    archive_done = threading.Event()
    results = {}
    thread_errors = []

    orig_is_valid = WorkflowDefinitionUpdateSerializer.is_valid

    def intercept_is_valid(self, *args, **kwargs):
        res = orig_is_valid(self, *args, **kwargs)
        # Signal that validation has completed, before atomic transaction
        patch_validated.set()
        # Wait for archive transaction to complete and commit
        if not archive_done.wait(timeout=5):
            raise TimeoutError("archive_done event was not set in time")
        return res

    def do_patch():
        try:
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
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_archive():
        try:
            connection.close()
            if not patch_validated.wait(timeout=5):
                raise TimeoutError("patch_validated event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
            results["archive_status"] = res.status_code
            archive_done.set()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_arch = threading.Thread(target=do_archive)

    t_patch.start()
    t_arch.start()

    t_patch.join(timeout=10)
    t_arch.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_arch.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["archive_status"] == status.HTTP_200_OK
    assert results["patch_status"] == status.HTTP_409_CONFLICT
    assert "archived" in results["patch_data"]["detail"].lower()

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is True
    assert sample_workflow.name == "Original Workflow"


@pytest.mark.django_db(transaction=True)
def test_concurrent_patch_wins_over_archive_preserves_edit(user_a, sample_workflow):
    """
    Ordering Proof: PATCH -> ARCHIVE
    Test scenario: PATCH acquires the row lock first.
    ARCHIVE arrives and is deterministically proven blocked at the database lock
    (via pg_locks query) while PATCH is in-flight.
    Once PATCH commits, ARCHIVE unblocks and archives the definition,
    preserving the patched name and steps.
    """
    connection.close()

    patch_in_lock = threading.Event()
    archive_blocked = threading.Event()
    results = {}
    thread_errors = []

    original_save = WorkflowDefinition.save

    def intercept_patch_save(self, *args, **kwargs):
        if self.id == sample_workflow.id and "update_fields" in kwargs:
            # We are inside PATCH holding the row lock
            patch_in_lock.set()
            # Wait until ARCHIVE is deterministically confirmed blocked on this row lock
            if not archive_blocked.wait(timeout=5):
                raise TimeoutError("archive_blocked event was not set in time")
        return original_save(self, *args, **kwargs)

    def do_patch():
        try:
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
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_archive():
        try:
            connection.close()
            # Wait for PATCH to acquire lock
            if not patch_in_lock.wait(timeout=5):
                raise TimeoutError("patch_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            # This call will block on the row lock held by do_patch
            res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
            results["archive_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_arch = threading.Thread(target=do_archive)

    t_patch.start()
    t_arch.start()

    # Main thread verifies via PostgreSQL engine that ARCHIVE is blocked waiting on lock
    assert wait_for_blocked_db_lock(timeout=5.0), (
        "ARCHIVE did not block on PostgreSQL lock as expected"
    )
    archive_blocked.set()

    t_patch.join(timeout=10)
    t_arch.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_arch.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["archive_status"] == status.HTTP_200_OK

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is True
    assert sample_workflow.name == "Pre-Archive Patched Name"


@pytest.mark.django_db(transaction=True)
def test_concurrent_archive_then_restore_forced_ordering(user_a, sample_workflow):
    """
    Ordering Proof: ARCHIVE -> RESTORE
    Initial state: is_archived=False.
    ARCHIVE acquires lock and changes is_archived to True.
    RESTORE is deterministically blocked at PostgreSQL row lock.
    When ARCHIVE commits, RESTORE acquires lock, reads fresh state, and sets is_archived to False.
    Final state: is_archived=False, archived_at=None.
    """
    connection.close()
    assert sample_workflow.is_archived is False

    archive_in_lock = threading.Event()
    restore_blocked = threading.Event()
    results = {}
    thread_errors = []

    original_save = WorkflowDefinition.save

    def intercept_archive_save(self, *args, **kwargs):
        if self.id == sample_workflow.id and "is_archived" in kwargs.get("update_fields", []):
            archive_in_lock.set()
            if not restore_blocked.wait(timeout=5):
                raise TimeoutError("restore_blocked event was not set in time")
        return original_save(self, *args, **kwargs)

    def do_archive():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition, "save", autospec=True, side_effect=intercept_archive_save
            ):
                res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
                results["archive_res"] = res.json()
                results["archive_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_restore():
        try:
            connection.close()
            if not archive_in_lock.wait(timeout=5):
                raise TimeoutError("archive_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            # Blocks on row lock held by do_archive
            res = client.post(f"/api/v1/workflows/{sample_workflow.id}/restore/")
            results["restore_res"] = res.json()
            results["restore_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_arch = threading.Thread(target=do_archive)
    t_rest = threading.Thread(target=do_restore)

    t_arch.start()
    t_rest.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "RESTORE did not block on PostgreSQL lock"
    restore_blocked.set()

    t_arch.join(timeout=10)
    t_rest.join(timeout=10)

    assert not t_arch.is_alive()
    assert not t_rest.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["archive_status"] == status.HTTP_200_OK
    assert results["restore_status"] == status.HTTP_200_OK
    assert results["archive_res"]["is_archived"] is True
    assert results["restore_res"]["is_archived"] is False

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is False
    assert sample_workflow.archived_at is None


@pytest.mark.django_db(transaction=True)
def test_concurrent_restore_then_archive_forced_ordering(user_a, sample_workflow):
    """
    Ordering Proof: RESTORE -> ARCHIVE
    Initial state: is_archived=True.
    RESTORE acquires lock and changes is_archived to False.
    ARCHIVE is deterministically blocked at PostgreSQL row lock.
    When RESTORE commits, ARCHIVE acquires lock, reads fresh state, and sets is_archived to True.
    Final state: is_archived=True, archived_at is not None.
    """
    connection.close()
    sample_workflow.is_archived = True
    sample_workflow.archived_at = timezone.now()
    sample_workflow.save(update_fields=["is_archived", "archived_at"])

    restore_in_lock = threading.Event()
    archive_blocked = threading.Event()
    results = {}
    thread_errors = []

    original_save = WorkflowDefinition.save

    def intercept_restore_save(self, *args, **kwargs):
        if self.id == sample_workflow.id and "is_archived" in kwargs.get("update_fields", []):
            restore_in_lock.set()
            if not archive_blocked.wait(timeout=5):
                raise TimeoutError("archive_blocked event was not set in time")
        return original_save(self, *args, **kwargs)

    def do_restore():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition, "save", autospec=True, side_effect=intercept_restore_save
            ):
                res = client.post(f"/api/v1/workflows/{sample_workflow.id}/restore/")
                results["restore_res"] = res.json()
                results["restore_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_archive():
        try:
            connection.close()
            if not restore_in_lock.wait(timeout=5):
                raise TimeoutError("restore_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            # Blocks on row lock held by do_restore
            res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
            results["archive_res"] = res.json()
            results["archive_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_rest = threading.Thread(target=do_restore)
    t_arch = threading.Thread(target=do_archive)

    t_rest.start()
    t_arch.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "ARCHIVE did not block on PostgreSQL lock"
    archive_blocked.set()

    t_rest.join(timeout=10)
    t_arch.join(timeout=10)

    assert not t_rest.is_alive()
    assert not t_arch.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["restore_status"] == status.HTTP_200_OK
    assert results["archive_status"] == status.HTTP_200_OK
    assert results["restore_res"]["is_archived"] is False
    assert results["archive_res"]["is_archived"] is True

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is True
    assert sample_workflow.archived_at is not None


@pytest.mark.django_db(transaction=True)
def test_concurrent_patch_then_sync_run_uses_post_patch_snapshot(
    user_a, sample_workflow, sample_job
):
    """
    Ordering Proof: PATCH -> RUN (Sync)
    PATCH acquires row lock to update steps. RUN arrives and blocks on row lock.
    When PATCH commits new steps, RUN unblocks, snapshots the NEW steps, and completes.
    """
    connection.close()

    patch_in_lock = threading.Event()
    run_blocked = threading.Event()
    results = {}
    thread_errors = []

    original_save = WorkflowDefinition.save

    def intercept_patch_save(self, *args, **kwargs):
        if self.id == sample_workflow.id and "update_fields" in kwargs:
            patch_in_lock.set()
            if not run_blocked.wait(timeout=5):
                raise TimeoutError("run_blocked event was not set in time")
        return original_save(self, *args, **kwargs)

    def do_patch():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition, "save", autospec=True, side_effect=intercept_patch_save
            ):
                res = client.patch(
                    f"/api/v1/workflows/{sample_workflow.id}/",
                    data={"steps": [{"op": "select_columns", "columns": ["id"]}]},
                    format="json",
                )
                results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_run():
        try:
            connection.close()
            if not patch_in_lock.wait(timeout=5):
                raise TimeoutError("patch_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.post(
                f"/api/v1/workflows/{sample_workflow.id}/runs/",
                data={"source_job_id": str(sample_job.id)},
                format="json",
            )
            results["run_status"] = res.status_code
            results["run_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_run = threading.Thread(target=do_run)

    t_patch.start()
    t_run.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "RUN did not block on PostgreSQL lock"
    run_blocked.set()

    t_patch.join(timeout=10)
    t_run.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_run.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["run_status"] == status.HTTP_201_CREATED

    run = WorkflowRun.objects.get(id=results["run_data"]["id"])
    assert run.step_snapshot == [{"op": "select_columns", "columns": ["id"]}]
    assert run.output_columns == ["id"]
    assert run.state == WorkflowRun.State.COMPLETED


@pytest.mark.django_db(transaction=True)
def test_concurrent_sync_run_then_patch_uses_pre_patch_snapshot(
    user_a, sample_workflow, sample_job
):
    """
    Ordering Proof: RUN (Sync) -> PATCH
    RUN acquires row lock first and creates execution using initial steps.
    PATCH arrives and blocks on row lock.
    When RUN commits, PATCH unblocks and updates steps.
    The created run strictly retains the original pre-patch snapshot.
    """
    connection.close()

    run_in_lock = threading.Event()
    patch_blocked = threading.Event()
    results = {}
    thread_errors = []

    original_run_create = WorkflowRun.objects.create

    def intercept_run_create(*args, **kwargs):
        # We are inside the atomic transaction holding select_for_update() on workflow definition
        run_obj = original_run_create(*args, **kwargs)
        run_in_lock.set()
        if not patch_blocked.wait(timeout=5):
            raise TimeoutError("patch_blocked event was not set in time")
        return run_obj

    def do_run():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(WorkflowRun.objects, "create", side_effect=intercept_run_create):
                res = client.post(
                    f"/api/v1/workflows/{sample_workflow.id}/runs/",
                    data={"source_job_id": str(sample_job.id)},
                    format="json",
                )
                results["run_status"] = res.status_code
                results["run_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_patch():
        try:
            connection.close()
            if not run_in_lock.wait(timeout=5):
                raise TimeoutError("run_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.patch(
                f"/api/v1/workflows/{sample_workflow.id}/",
                data={"steps": [{"op": "select_columns", "columns": ["name"]}]},
                format="json",
            )
            results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_run = threading.Thread(target=do_run)
    t_patch = threading.Thread(target=do_patch)

    t_run.start()
    t_patch.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "PATCH did not block on PostgreSQL lock"
    patch_blocked.set()

    t_run.join(timeout=10)
    t_patch.join(timeout=10)

    assert not t_run.is_alive()
    assert not t_patch.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["run_status"] == status.HTTP_201_CREATED
    assert results["patch_status"] == status.HTTP_200_OK

    run = WorkflowRun.objects.get(id=results["run_data"]["id"])
    # Run used the original snapshot before PATCH
    assert run.step_snapshot == [{"op": "select_columns", "columns": ["id", "name"]}]

    # Workflow definition in DB now has the updated steps
    sample_workflow.refresh_from_db()
    assert sample_workflow.steps == [{"op": "select_columns", "columns": ["name"]}]


@pytest.mark.django_db(transaction=True)
def test_concurrent_patch_then_async_run_uses_post_patch_snapshot(
    user_a, sample_workflow, sample_job
):
    """
    Ordering Proof: PATCH -> ASYNC RUN
    PATCH holds row lock to update steps. ASYNC RUN dispatch arrives and blocks on row lock.
    When PATCH commits, ASYNC RUN unblocks, captures the post-patch steps snapshot,
    and returns 202 Accepted.
    Only Celery broker dispatch is mocked to prevent external task execution during test.
    """
    connection.close()

    patch_in_lock = threading.Event()
    async_run_blocked = threading.Event()
    results = {}
    thread_errors = []

    original_save = WorkflowDefinition.save

    def intercept_patch_save(self, *args, **kwargs):
        if self.id == sample_workflow.id and "update_fields" in kwargs:
            patch_in_lock.set()
            if not async_run_blocked.wait(timeout=5):
                raise TimeoutError("async_run_blocked event was not set in time")
        return original_save(self, *args, **kwargs)

    def do_patch():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition, "save", autospec=True, side_effect=intercept_patch_save
            ):
                res = client.patch(
                    f"/api/v1/workflows/{sample_workflow.id}/",
                    data={"steps": [{"op": "select_columns", "columns": ["id"]}]},
                    format="json",
                )
                results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_async_run():
        try:
            connection.close()
            if not patch_in_lock.wait(timeout=5):
                raise TimeoutError("patch_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch("flowforge.workflows.views.execute_workflow_run_task.delay"):
                res = client.post(
                    f"/api/v1/workflows/{sample_workflow.id}/runs/async/",
                    data={"source_job_id": str(sample_job.id)},
                    format="json",
                )
                results["async_run_status"] = res.status_code
                results["async_run_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_async = threading.Thread(target=do_async_run)

    t_patch.start()
    t_async.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "ASYNC RUN did not block on PostgreSQL lock"
    async_run_blocked.set()

    t_patch.join(timeout=10)
    t_async.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_async.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["async_run_status"] == status.HTTP_202_ACCEPTED

    run = WorkflowRun.objects.get(id=results["async_run_data"]["id"])
    assert run.step_snapshot == [{"op": "select_columns", "columns": ["id"]}]
    assert run.state == WorkflowRun.State.QUEUED


@pytest.mark.django_db(transaction=True)
def test_concurrent_patch_then_preview_uses_post_patch_snapshot(
    user_a, sample_workflow, sample_job
):
    """
    Ordering Proof: PATCH -> PREVIEW
    Test scenario: PATCH acquires the row lock to update definition steps.
    PREVIEW arrives and blocks on the PostgreSQL row lock.
    Once PATCH commits, PREVIEW unblocks, snapshots the updated definition,
    and returns preview results reflecting the new steps.
    """
    patch_in_lock = threading.Event()
    preview_blocked = threading.Event()

    orig_save = WorkflowDefinition.save

    def intercept_patch_save(self, *args, **kwargs):
        if getattr(self, "id", None) == sample_workflow.id:
            patch_in_lock.set()
            if not preview_blocked.wait(timeout=5):
                raise TimeoutError("preview_blocked event was not set in time")
        return orig_save(self, *args, **kwargs)

    results = {}
    thread_errors = []

    def do_patch():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition, "save", autospec=True, side_effect=intercept_patch_save
            ):
                res = client.patch(
                    f"/api/v1/workflows/{sample_workflow.id}/",
                    data={"steps": [{"op": "select_columns", "columns": ["name"]}]},
                    format="json",
                )
                results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_preview():
        try:
            connection.close()
            if not patch_in_lock.wait(timeout=5):
                raise TimeoutError("patch_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.post(
                f"/api/v1/workflows/{sample_workflow.id}/preview/",
                data={"source_job_id": str(sample_job.id)},
                format="json",
            )
            results["preview_status"] = res.status_code
            results["preview_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_patch = threading.Thread(target=do_patch)
    t_preview = threading.Thread(target=do_preview)

    t_patch.start()
    t_preview.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "PREVIEW did not block on PostgreSQL lock"
    preview_blocked.set()

    t_patch.join(timeout=10)
    t_preview.join(timeout=10)

    assert not t_patch.is_alive()
    assert not t_preview.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["patch_status"] == status.HTTP_200_OK
    assert results["preview_status"] == status.HTTP_200_OK
    assert results["preview_data"]["output_columns"] == ["name"]
    assert results["preview_data"]["preview_rows"][0] == {"name": "Alice"}


@pytest.mark.django_db(transaction=True)
def test_concurrent_preview_then_patch_uses_pre_patch_snapshot(user_a, sample_workflow, sample_job):
    """
    Ordering Proof: PREVIEW -> PATCH
    Test scenario: PREVIEW acquires the row lock to snapshot the active definition.
    PATCH arrives and blocks on the PostgreSQL row lock.
    PREVIEW finishes taking its snapshot and completes.
    PATCH unblocks and commits the edit.
    The preview output reflects the pre-patch definition.
    """
    preview_in_lock = threading.Event()
    patch_blocked = threading.Event()

    orig_select = WorkflowDefinition.objects.select_for_update

    def intercept_select(*args, **kwargs):
        qs = orig_select(*args, **kwargs)
        orig_get = qs.get

        def intercept_get(*g_args, **g_kwargs):
            obj = orig_get(*g_args, **g_kwargs)
            if getattr(obj, "id", None) == sample_workflow.id:
                preview_in_lock.set()
                if not patch_blocked.wait(timeout=5):
                    raise TimeoutError("patch_blocked event was not set in time")
            return obj

        qs.get = intercept_get
        return qs

    results = {}
    thread_errors = []

    def do_preview():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition.objects,
                "select_for_update",
                side_effect=intercept_select,
            ):
                res = client.post(
                    f"/api/v1/workflows/{sample_workflow.id}/preview/",
                    data={"source_job_id": str(sample_job.id)},
                    format="json",
                )
                results["preview_status"] = res.status_code
                results["preview_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_patch():
        try:
            connection.close()
            if not preview_in_lock.wait(timeout=5):
                raise TimeoutError("preview_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.patch(
                f"/api/v1/workflows/{sample_workflow.id}/",
                data={"steps": [{"op": "select_columns", "columns": ["name"]}]},
                format="json",
            )
            results["patch_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_preview = threading.Thread(target=do_preview)
    t_patch = threading.Thread(target=do_patch)

    t_preview.start()
    t_patch.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "PATCH did not block on PostgreSQL lock"
    patch_blocked.set()

    t_preview.join(timeout=10)
    t_patch.join(timeout=10)

    assert not t_preview.is_alive()
    assert not t_patch.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["preview_status"] == status.HTTP_200_OK
    assert results["patch_status"] == status.HTTP_200_OK

    # Pre-patch had columns ['id', 'name']
    assert results["preview_data"]["output_columns"] == ["id", "name"]
    assert results["preview_data"]["preview_rows"][0] == {"id": "001", "name": "Alice"}


@pytest.mark.django_db(transaction=True)
def test_concurrent_archive_wins_over_preview_forces_409(user_a, sample_workflow, sample_job):
    """
    Ordering Proof: ARCHIVE -> PREVIEW
    Test scenario: ARCHIVE acquires the row lock. PREVIEW arrives and blocks on the lock.
    ARCHIVE commits is_archived=True.
    PREVIEW unblocks inside its lock, detects the archived state, and returns 409 Conflict.
    """
    archive_in_lock = threading.Event()
    preview_blocked = threading.Event()

    orig_save = WorkflowDefinition.save

    def intercept_archive_save(self, *args, **kwargs):
        if getattr(self, "id", None) == sample_workflow.id and self.is_archived:
            archive_in_lock.set()
            if not preview_blocked.wait(timeout=5):
                raise TimeoutError("preview_blocked event was not set in time")
        return orig_save(self, *args, **kwargs)

    results = {}
    thread_errors = []

    def do_archive():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition, "save", autospec=True, side_effect=intercept_archive_save
            ):
                res = client.post(
                    f"/api/v1/workflows/{sample_workflow.id}/archive/",
                    format="json",
                )
                results["archive_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_preview():
        try:
            connection.close()
            if not archive_in_lock.wait(timeout=5):
                raise TimeoutError("archive_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.post(
                f"/api/v1/workflows/{sample_workflow.id}/preview/",
                data={"source_job_id": str(sample_job.id)},
                format="json",
            )
            results["preview_status"] = res.status_code
            results["preview_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_archive = threading.Thread(target=do_archive)
    t_preview = threading.Thread(target=do_preview)

    t_archive.start()
    t_preview.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "PREVIEW did not block on PostgreSQL lock"
    preview_blocked.set()

    t_archive.join(timeout=10)
    t_preview.join(timeout=10)

    assert not t_archive.is_alive()
    assert not t_preview.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["archive_status"] == status.HTTP_200_OK
    assert results["preview_status"] == status.HTTP_409_CONFLICT
    assert "archived" in results["preview_data"]["detail"].lower()


@pytest.mark.django_db(transaction=True)
def test_concurrent_preview_wins_before_archive_completes_safely(
    user_a, sample_workflow, sample_job
):
    """
    Ordering Proof: PREVIEW -> ARCHIVE
    Test scenario: PREVIEW acquires the row lock and snapshots definition.
    ARCHIVE arrives and blocks on the PostgreSQL lock.
    PREVIEW commits/releases lock and completes execution.
    ARCHIVE unblocks and archives the definition.
    """
    preview_in_lock = threading.Event()
    archive_blocked = threading.Event()

    orig_select = WorkflowDefinition.objects.select_for_update

    def intercept_select(*args, **kwargs):
        qs = orig_select(*args, **kwargs)
        orig_get = qs.get

        def intercept_get(*g_args, **g_kwargs):
            obj = orig_get(*g_args, **g_kwargs)
            if getattr(obj, "id", None) == sample_workflow.id:
                preview_in_lock.set()
                if not archive_blocked.wait(timeout=5):
                    raise TimeoutError("archive_blocked event was not set in time")
            return obj

        qs.get = intercept_get
        return qs

    results = {}
    thread_errors = []

    def do_preview():
        try:
            connection.close()
            client = APIClient()
            client.force_authenticate(user=user_a)
            with patch.object(
                WorkflowDefinition.objects,
                "select_for_update",
                side_effect=intercept_select,
            ):
                res = client.post(
                    f"/api/v1/workflows/{sample_workflow.id}/preview/",
                    data={"source_job_id": str(sample_job.id)},
                    format="json",
                )
                results["preview_status"] = res.status_code
                results["preview_data"] = res.json()
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    def do_archive():
        try:
            connection.close()
            if not preview_in_lock.wait(timeout=5):
                raise TimeoutError("preview_in_lock event was not set in time")
            client = APIClient()
            client.force_authenticate(user=user_a)
            res = client.post(
                f"/api/v1/workflows/{sample_workflow.id}/archive/",
                format="json",
            )
            results["archive_status"] = res.status_code
        except Exception as exc:
            thread_errors.append(exc)
        finally:
            connection.close()

    t_preview = threading.Thread(target=do_preview)
    t_archive = threading.Thread(target=do_archive)

    t_preview.start()
    t_archive.start()

    assert wait_for_blocked_db_lock(timeout=5.0), "ARCHIVE did not block on PostgreSQL lock"
    archive_blocked.set()

    t_preview.join(timeout=10)
    t_archive.join(timeout=10)

    assert not t_preview.is_alive()
    assert not t_archive.is_alive()
    assert not thread_errors, f"Exceptions in background threads: {thread_errors}"

    assert results["preview_status"] == status.HTTP_200_OK
    assert results["archive_status"] == status.HTTP_200_OK

    # Preview succeeded with active definition
    assert results["preview_data"]["output_columns"] == ["id", "name"]
    # Final database state is archived
    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is True
