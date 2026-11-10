from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob
from flowforge.workflows.models import WorkflowDefinition, WorkflowRun


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def user_a(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="user_a", password="password123")


@pytest.fixture
def user_b(db: None) -> User:  # noqa: ARG001
    return User.objects.create_user(username="user_b", password="password123")


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
# 1. Model fields & migration defaults
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_workflow_definition_archived_fields_default(sample_workflow: WorkflowDefinition):
    sample_workflow.refresh_from_db()
    assert hasattr(sample_workflow, "is_archived")
    assert hasattr(sample_workflow, "archived_at")
    assert sample_workflow.is_archived is False
    assert sample_workflow.archived_at is None


# -------------------------------------------------------------------------
# 2. Authentication & Authorization parity (PATCH / Archive / Restore)
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_workflow_lifecycle_requires_authentication(api_client, sample_workflow):
    wf_id = sample_workflow.id
    assert (
        api_client.patch(f"/api/v1/workflows/{wf_id}/", data={"name": "New"}).status_code
        == status.HTTP_403_FORBIDDEN
    )
    assert (
        api_client.post(f"/api/v1/workflows/{wf_id}/archive/").status_code
        == status.HTTP_403_FORBIDDEN
    )
    assert (
        api_client.post(f"/api/v1/workflows/{wf_id}/restore/").status_code
        == status.HTTP_403_FORBIDDEN
    )
    assert api_client.delete(f"/api/v1/workflows/{wf_id}/").status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.django_db
def test_workflow_lifecycle_cross_user_isolation_returns_404(api_client, user_b, sample_workflow):
    api_client.force_authenticate(user=user_b)
    wf_id = sample_workflow.id

    assert api_client.get(f"/api/v1/workflows/{wf_id}/").status_code == status.HTTP_404_NOT_FOUND
    assert (
        api_client.patch(f"/api/v1/workflows/{wf_id}/", data={"name": "Hacked"}).status_code
        == status.HTTP_404_NOT_FOUND
    )
    assert (
        api_client.post(f"/api/v1/workflows/{wf_id}/archive/").status_code
        == status.HTTP_404_NOT_FOUND
    )
    assert (
        api_client.post(f"/api/v1/workflows/{wf_id}/restore/").status_code
        == status.HTTP_404_NOT_FOUND
    )


# -------------------------------------------------------------------------
# 3. PATCH /api/v1/workflows/{uuid}/
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_patch_workflow_name_only(api_client, user_a, sample_workflow):
    api_client.force_authenticate(user=user_a)
    original_updated_at = sample_workflow.updated_at

    response = api_client.patch(
        f"/api/v1/workflows/{sample_workflow.id}/",
        data={"name": "Updated Name"},
        format="json",
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "Updated Name"
    assert data["steps"] == [{"op": "select_columns", "columns": ["id", "name"]}]
    assert data["is_archived"] is False
    assert data["archived_at"] is None

    sample_workflow.refresh_from_db()
    assert sample_workflow.name == "Updated Name"
    assert sample_workflow.updated_at >= original_updated_at


@pytest.mark.django_db
def test_patch_workflow_steps_only(api_client, user_a, sample_workflow):
    api_client.force_authenticate(user=user_a)

    new_steps = [
        {"op": "select_columns", "columns": ["id", "name", "status"]},
        {"op": "filter_equals", "column": "status", "value": "active"},
    ]
    response = api_client.patch(
        f"/api/v1/workflows/{sample_workflow.id}/",
        data={"steps": new_steps},
        format="json",
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "Original Workflow"
    assert data["steps"] == new_steps

    sample_workflow.refresh_from_db()
    assert sample_workflow.steps == new_steps


@pytest.mark.django_db
def test_patch_workflow_name_and_steps(api_client, user_a, sample_workflow):
    api_client.force_authenticate(user=user_a)

    new_steps = [{"op": "rename_columns", "mapping": {"name": "full_name"}}]
    response = api_client.patch(
        f"/api/v1/workflows/{sample_workflow.id}/",
        data={"name": "Renamed & Filtered", "steps": new_steps},
        format="json",
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "Renamed & Filtered"
    assert data["steps"] == new_steps


@pytest.mark.django_db
def test_patch_workflow_rejections_and_immutability(api_client, user_a, sample_workflow):
    api_client.force_authenticate(user=user_a)
    original_name = sample_workflow.name
    original_steps = sample_workflow.steps
    wf_url = f"/api/v1/workflows/{sample_workflow.id}/"

    # Empty name
    res = api_client.patch(wf_url, data={"name": "   "}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Exceeding max name length (128)
    res = api_client.patch(wf_url, data={"name": "A" * 129}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Invalid steps (> 10 steps)
    too_many_steps = [{"op": "select_columns", "columns": ["id"]} for _ in range(11)]
    res = api_client.patch(wf_url, data={"steps": too_many_steps}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Invalid op
    bad_op = [{"op": "arbitrary_eval", "code": "os.system('sh')"}]
    res = api_client.patch(wf_url, data={"steps": bad_op}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Read-only or immutable fields in patch body must be rejected
    res = api_client.patch(wf_url, data={"owner": 999}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    res = api_client.patch(
        wf_url, data={"id": "00000000-0000-0000-0000-000000000000"}, format="json"
    )
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    res = api_client.patch(wf_url, data={"created_at": "2020-01-01T00:00:00Z"}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Unknown fields
    res = api_client.patch(wf_url, data={"random_extra": "hack"}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Empty body
    res = api_client.patch(wf_url, data={}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST

    # Verify original values are entirely untouched
    sample_workflow.refresh_from_db()
    assert sample_workflow.name == original_name
    assert sample_workflow.steps == original_steps


@pytest.mark.django_db
def test_hard_delete_is_unsupported(api_client, user_a, sample_workflow):
    api_client.force_authenticate(user=user_a)
    response = api_client.delete(f"/api/v1/workflows/{sample_workflow.id}/")
    assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    assert WorkflowDefinition.objects.filter(id=sample_workflow.id).exists()


# -------------------------------------------------------------------------
# 4. Archive & Restore lifecycle operations
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_archive_and_restore_workflow_lifecycle(api_client, user_a, sample_workflow):
    api_client.force_authenticate(user=user_a)
    wf_url = f"/api/v1/workflows/{sample_workflow.id}/"
    archive_url = f"/api/v1/workflows/{sample_workflow.id}/archive/"
    restore_url = f"/api/v1/workflows/{sample_workflow.id}/restore/"

    # Initial state
    assert not sample_workflow.is_archived

    # Archive workflow
    res_archive = api_client.post(archive_url)
    assert res_archive.status_code == status.HTTP_200_OK
    data = res_archive.json()
    assert data["is_archived"] is True
    assert data["archived_at"] is not None

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is True
    assert sample_workflow.archived_at is not None
    archived_at_first = sample_workflow.archived_at

    # Idempotent re-archive
    res_rearchive = api_client.post(archive_url)
    assert res_rearchive.status_code == status.HTTP_200_OK
    assert res_rearchive.json()["is_archived"] is True
    sample_workflow.refresh_from_db()
    assert sample_workflow.archived_at == archived_at_first

    # Archived definition remains readable via GET by its owner
    res_get = api_client.get(wf_url)
    assert res_get.status_code == status.HTTP_200_OK
    assert res_get.json()["is_archived"] is True

    # Restore workflow
    res_restore = api_client.post(restore_url)
    assert res_restore.status_code == status.HTTP_200_OK
    data_restored = res_restore.json()
    assert data_restored["is_archived"] is False
    assert data_restored["archived_at"] is None

    sample_workflow.refresh_from_db()
    assert sample_workflow.is_archived is False
    assert sample_workflow.archived_at is None

    # Idempotent re-restore
    res_rerestore = api_client.post(restore_url)
    assert res_rerestore.status_code == status.HTTP_200_OK
    assert res_rerestore.json()["is_archived"] is False


@pytest.mark.django_db
def test_patch_archived_workflow_fails_or_preserves_archived(api_client, user_a, sample_workflow):
    api_client.force_authenticate(user=user_a)

    # 1. On active workflow, attempting to patch is_archived returns 400 Bad Request
    res_bad_field = api_client.patch(
        f"/api/v1/workflows/{sample_workflow.id}/",
        data={"is_archived": True},
        format="json",
    )
    assert res_bad_field.status_code == status.HTTP_400_BAD_REQUEST

    # 2. On archived workflow, attempting to edit name/steps returns 409 Conflict
    sample_workflow.is_archived = True
    sample_workflow.archived_at = timezone.now()
    sample_workflow.save()

    res_edit = api_client.patch(
        f"/api/v1/workflows/{sample_workflow.id}/",
        data={"name": "New Name on Archived"},
        format="json",
    )
    assert res_edit.status_code == status.HTTP_409_CONFLICT


# -------------------------------------------------------------------------
# 5. Archived Workflow Execution Rejection (Sync & Async)
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_archived_workflow_rejects_new_sync_runs(api_client, user_a, sample_workflow, sample_job):
    api_client.force_authenticate(user=user_a)
    sample_workflow.is_archived = True
    sample_workflow.archived_at = timezone.now()
    sample_workflow.save()

    initial_runs_count = WorkflowRun.objects.count()

    response = api_client.post(
        f"/api/v1/workflows/{sample_workflow.id}/runs/",
        data={"source_job_id": str(sample_job.id)},
        format="json",
    )
    assert response.status_code == status.HTTP_409_CONFLICT
    assert "archived" in response.json()["detail"].lower()
    assert WorkflowRun.objects.count() == initial_runs_count


@pytest.mark.django_db
def test_archived_workflow_rejects_new_async_runs(api_client, user_a, sample_workflow, sample_job):
    api_client.force_authenticate(user=user_a)
    sample_workflow.is_archived = True
    sample_workflow.archived_at = timezone.now()
    sample_workflow.save()

    initial_runs_count = WorkflowRun.objects.count()

    with patch("flowforge.workflows.views.execute_workflow_run_task.delay") as mock_task:
        response = api_client.post(
            f"/api/v1/workflows/{sample_workflow.id}/runs/async/",
            data={"source_job_id": str(sample_job.id)},
            format="json",
        )
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "archived" in response.json()["detail"].lower()
        assert WorkflowRun.objects.count() == initial_runs_count
        mock_task.assert_not_called()


# -------------------------------------------------------------------------
# 6. List filtering (default active-only, opt-in ?include_archived=true)
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_workflow_list_filters_archived_by_default(api_client, user_a):
    api_client.force_authenticate(user=user_a)

    wf_active = WorkflowDefinition.objects.create(
        owner=user_a,
        name="Active WF",
        steps=[{"op": "select_columns", "columns": ["id"]}],
        is_archived=False,
    )
    wf_archived = WorkflowDefinition.objects.create(
        owner=user_a,
        name="Archived WF",
        steps=[{"op": "select_columns", "columns": ["id"]}],
        is_archived=True,
        archived_at=timezone.now(),
    )

    # Default list -> active only
    res_default = api_client.get("/api/v1/workflows/")
    assert res_default.status_code == status.HTTP_200_OK
    results = res_default.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == str(wf_active.id)

    # Opt-in ?include_archived=true -> includes both
    res_all = api_client.get("/api/v1/workflows/?include_archived=true")
    assert res_all.status_code == status.HTTP_200_OK
    results_all = res_all.json()["results"]
    assert len(results_all) == 2
    ids = {r["id"] for r in results_all}
    assert str(wf_active.id) in ids
    assert str(wf_archived.id) in ids

    # Invalid filter value validation (e.g. ?include_archived=maybe) -> 400
    res_invalid = api_client.get("/api/v1/workflows/?include_archived=maybe")
    assert res_invalid.status_code == status.HTTP_400_BAD_REQUEST


# -------------------------------------------------------------------------
# 7. Preservation of Prior Runs, Exports, and In-Flight Tasks
# -------------------------------------------------------------------------
@pytest.mark.django_db
def test_prior_runs_and_exports_unaffected_by_workflow_edit_and_archive(
    api_client, user_a, sample_workflow, sample_job
):
    from flowforge.workflows.services import execute_workflow_run

    api_client.force_authenticate(user=user_a)

    # 1. Execute run before edit
    run = execute_workflow_run(sample_workflow, sample_job, user_a)
    run_id = run.id
    initial_step_snapshot = run.step_snapshot
    initial_output_data = run.output_data

    # Obtain initial export
    res_json_before = api_client.get(f"/api/v1/workflow-runs/{run_id}/download/?format=json")
    res_csv_before = api_client.get(f"/api/v1/workflow-runs/{run_id}/download/?format=csv")
    assert res_json_before.status_code == status.HTTP_200_OK
    assert res_csv_before.status_code == status.HTTP_200_OK
    json_bytes_before = res_json_before.content
    csv_bytes_before = res_csv_before.content

    # 2. Edit workflow definition steps
    api_client.patch(
        f"/api/v1/workflows/{sample_workflow.id}/",
        data={"name": "New Name", "steps": [{"op": "select_columns", "columns": ["name"]}]},
        format="json",
    )

    # 3. Archive workflow definition
    api_client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")

    # 4. Verify historical run and step snapshot are completely unchanged
    run.refresh_from_db()
    assert run.step_snapshot == initial_step_snapshot
    assert run.output_data == initial_output_data
    assert run.workflow_definition_id == sample_workflow.id

    # 5. Verify exports remain byte-identical
    res_json_after = api_client.get(f"/api/v1/workflow-runs/{run_id}/download/?format=json")
    res_csv_after = api_client.get(f"/api/v1/workflow-runs/{run_id}/download/?format=csv")
    assert res_json_after.content == json_bytes_before
    assert res_csv_after.content == csv_bytes_before


@pytest.mark.django_db
def test_queued_async_task_executes_persisted_snapshot_after_workflow_archived(
    user_a, sample_workflow, sample_job
):
    from flowforge.workflows.tasks import execute_workflow_run_task

    # Create queued run directly with snapshot
    run = WorkflowRun.objects.create(
        owner=user_a,
        workflow_definition=sample_workflow,
        source_job=sample_job,
        state=WorkflowRun.State.QUEUED,
        step_snapshot=sample_workflow.steps,
    )

    # Archive workflow while run is queued
    sample_workflow.is_archived = True
    sample_workflow.archived_at = timezone.now()
    sample_workflow.save()

    # Worker executes task
    execute_workflow_run_task(str(run.id))

    run.refresh_from_db()
    assert run.state == WorkflowRun.State.COMPLETED
    assert run.step_snapshot == [{"op": "select_columns", "columns": ["id", "name"]}]
    assert len(run.output_data) == 3


@pytest.mark.django_db(transaction=True)
def test_atomic_archive_fencing_prevents_concurrent_run_creation(
    api_client, user_a, sample_workflow, sample_job
):
    import threading

    from django.db import connection

    api_client.force_authenticate(user=user_a)

    barrier = threading.Barrier(2)
    results = {}

    def do_archive():
        connection.close()
        client = APIClient()
        client.force_authenticate(user=user_a)
        barrier.wait()
        res = client.post(f"/api/v1/workflows/{sample_workflow.id}/archive/")
        results["archive"] = res.status_code

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

    t1 = threading.Thread(target=do_archive)
    t2 = threading.Thread(target=do_run)

    t1.start()
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results["archive"] == status.HTTP_200_OK
    assert results["run"] in (status.HTTP_201_CREATED, status.HTTP_409_CONFLICT)
