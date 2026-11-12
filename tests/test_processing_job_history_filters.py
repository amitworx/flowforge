"""
Unit and integration tests for ProcessingJob history filtering, deterministic sorting,
and strict pagination (Milestone M0-012).
"""

from datetime import UTC, datetime, timedelta

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from flowforge.processing.models import ProcessingJob


@pytest.fixture
def auth_client(db: None, django_user_model):  # noqa: ARG001
    user = django_user_model.objects.create_user(
        username="job_history_user", email="job_history@example.com", password="password123"
    )
    client = APIClient()
    client.force_authenticate(user=user)
    client.user = user
    return client


@pytest.fixture
def other_user(db: None, django_user_model):  # noqa: ARG001
    return django_user_model.objects.create_user(
        username="job_history_other", email="job_other@example.com", password="password123"
    )


@pytest.fixture
def sample_jobs(auth_client, other_user):
    now = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)

    # User jobs with explicit timestamps
    # Job 1: csv, completed, now - 3 hours
    job1 = ProcessingJob.objects.create(
        owner=auth_client.user,
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.COMPLETED,
        original_filename="records1.csv",
        input_row_count=10,
        output_row_count=10,
        column_names=["id", "name"],
        output_data=[{"id": 1, "name": "A"}],
    )
    ProcessingJob.objects.filter(id=job1.id).update(created_at=now - timedelta(hours=3))

    # Job 2: json, pending, now - 2 hours
    job2 = ProcessingJob.objects.create(
        owner=auth_client.user,
        file_type=ProcessingJob.FileType.JSON,
        state=ProcessingJob.State.PENDING,
        original_filename="records2.json",
        input_row_count=5,
        output_row_count=5,
        column_names=["id", "val"],
        output_data=[{"id": 2, "val": "B"}],
    )
    ProcessingJob.objects.filter(id=job2.id).update(created_at=now - timedelta(hours=2))

    # Job 3: csv, failed, now - 1 hour
    job3 = ProcessingJob.objects.create(
        owner=auth_client.user,
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.FAILED,
        original_filename="records3.csv",
        input_row_count=0,
        output_row_count=0,
        column_names=[],
        output_data=[],
    )
    ProcessingJob.objects.filter(id=job3.id).update(created_at=now - timedelta(hours=1))

    # Foreign job: csv, completed, now - 2 hours
    foreign_job = ProcessingJob.objects.create(
        owner=other_user,
        file_type=ProcessingJob.FileType.CSV,
        state=ProcessingJob.State.COMPLETED,
        original_filename="foreign.csv",
        input_row_count=20,
        output_row_count=20,
        column_names=["x"],
        output_data=[{"x": 1}],
    )
    ProcessingJob.objects.filter(id=foreign_job.id).update(created_at=now - timedelta(hours=2))

    return {
        "job1": job1,
        "job2": job2,
        "job3": job3,
        "foreign_job": foreign_job,
        "now": now,
    }


@pytest.mark.django_db
class TestProcessingJobHistoryFilters:
    def test_anonymous_rejected(self):
        client = APIClient()
        response = client.get("/api/v1/jobs/")
        assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_default_list_shows_only_owned_jobs_newest_first(self, auth_client, sample_jobs):
        response = auth_client.get("/api/v1/jobs/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 3
        # Should be newest first: job3 (now - 1h), job2 (now - 2h), job1 (now - 3h)
        returned_ids = [r["id"] for r in data["results"]]
        assert returned_ids == [
            str(sample_jobs["job3"].id),
            str(sample_jobs["job2"].id),
            str(sample_jobs["job1"].id),
        ]
        # Ensure foreign job is not present
        assert str(sample_jobs["foreign_job"].id) not in returned_ids

    def test_filter_by_file_type(self, auth_client, sample_jobs):
        # CSV filter
        res_csv = auth_client.get("/api/v1/jobs/?file_type=csv")
        assert res_csv.status_code == status.HTTP_200_OK
        data_csv = res_csv.json()
        assert data_csv["count"] == 2
        assert {r["id"] for r in data_csv["results"]} == {
            str(sample_jobs["job1"].id),
            str(sample_jobs["job3"].id),
        }

        # JSON filter
        res_json = auth_client.get("/api/v1/jobs/?file_type=json")
        assert res_json.status_code == status.HTTP_200_OK
        data_json = res_json.json()
        assert data_json["count"] == 1
        assert data_json["results"][0]["id"] == str(sample_jobs["job2"].id)

    def test_filter_by_state(self, auth_client, sample_jobs):
        res_completed = auth_client.get("/api/v1/jobs/?state=completed")
        assert res_completed.status_code == status.HTTP_200_OK
        data = res_completed.json()
        assert data["count"] == 1
        assert data["results"][0]["id"] == str(sample_jobs["job1"].id)

        res_pending = auth_client.get("/api/v1/jobs/?state=pending")
        assert res_pending.status_code == status.HTTP_200_OK
        assert res_pending.json()["count"] == 1
        assert res_pending.json()["results"][0]["id"] == str(sample_jobs["job2"].id)

        res_failed = auth_client.get("/api/v1/jobs/?state=failed")
        assert res_failed.status_code == status.HTTP_200_OK
        assert res_failed.json()["count"] == 1
        assert res_failed.json()["results"][0]["id"] == str(sample_jobs["job3"].id)

        res_processing = auth_client.get("/api/v1/jobs/?state=processing")
        assert res_processing.status_code == status.HTTP_200_OK
        assert res_processing.json()["count"] == 0

    def test_filter_by_created_from_and_created_to_inclusive(self, auth_client, sample_jobs):
        now = sample_jobs["now"]
        # Exact created_from at job2 time (now - 2h) to now
        from_str = (now - timedelta(hours=2)).isoformat()
        res = auth_client.get(f"/api/v1/jobs/?created_from={from_str}")
        assert res.status_code == status.HTTP_200_OK
        data = res.json()
        assert data["count"] == 2
        assert {r["id"] for r in data["results"]} == {
            str(sample_jobs["job2"].id),
            str(sample_jobs["job3"].id),
        }

        # Exact window containing only job2
        to_str = (now - timedelta(hours=1, minutes=30)).isoformat()
        res_window = auth_client.get(f"/api/v1/jobs/?created_from={from_str}&created_to={to_str}")
        assert res_window.status_code == status.HTTP_200_OK
        data_window = res_window.json()
        assert data_window["count"] == 1
        assert data_window["results"][0]["id"] == str(sample_jobs["job2"].id)

    def test_filter_by_datetime_with_timezone_offset(self, auth_client, sample_jobs):
        now = sample_jobs["now"]
        # UTC is 12:00. In +05:30 offset, 10:00 UTC (job2) is 15:30.
        # Construct ISO string with +05:30: 2026-11-12T15:30:00+05:30
        job2_dt_offset = (
            (now - timedelta(hours=2)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        )
        res = auth_client.get(
            f"/api/v1/jobs/?created_from={job2_dt_offset}&created_to={job2_dt_offset}"
        )
        assert res.status_code == status.HTTP_200_OK
        data = res.json()
        assert data["count"] == 1
        assert data["results"][0]["id"] == str(sample_jobs["job2"].id)

    def test_compound_and_composition(self, auth_client, sample_jobs):
        # CSV and completed -> job1 only
        res = auth_client.get("/api/v1/jobs/?file_type=csv&state=completed")
        assert res.status_code == status.HTTP_200_OK
        data = res.json()
        assert data["count"] == 1
        assert data["results"][0]["id"] == str(sample_jobs["job1"].id)

        # CSV and pending -> 0
        res2 = auth_client.get("/api/v1/jobs/?file_type=csv&state=pending")
        assert res2.status_code == status.HTTP_200_OK
        assert res2.json()["count"] == 0

    def test_ordering_created_at_ascending_and_descending(self, auth_client, sample_jobs):
        res_asc = auth_client.get("/api/v1/jobs/?ordering=created_at")
        assert res_asc.status_code == status.HTTP_200_OK
        ids_asc = [r["id"] for r in res_asc.json()["results"]]
        assert ids_asc == [
            str(sample_jobs["job1"].id),
            str(sample_jobs["job2"].id),
            str(sample_jobs["job3"].id),
        ]

        res_desc = auth_client.get("/api/v1/jobs/?ordering=-created_at")
        assert res_desc.status_code == status.HTTP_200_OK
        ids_desc = [r["id"] for r in res_desc.json()["results"]]
        assert ids_desc == [
            str(sample_jobs["job3"].id),
            str(sample_jobs["job2"].id),
            str(sample_jobs["job1"].id),
        ]

    def test_stable_tie_breaker_for_identical_timestamps(self, auth_client):
        # Create 5 jobs with the exact same timestamp
        fixed_time = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)
        job_ids = []
        for i in range(5):
            j = ProcessingJob.objects.create(
                owner=auth_client.user,
                file_type=ProcessingJob.FileType.CSV,
                state=ProcessingJob.State.COMPLETED,
                original_filename=f"tie_{i}.csv",
            )
            job_ids.append(j.id)
        ProcessingJob.objects.filter(id__in=job_ids).update(created_at=fixed_time)

        # Sort job_ids descending by uuid for -id tie-breaker
        sorted_uuids_desc = [str(uid) for uid in sorted(job_ids, reverse=True)]
        sorted_uuids_asc = [str(uid) for uid in sorted(job_ids)]

        # Fetch with page_size=2 across 3 pages
        res_p1 = auth_client.get("/api/v1/jobs/?ordering=-created_at&page=1&page_size=2")
        res_p2 = auth_client.get("/api/v1/jobs/?ordering=-created_at&page=2&page_size=2")
        res_p3 = auth_client.get("/api/v1/jobs/?ordering=-created_at&page=3&page_size=2")

        assert res_p1.status_code == status.HTTP_200_OK
        assert res_p2.status_code == status.HTTP_200_OK
        assert res_p3.status_code == status.HTTP_200_OK

        p1_ids = [r["id"] for r in res_p1.json()["results"]]
        p2_ids = [r["id"] for r in res_p2.json()["results"]]
        p3_ids = [r["id"] for r in res_p3.json()["results"]]

        # Assert no overlap
        assert set(p1_ids).isdisjoint(set(p2_ids))
        assert set(p2_ids).isdisjoint(set(p3_ids))
        assert (p1_ids + p2_ids + p3_ids) == sorted_uuids_desc

        # Ascending tie breaker
        res_asc_p1 = auth_client.get("/api/v1/jobs/?ordering=created_at&page=1&page_size=2")
        res_asc_p2 = auth_client.get("/api/v1/jobs/?ordering=created_at&page=2&page_size=2")
        res_asc_p3 = auth_client.get("/api/v1/jobs/?ordering=created_at&page=3&page_size=2")
        asc_combined = (
            [r["id"] for r in res_asc_p1.json()["results"]]
            + [r["id"] for r in res_asc_p2.json()["results"]]
            + [r["id"] for r in res_asc_p3.json()["results"]]
        )
        assert asc_combined == sorted_uuids_asc

    def test_validation_unknown_query_parameter_rejected(self, auth_client):
        res = auth_client.get("/api/v1/jobs/?foo=bar")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "foo" in res.json()

    def test_validation_duplicate_query_parameter_rejected(self, auth_client):
        res = auth_client.get("/api/v1/jobs/?state=completed&state=pending")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "state" in res.json() or "detail" in res.json()

    def test_validation_invalid_file_type_rejected(self, auth_client):
        res = auth_client.get("/api/v1/jobs/?file_type=parquet")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "file_type" in res.json()

    def test_validation_invalid_state_rejected(self, auth_client):
        res = auth_client.get("/api/v1/jobs/?state=unknown_state")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "state" in res.json()

    def test_validation_invalid_datetime_rejected(self, auth_client):
        res = auth_client.get("/api/v1/jobs/?created_from=not-a-date")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "created_from" in res.json()

    def test_validation_naive_datetime_rejected(self, auth_client):
        res = auth_client.get("/api/v1/jobs/?created_from=2026-11-12T10:00:00")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "created_from" in res.json()

    def test_validation_inverted_time_range_rejected(self, auth_client):
        from_str = "2026-11-12T12:00:00Z"
        to_str = "2026-11-12T10:00:00Z"
        res = auth_client.get(f"/api/v1/jobs/?created_from={from_str}&created_to={to_str}")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "created_from" in res.json()

    def test_validation_unsupported_ordering_rejected(self, auth_client):
        res = auth_client.get("/api/v1/jobs/?ordering=id")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "ordering" in res.json()

    def test_pagination_parameters_validation(self, auth_client):
        # Negative / zero / non-integer page_size
        assert (
            auth_client.get("/api/v1/jobs/?page_size=0").status_code == status.HTTP_400_BAD_REQUEST
        )
        assert (
            auth_client.get("/api/v1/jobs/?page_size=-5").status_code == status.HTTP_400_BAD_REQUEST
        )
        assert (
            auth_client.get("/api/v1/jobs/?page_size=abc").status_code
            == status.HTTP_400_BAD_REQUEST
        )
        assert (
            auth_client.get("/api/v1/jobs/?page_size=101").status_code
            == status.HTTP_400_BAD_REQUEST
        )

        # Valid boundaries: min=1, max=100
        assert auth_client.get("/api/v1/jobs/?page_size=1").status_code == status.HTTP_200_OK
        assert auth_client.get("/api/v1/jobs/?page_size=100").status_code == status.HTTP_200_OK

        # Negative / zero / non-integer page
        assert auth_client.get("/api/v1/jobs/?page=0").status_code == status.HTTP_400_BAD_REQUEST
        assert auth_client.get("/api/v1/jobs/?page=-1").status_code == status.HTTP_400_BAD_REQUEST
        assert auth_client.get("/api/v1/jobs/?page=xyz").status_code == status.HTTP_400_BAD_REQUEST

    def test_query_cost_and_no_output_data_serialization(
        self,
        auth_client,
        sample_jobs,  # noqa: ARG002
    ):
        res = auth_client.get("/api/v1/jobs/")
        assert res.status_code == status.HTTP_200_OK
        for row in res.json()["results"]:
            assert "output_data" not in row

    def test_zero_side_effects_on_get(self, auth_client, sample_jobs):  # noqa: ARG002
        init_count = ProcessingJob.objects.count()
        res = auth_client.get("/api/v1/jobs/?file_type=csv")
        assert res.status_code == status.HTTP_200_OK
        assert ProcessingJob.objects.count() == init_count
