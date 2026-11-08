from django.urls import path

from flowforge.processing.views import (
    ProcessingJobDetailView,
    ProcessingJobDownloadView,
    ProcessingJobListCreateView,
)

urlpatterns = [
    path("jobs/", ProcessingJobListCreateView.as_view(), name="job_list_create"),
    path("jobs/<uuid:job_id>/", ProcessingJobDetailView.as_view(), name="job_detail"),
    path(
        "jobs/<uuid:job_id>/download/",
        ProcessingJobDownloadView.as_view(),
        name="job_download",
    ),
]
