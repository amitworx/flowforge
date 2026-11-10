from django.urls import path

from flowforge.workflows.views import (
    WorkflowArchiveView,
    WorkflowDefinitionDetailView,
    WorkflowDefinitionListCreateView,
    WorkflowExecuteAsyncRunView,
    WorkflowExecuteRunView,
    WorkflowRestoreView,
    WorkflowRunDetailView,
    WorkflowRunDownloadView,
    WorkflowRunListView,
)

urlpatterns = [
    path("workflows/", WorkflowDefinitionListCreateView.as_view(), name="workflow-list-create"),
    path("workflows/<uuid:uuid>/", WorkflowDefinitionDetailView.as_view(), name="workflow-detail"),
    path(
        "workflows/<uuid:uuid>/archive/",
        WorkflowArchiveView.as_view(),
        name="workflow-archive",
    ),
    path(
        "workflows/<uuid:uuid>/restore/",
        WorkflowRestoreView.as_view(),
        name="workflow-restore",
    ),
    path(
        "workflows/<uuid:uuid>/runs/", WorkflowExecuteRunView.as_view(), name="workflow-run-create"
    ),
    path(
        "workflows/<uuid:uuid>/runs/async/",
        WorkflowExecuteAsyncRunView.as_view(),
        name="workflow-async-run-create",
    ),
    path("workflow-runs/", WorkflowRunListView.as_view(), name="workflow-run-list"),
    path("workflow-runs/<uuid:uuid>/", WorkflowRunDetailView.as_view(), name="workflow-run-detail"),
    path(
        "workflow-runs/<uuid:uuid>/download/",
        WorkflowRunDownloadView.as_view(),
        name="workflow-run-download",
    ),
]
