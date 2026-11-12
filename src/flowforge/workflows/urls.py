from django.urls import path

from flowforge.workflows.views import (
    WorkflowArchiveView,
    WorkflowCloneView,
    WorkflowDefinitionDetailView,
    WorkflowDefinitionListCreateView,
    WorkflowExecuteAsyncRunView,
    WorkflowExecuteRunView,
    WorkflowExportView,
    WorkflowImportView,
    WorkflowPreviewView,
    WorkflowRestoreView,
    WorkflowRunDetailView,
    WorkflowRunDownloadView,
    WorkflowRunListView,
    WorkflowRunSummaryView,
)

urlpatterns = [
    path("workflows/", WorkflowDefinitionListCreateView.as_view(), name="workflow-list-create"),
    path("workflows/import/", WorkflowImportView.as_view(), name="workflow-import"),
    path("workflows/<uuid:uuid>/", WorkflowDefinitionDetailView.as_view(), name="workflow-detail"),
    path(
        "workflows/<uuid:uuid>/export/",
        WorkflowExportView.as_view(),
        name="workflow-export",
    ),
    path(
        "workflows/<uuid:uuid>/clone/",
        WorkflowCloneView.as_view(),
        name="workflow-clone",
    ),
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
        "workflows/<uuid:uuid>/preview/",
        WorkflowPreviewView.as_view(),
        name="workflow-preview",
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
    path(
        "workflow-runs/summary/",
        WorkflowRunSummaryView.as_view(),
        name="workflow-run-summary",
    ),
    path("workflow-runs/<uuid:uuid>/", WorkflowRunDetailView.as_view(), name="workflow-run-detail"),
    path(
        "workflow-runs/<uuid:uuid>/download/",
        WorkflowRunDownloadView.as_view(),
        name="workflow-run-download",
    ),
]
