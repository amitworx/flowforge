from django.urls import path

from flowforge.workflows.views import (
    WorkflowDefinitionDetailView,
    WorkflowDefinitionListCreateView,
    WorkflowExecuteAsyncRunView,
    WorkflowExecuteRunView,
    WorkflowRunDetailView,
    WorkflowRunListView,
)

urlpatterns = [
    path("workflows/", WorkflowDefinitionListCreateView.as_view(), name="workflow-list-create"),
    path("workflows/<uuid:uuid>/", WorkflowDefinitionDetailView.as_view(), name="workflow-detail"),
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
]
