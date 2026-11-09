from django.urls import include, path

urlpatterns = [
    path("", include("flowforge.core.urls")),
    path("", include("flowforge.processing.urls")),
    path("", include("flowforge.workflows.urls")),
]
