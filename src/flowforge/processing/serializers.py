from rest_framework import serializers

from flowforge.processing.models import ProcessingJob


class ProcessingJobListSerializer(serializers.ModelSerializer):
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = ProcessingJob
        fields = [
            "id",
            "file_type",
            "state",
            "original_filename",
            "input_row_count",
            "output_row_count",
            "validation_errors",
            "column_names",
            "download_url",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_download_url(self, obj: ProcessingJob) -> str:
        request = self.context.get("request")
        path = f"/api/v1/jobs/{obj.id}/download/"
        if request is not None:
            return request.build_absolute_uri(path)
        return path


class ProcessingJobDetailSerializer(serializers.ModelSerializer):
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = ProcessingJob
        fields = [
            "id",
            "file_type",
            "state",
            "original_filename",
            "input_row_count",
            "output_row_count",
            "validation_errors",
            "column_names",
            "output_data",
            "download_url",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_download_url(self, obj: ProcessingJob) -> str:
        request = self.context.get("request")
        path = f"/api/v1/jobs/{obj.id}/download/"
        if request is not None:
            return request.build_absolute_uri(path)
        return path
