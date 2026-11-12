from rest_framework import serializers

from flowforge.core.datetime_utils import parse_iso_aware_datetime
from flowforge.processing.models import ProcessingJob


class ProcessingJobFilterSerializer(serializers.Serializer):
    ALLOWED_PARAMS = {
        "file_type",
        "state",
        "created_from",
        "created_to",
        "ordering",
        "page",
        "page_size",
    }

    file_type = serializers.ChoiceField(choices=ProcessingJob.FileType.choices, required=False)
    state = serializers.ChoiceField(choices=ProcessingJob.State.choices, required=False)
    created_from = serializers.CharField(required=False)
    created_to = serializers.CharField(required=False)
    ordering = serializers.ChoiceField(
        choices=["-created_at", "created_at"], required=False, default="-created_at"
    )
    page = serializers.IntegerField(min_value=1, required=False)
    page_size = serializers.IntegerField(min_value=1, max_value=100, required=False)

    def validate(self, attrs):
        # Check unknown parameters from initial data
        unknown = set(self.initial_data.keys()) - self.ALLOWED_PARAMS
        if unknown:
            raise serializers.ValidationError(
                dict.fromkeys(
                    sorted(unknown),
                    f"Unknown query parameter. Allowed parameters are: {sorted(self.ALLOWED_PARAMS)}.",
                )
            )

        created_from_str = attrs.get("created_from")
        if created_from_str:
            attrs["created_from_dt"] = parse_iso_aware_datetime(created_from_str, "created_from")

        created_to_str = attrs.get("created_to")
        if created_to_str:
            attrs["created_to_dt"] = parse_iso_aware_datetime(created_to_str, "created_to")

        if (
            "created_from_dt" in attrs
            and "created_to_dt" in attrs
            and attrs["created_from_dt"] > attrs["created_to_dt"]
        ):
            raise serializers.ValidationError(
                {"created_from": "'created_from' cannot be later than 'created_to'."}
            )

        return attrs


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
