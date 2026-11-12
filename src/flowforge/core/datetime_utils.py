"""Shared datetime and query parameter parsing utilities for FlowForge APIs."""

from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework import serializers


def parse_iso_aware_datetime(val: str, field_name: str):
    """
    Parses a string as an ISO-8601 timezone-aware datetime.
    Restores unencoded '+' timezone offsets converted into spaces during standard URL query decoding.
    Raises serializers.ValidationError if malformed or naive.
    """
    val_clean = val.strip()
    if " " in val_clean and ("+" not in val_clean and not val_clean.endswith("Z")):
        parts = val_clean.split(" ")
        if len(parts) == 2 and (":" in parts[1] or len(parts[1]) == 4):
            val_clean = f"{parts[0]}+{parts[1]}"

    dt = parse_datetime(val_clean)
    if dt is None:
        raise serializers.ValidationError({field_name: "Invalid ISO 8601 datetime format."})
    if not timezone.is_aware(dt):
        raise serializers.ValidationError(
            {field_name: "Datetime must be timezone-aware (e.g. UTC 'Z' or offset)."}
        )
    return dt
