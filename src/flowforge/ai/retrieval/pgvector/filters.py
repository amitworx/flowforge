"""Safe metadata filter compiler for parameterized SQL queries."""

from __future__ import annotations

import re
from typing import Any

from flowforge.ai.retrieval.pgvector.config import MAX_FILTER_IN_ITEMS
from flowforge.ai.retrieval.pgvector.exceptions import PgVectorFilterError

_SAFE_KEY_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\.]{0,63}$")
_SUPPORTED_OPERATORS = {"$eq", "$in", "$ne"}


def compile_metadata_filters(
    filters: dict[str, Any] | None,
) -> tuple[str, list[Any]]:
    """Compile a dictionary of metadata filters into parameterized SQL clauses.

    Safety requirements:
    - Metadata keys must match safe identifier pattern (alphanumeric, underscore, dot).
    - Keys and values are NEVER interpolated into SQL strings; values are strictly bound parameters.
    - Supported filter formats:
        - `{"key": "value"}` -> metadata->>%s = %s
        - `{"key": {"$eq": "value"}}` -> metadata->>%s = %s
        - `{"key": {"$ne": "value"}}` -> (metadata->>%s IS NULL OR metadata->>%s != %s)
        - `{"key": {"$in": ["val1", "val2"]}}` -> metadata->>%s = ANY(%s)
    - Bounds: $in list length is capped at MAX_FILTER_IN_ITEMS.

    Returns:
        tuple of (sql_clause_fragment, param_list)
        where sql_clause_fragment starts with ' AND ...' if filters exist, or '' if empty.

    Raises:
        PgVectorFilterError: If filter keys, operators, or bound lengths are invalid.
    """
    if not filters:
        return "", []

    if not isinstance(filters, dict):
        raise PgVectorFilterError("filter_metadata must be a dictionary")

    clauses: list[str] = []
    params: list[Any] = []

    for key, value in filters.items():
        if not isinstance(key, str) or not _SAFE_KEY_PATTERN.match(key):
            raise PgVectorFilterError(
                f"Invalid filter metadata key '{key}'. Must be alphanumeric identifier."
            )

        if isinstance(value, dict):
            # Operator dict
            for op, op_val in value.items():
                if op not in _SUPPORTED_OPERATORS:
                    raise PgVectorFilterError(
                        f"Unsupported filter operator '{op}'. Supported: {sorted(_SUPPORTED_OPERATORS)}"
                    )
                if op == "$eq":
                    clauses.append("metadata->>%s = %s")
                    params.extend([key, str(op_val) if op_val is not None else None])
                elif op == "$ne":
                    clauses.append("(metadata->>%s IS NULL OR metadata->>%s != %s)")
                    str_val = str(op_val) if op_val is not None else None
                    params.extend([key, key, str_val])
                elif op == "$in":
                    if not isinstance(op_val, (list, tuple, set)):
                        raise PgVectorFilterError(
                            f"Operator $in expects a list/tuple/set, got {type(op_val).__name__}"
                        )
                    if len(op_val) > MAX_FILTER_IN_ITEMS:
                        raise PgVectorFilterError(
                            f"$in operator exceeds maximum allowed items ({len(op_val)} > {MAX_FILTER_IN_ITEMS})"
                        )
                    str_list = [str(item) for item in op_val]
                    clauses.append("metadata->>%s = ANY(%s)")
                    params.extend([key, str_list])
        else:
            # Direct equality
            clauses.append("metadata->>%s = %s")
            params.extend([key, str(value) if value is not None else None])

    if not clauses:
        return "", []

    return " AND " + " AND ".join(clauses), params
