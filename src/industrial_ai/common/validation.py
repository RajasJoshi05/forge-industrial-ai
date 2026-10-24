from collections.abc import Mapping
from typing import Any


def validate_required_fields(
    record: Mapping[str, Any],
    required_fields: set[str],
) -> list[str]:
    """Return required field names that are missing or empty."""
    errors: list[str] = []

    for field in sorted(required_fields):
        if field not in record:
            errors.append(f"Missing required field: {field}")
        elif record[field] is None:
            errors.append(f"Required field is None: {field}")
        elif isinstance(record[field], str) and not record[field].strip():
            errors.append(f"Required field is empty: {field}")

    return errors
