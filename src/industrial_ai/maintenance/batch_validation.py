from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from industrial_ai.maintenance.contracts import SensorRecord
from industrial_ai.maintenance.validation import validate_sensor_record


def validate_sensor_batch(
    records: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Validate sensor records and flag every copy of a duplicate observation."""
    errors: list[str] = []
    seen_rows: dict[tuple[str, str], list[int]] = defaultdict(list)

    for index, record in enumerate(records):
        record_errors = validate_sensor_record(dict(record))

        for error in record_errors:
            errors.append(f"Row {index}: {error}")

        if record_errors:
            continue

        sensor = SensorRecord.model_validate(dict(record))
        key = (sensor.machine_id, sensor.timestamp.isoformat())
        seen_rows[key].append(index)

    for (machine_id, timestamp), row_indices in seen_rows.items():
        if len(row_indices) > 1:
            for index in row_indices:
                errors.append(
                    f"Row {index}: duplicate machine/timestamp pair "
                    f"({machine_id}, {timestamp}); duplicate rows: {row_indices}"
                )

    return errors
