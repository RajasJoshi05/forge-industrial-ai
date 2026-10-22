from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from industrial_ai.maintenance.batch_validation import validate_sensor_batch
from industrial_ai.maintenance.contracts import SensorRecord


def ingest_sensor_batch(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Validate a sensor batch and return accepted and rejected records."""
    batch = [dict(record) for record in records]
    batch_errors = validate_sensor_batch(batch)

    errors_by_row: dict[int, list[str]] = {}
    for error in batch_errors:
        if not error.startswith("Row "):
            continue

        row_prefix, message = error.split(": ", maxsplit=1)
        row_index = int(row_prefix.removeprefix("Row "))
        errors_by_row.setdefault(row_index, []).append(message)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, record in enumerate(batch):
        if index in errors_by_row:
            rejected.append(
                {
                    "row_index": index,
                    "record": record,
                    "errors": errors_by_row[index],
                }
            )
            continue

        try:
            sensor = SensorRecord.model_validate(record)
        except ValidationError as exc:
            rejected.append(
                {
                    "row_index": index,
                    "record": record,
                    "errors": [
                        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                        for error in exc.errors()
                    ],
                }
            )
            continue

        accepted.append(sensor.model_dump(mode="json"))

    return {"accepted": accepted, "rejected": rejected}
