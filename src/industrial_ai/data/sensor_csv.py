import csv
from pathlib import Path
from typing import Any

from industrial_ai.maintenance.batch_validation import validate_sensor_batch

REQUIRED_COLUMNS = {
    "machine_id",
    "timestamp",
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
}


def load_sensor_csv(
    csv_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load a sensor CSV and separate valid records from rejected records."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Sensor CSV must contain a header row")

        columns = [name.strip() for name in reader.fieldnames if name]
        missing_columns = sorted(REQUIRED_COLUMNS - set(columns))

        if missing_columns:
            raise ValueError(f"Sensor CSV is missing required columns: {missing_columns}")

        records = [
            {
                key.strip(): value.strip() if value is not None else ""
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]

    errors_by_row: dict[int, list[str]] = {}
    for error in validate_sensor_batch(records):
        row_prefix, message = error.split(": ", maxsplit=1)
        row_index = int(row_prefix.removeprefix("Row "))
        errors_by_row.setdefault(row_index, []).append(message)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if index in errors_by_row:
            rejected.append(
                {
                    "row_index": index,
                    "record": record,
                    "errors": errors_by_row[index],
                }
            )
        else:
            accepted.append(record)

    return {"accepted": accepted, "rejected": rejected}
