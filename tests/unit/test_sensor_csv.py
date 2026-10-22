import csv

import pytest

from industrial_ai.data.sensor_csv import load_sensor_csv


def test_sensor_csv_accepts_valid_rows_and_rejects_bad_rows(
    tmp_path,
) -> None:
    csv_path = tmp_path / "sensors.csv"
    fieldnames = [
        "machine_id",
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
    ]
    rows = [
        ["M-001", "2026-10-18T09:00:00+05:30", "65.0", "2.4", "1500"],
        ["M-002", "2026-10-18T09:05:00+05:30", "70.0", "3.1", "1600"],
        ["M-002", "2026-10-18T09:05:00+05:30", "70.0", "3.1", "1600"],
        ["M-003", "2026-10-18T09:10:00+05:30", "75.0", "-1.0", "1700"],
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    result = load_sensor_csv(csv_path)

    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["machine_id"] == "M-001"
    assert len(result["rejected"]) == 3


def test_sensor_csv_rejects_missing_required_columns(tmp_path) -> None:
    csv_path = tmp_path / "incomplete.csv"
    csv_path.write_text(
        "machine_id,timestamp\\nM-001,2026-10-18T09:00:00+05:30\\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_sensor_csv(csv_path)


def test_sensor_csv_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sensor_csv(tmp_path / "does_not_exist.csv")
