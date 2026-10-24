import json
from pathlib import Path

from industrial_ai.maintenance.ingestion import ingest_sensor_batch
from industrial_ai.maintenance.ingestion_report import save_ingestion_report


def test_sensor_ingestion_and_report_pipeline(tmp_path: Path) -> None:
    records = [
        {
            "machine_id": "M-001",
            "timestamp": "2026-10-18T09:00:00+05:30",
            "temperature_c": 65.0,
            "vibration_mm_s": 2.4,
            "rotational_speed_rpm": 1500,
        },
        {
            "machine_id": "M-002",
            "timestamp": "2026-10-18T09:05:00+05:30",
            "temperature_c": 70.0,
            "vibration_mm_s": 3.1,
            "rotational_speed_rpm": 1600,
        },
        {
            "machine_id": "M-002",
            "timestamp": "2026-10-18T09:05:00+05:30",
            "temperature_c": 70.0,
            "vibration_mm_s": 3.1,
            "rotational_speed_rpm": 1600,
        },
        {
            "machine_id": "M-003",
            "timestamp": "2026-10-18T09:10:00+05:30",
            "temperature_c": 75.0,
            "vibration_mm_s": -1.0,
            "rotational_speed_rpm": 1700,
        },
    ]

    result = ingest_sensor_batch(records)

    assert len(result["accepted"]) == 1
    assert len(result["rejected"]) == 3

    report_path = save_ingestion_report(result, output_dir=tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["total_records"] == 4
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 3
    assert len(report["rejection_details"]) == 3
    assert report_path.exists()
