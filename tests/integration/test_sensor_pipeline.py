import csv
import json
from pathlib import Path

import pandas as pd

from industrial_ai.data.pipeline import run_sensor_pipeline


def test_sensor_pipeline_writes_dataset_and_reports(tmp_path: Path) -> None:
    input_csv = tmp_path / "sample_sensors.csv"
    output_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"

    rows = [
        ["M-001", "2026-10-18T09:00:00+05:30", "65.0", "2.4", "1500"],
        ["M-001", "2026-10-18T09:01:00+05:30", "67.0", "2.6", "1510"],
        ["M-002", "2026-10-18T09:02:00+05:30", "70.0", "-1.0", "1600"],
    ]
    headers = [
        "machine_id",
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
    ]

    with input_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    result = run_sensor_pipeline(
        input_csv=input_csv,
        output_dir=output_dir,
        report_dir=report_dir,
        rolling_window=2,
    )

    output_csv = Path(result["output_csv"])
    pipeline_report_path = Path(result["report_json"])
    ingestion_report_path = Path(result["ingestion_report"])

    assert output_csv.is_file()
    assert pipeline_report_path.is_file()
    assert ingestion_report_path.is_file()

    processed = pd.read_csv(output_csv)
    assert len(processed) == 2
    assert "temperature_c_rolling_mean" in processed.columns
    assert processed["machine_id"].tolist() == ["M-001", "M-001"]

    pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
    assert pipeline_report["total_records"] == 3
    assert pipeline_report["accepted_records"] == 2
    assert pipeline_report["rejected_records"] == 1
    assert pipeline_report["feature_rows"] == 2

    ingestion_report = json.loads(ingestion_report_path.read_text(encoding="utf-8"))
    assert ingestion_report["accepted_count"] == 2
    assert ingestion_report["rejected_count"] == 1
