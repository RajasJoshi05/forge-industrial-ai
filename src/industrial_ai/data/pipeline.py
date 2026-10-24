from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from industrial_ai.data.features import build_sensor_features
from industrial_ai.data.sensor_csv import load_sensor_csv
from industrial_ai.maintenance.ingestion_report import save_ingestion_report

PIPELINE_VERSION = "0.2.0"


def calculate_sha256(file_path: str | Path) -> str:
    """Calculate a file's SHA-256 fingerprint without loading it all into memory."""
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def run_sensor_pipeline(
    input_csv: str | Path,
    output_dir: str | Path = "data/processed",
    report_dir: str | Path = "artifacts/reports/pipeline",
    rolling_window: int = 5,
) -> dict[str, object]:
    """Validate sensor CSV data, engineer features, and record data lineage."""
    source = Path(input_csv)

    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")

    if not source.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {source}")

    run_id = str(uuid4())
    input_sha256 = calculate_sha256(source)

    result = load_sensor_csv(source)
    reports = Path(report_dir)

    ingestion_report_path = save_ingestion_report(
        result,
        output_dir=reports / "ingestion",
    )

    features = build_sensor_features(
        result["accepted"],
        rolling_window=rolling_window,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    output_csv = destination / f"{source.stem}_features.csv"
    features.to_csv(output_csv, index=False)
    output_sha256 = calculate_sha256(output_csv)

    pipeline_report = {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "input_file": str(source.resolve()),
        "input_sha256": input_sha256,
        "output_file": str(output_csv.resolve()),
        "output_sha256": output_sha256,
        "total_records": len(result["accepted"]) + len(result["rejected"]),
        "accepted_records": len(result["accepted"]),
        "rejected_records": len(result["rejected"]),
        "feature_rows": len(features),
        "feature_columns": list(features.columns),
        "rolling_window": rolling_window,
        "ingestion_report": str(ingestion_report_path.resolve()),
    }

    report_path = reports / f"{source.stem}_{run_id}_pipeline_report.json"
    report_path.write_text(
        json.dumps(pipeline_report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    return {
        "output_csv": output_csv,
        "report_json": report_path,
        "ingestion_report": ingestion_report_path,
        "summary": pipeline_report,
    }
