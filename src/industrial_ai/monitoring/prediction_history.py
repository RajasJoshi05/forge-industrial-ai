from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "artifacts" / "reports" / "prediction_history.csv"

FIELDNAMES = [
    "event_id",
    "recorded_at_utc",
    "machine_id",
    "observation_timestamp",
    "model_name",
    "target",
    "warning_probability",
    "threshold",
    "prediction",
    "dataset_type",
]


def record_prediction(
    machine_id: str,
    observation_timestamp: str,
    prediction: dict[str, Any],
    history_path: str | Path = DEFAULT_HISTORY_PATH,
) -> dict[str, str]:
    """Append one prediction event to a persistent CSV history."""
    if not machine_id.strip():
        raise ValueError("machine_id must not be empty")

    probability = float(prediction["warning_probability"])
    threshold = float(prediction["threshold"])

    if not 0.0 <= probability <= 1.0:
        raise ValueError("warning_probability must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if prediction["prediction"] not in {"NORMAL", "WARNING"}:
        raise ValueError("prediction must be NORMAL or WARNING")

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    event = {
        "event_id": str(uuid4()),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "machine_id": machine_id,
        "observation_timestamp": str(observation_timestamp),
        "model_name": str(prediction["model_name"]),
        "target": str(prediction["target"]),
        "warning_probability": str(probability),
        "threshold": str(threshold),
        "prediction": str(prediction["prediction"]),
        "dataset_type": str(prediction.get("dataset_type", "unknown")),
    }

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(event)

    return event
