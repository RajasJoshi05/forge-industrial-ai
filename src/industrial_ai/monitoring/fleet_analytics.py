from __future__ import annotations

from typing import Any

import pandas as pd

SENSOR_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
]


def build_fleet_snapshot(
    data: pd.DataFrame,
    predictor: Any,
) -> pd.DataFrame:
    """Score each machine's latest available engineered observation.

    This uses historical dataset rows, not live sensor telemetry.
    """
    required = {
        "machine_id",
        "timestamp",
        *SENSOR_COLUMNS,
        "failure_within_next_30min",
        *predictor.features,
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Fleet dataset is missing columns: {sorted(missing)}")

    if data.empty:
        return pd.DataFrame(
            columns=[
                "machine_id",
                "timestamp",
                *SENSOR_COLUMNS,
                "warning_probability",
                "prediction",
                "actual_synthetic_label",
            ]
        )

    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")

    latest_rows = (
        frame.sort_values("timestamp")
        .groupby("machine_id", sort=True, as_index=False)
        .tail(1)
        .sort_values("machine_id")
    )

    results: list[dict[str, object]] = []

    for _, row in latest_rows.iterrows():
        observation = {feature: row[feature] for feature in predictor.features}

        result = predictor.predict(observation)

        results.append(
            {
                "machine_id": str(row["machine_id"]),
                "timestamp": row["timestamp"],
                "temperature_c": float(row["temperature_c"]),
                "vibration_mm_s": float(row["vibration_mm_s"]),
                "rotational_speed_rpm": float(row["rotational_speed_rpm"]),
                "warning_probability": result["warning_probability"],
                "prediction": result["prediction"],
                "actual_synthetic_label": int(row["failure_within_next_30min"]),
            }
        )

    return pd.DataFrame(results)


def summarize_fleet(snapshot: pd.DataFrame) -> dict[str, int]:
    """Calculate simple fleet-level counts from scored machine rows."""
    if snapshot.empty:
        return {
            "machines": 0,
            "warnings": 0,
            "normal": 0,
            "synthetic_positive_labels": 0,
        }

    return {
        "machines": int(snapshot["machine_id"].nunique()),
        "warnings": int((snapshot["prediction"] == "WARNING").sum()),
        "normal": int((snapshot["prediction"] == "NORMAL").sum()),
        "synthetic_positive_labels": int((snapshot["actual_synthetic_label"] == 1).sum()),
    }
