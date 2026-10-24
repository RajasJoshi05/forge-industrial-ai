import json
from pathlib import Path

import pandas as pd

from industrial_ai.monitoring.anomaly_detection import (
    SENSORS,
    fit_anomaly_detector,
    predict_sensor_anomalies,
    save_anomaly_detector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_features.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "isolation_forest_anomaly_detector.joblib"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_anomaly_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "anomaly_detection" / "training_report.json"

TRAINING_FRACTION = 0.80
CONTAMINATION = 0.02


def main() -> None:
    """Train an anomaly detector and report its chronological holdout results."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "machine_id",
        "timestamp",
        "failure_within_next_30min",
        *SENSORS,
    ]
    missing_columns = sorted(set(required_columns) - set(data.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if len(data) < 20:
        raise ValueError("At least 20 rows are required for a training and holdout split.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data = data.sort_values(
        ["timestamp", "machine_id"],
        kind="stable",
    ).reset_index(drop=True)

    cutoff = data["timestamp"].quantile(TRAINING_FRACTION)
    training_data = data.loc[data["timestamp"] <= cutoff, list(SENSORS)]
    holdout_mask = data["timestamp"] > cutoff

    if len(training_data) < 10:
        raise ValueError("The chronological training partition contains fewer than 10 rows.")

    if not holdout_mask.any():
        raise ValueError("The chronological holdout partition is empty.")

    model = fit_anomaly_detector(
        training_data,
        contamination=CONTAMINATION,
        random_state=42,
    )

    # Score every row using a model fitted only on the earlier time period.
    predictions = predict_sensor_anomalies(model, data[list(SENSORS)])
    predictions.insert(0, "timestamp", data["timestamp"])
    predictions.insert(0, "machine_id", data["machine_id"])
    predictions["failure_within_next_30min"] = data["failure_within_next_30min"].to_numpy()
    predictions["partition"] = "training"
    predictions.loc[holdout_mask.to_numpy(), "partition"] = "holdout"

    training_predictions = predictions.loc[~holdout_mask]
    holdout_predictions = predictions.loc[holdout_mask]

    anomaly_count = int(predictions["is_anomaly"].sum())
    holdout_anomaly_count = int(holdout_predictions["is_anomaly"].sum())

    report = {
        "model": "IsolationForest",
        "sensor_columns": list(SENSORS),
        "training_method": "Chronological split; model fitted on training partition only",
        "training_fraction_target": TRAINING_FRACTION,
        "contamination_setting": CONTAMINATION,
        "random_state": 42,
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "training_rows": int(len(training_predictions)),
        "holdout_rows": int(len(holdout_predictions)),
        "training_cutoff_utc": cutoff.isoformat(),
        "total_rows_scored": int(len(predictions)),
        "total_anomalies": anomaly_count,
        "total_anomaly_rate": float(predictions["is_anomaly"].mean()),
        "holdout_anomalies": holdout_anomaly_count,
        "holdout_anomaly_rate": float(holdout_predictions["is_anomaly"].mean()),
        "warning": (
            "Anomaly flags indicate unusual sensor patterns, not confirmed failures. "
            "The contamination setting is a modeling assumption, not measured accuracy. "
            "Failure labels were not used to fit the unsupervised model."
        ),
    }

    save_anomaly_detector(model, MODEL_PATH)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("FORGE // INDUSTRIAL AI")
    print("=" * 48)
    print("Anomaly detection training complete.")
    print(f"Training rows:       {len(training_predictions):,}")
    print(f"Holdout rows:        {len(holdout_predictions):,}")
    print(f"Training cutoff:     {cutoff}")
    print(f"All anomalies:       {anomaly_count:,}")
    print(f"Holdout anomalies:   {holdout_anomaly_count:,}")
    print(f"Model saved:         {MODEL_PATH}")
    print(f"Predictions saved:   {PREDICTIONS_PATH}")
    print(f"Report saved:        {REPORT_PATH}")
    print("Note: anomaly flags are not confirmed failure predictions.")


if __name__ == "__main__":
    main()
