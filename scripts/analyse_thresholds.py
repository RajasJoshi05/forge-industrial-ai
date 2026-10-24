"""Evaluate predictive-maintenance warning thresholds on a chronological holdout."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_features.csv"
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "predictive_maintenance_logistic_regression.joblib"
)
METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "predictive_maintenance_metadata.json"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
TARGET = "failure_within_next_30min"

EXCLUDED_COLUMNS = {"machine_id", "timestamp", TARGET}


def main() -> None:
    for path in (DATA_PATH, MODEL_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required project file not found: {path}")

    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "machine_id"]).reset_index(drop=True)

    if TARGET not in data.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET}")

    feature_columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]

    if not feature_columns:
        raise ValueError("No model feature columns were found.")

    # Split by unique timestamps so rows from the same timestamp
    # cannot appear on both sides of the holdout boundary.
    timestamps = sorted(data["timestamp"].dropna().unique())
    if len(timestamps) < 5:
        raise ValueError("Not enough unique timestamps for chronological evaluation.")

    split_index = int(len(timestamps) * 0.8)
    split_index = min(max(split_index, 1), len(timestamps) - 1)
    cutoff = timestamps[split_index]

    holdout = data.loc[data["timestamp"] >= cutoff].copy()
    y_true = holdout[TARGET].astype(int)

    if y_true.nunique() < 2:
        raise ValueError(
            "The holdout contains only one target class; threshold metrics would be misleading."
        )

    model = joblib.load(MODEL_PATH)

    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        missing = sorted(set(expected_features) - set(feature_columns))
        extra = sorted(set(feature_columns) - set(expected_features))
        if missing or extra:
            raise ValueError(f"Feature mismatch. Missing: {missing}; unexpected: {extra}")
        feature_columns = list(expected_features)

    probabilities = model.predict_proba(holdout[feature_columns])[:, 1]

    report_rows: list[dict[str, float | int]] = []

    for threshold in [round(value / 100, 2) for value in range(10, 100, 5)]:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

        report_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1_score": f1_score(y_true, predictions, zero_division=0),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "predicted_warnings": int(predictions.sum()),
            }
        )

    threshold_report = pd.DataFrame(report_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "threshold_analysis.csv"
    json_path = REPORT_DIR / "threshold_analysis_summary.json"

    threshold_report.to_csv(csv_path, index=False)

    summary = {
        "dataset_type": json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
            "dataset_type", "unknown"
        ),
        "evaluation": "chronological holdout",
        "holdout_start": str(holdout["timestamp"].min()),
        "holdout_end": str(holdout["timestamp"].max()),
        "holdout_rows": int(len(holdout)),
        "positive_labels": int(y_true.sum()),
        "negative_labels": int((y_true == 0).sum()),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold_0_50": threshold_report.loc[threshold_report["threshold"] == 0.5]
        .iloc[0]
        .to_dict(),
        "warning": (
            "Results use synthetic data and do not establish real-world "
            "industrial predictive performance."
        ),
    }

    json_path.write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )

    print("\nFORGE // INDUSTRIAL AI")
    print("=" * 58)
    print("Chronological holdout threshold analysis")
    print(f"Holdout rows:       {len(holdout):,}")
    print(f"Holdout start:      {holdout['timestamp'].min()}")
    print(f"Holdout end:        {holdout['timestamp'].max()}")
    print(f"Positive labels:    {int(y_true.sum()):,}")
    print(f"Average precision:  {summary['average_precision']:.4f}")
    print(f"ROC-AUC:            {summary['roc_auc']:.4f}")

    print("\nThreshold comparison:")
    print(
        threshold_report[
            [
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "false_positives",
                "false_negatives",
                "predicted_warnings",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nReports saved:")
    print(f"  {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  {json_path.relative_to(PROJECT_ROOT)}")
    print("\nReminder: these results are based on synthetic data.")


if __name__ == "__main__":
    main()
