"""Evaluate predictive maintenance with chronological train/validation/test splits.

The model is fitted on training data only. The warning threshold is selected
using validation data, then evaluated once on the untouched final test set.
A six-timestamp purge protects against the target's 30-minute future window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "industrial_features.csv"
REPORT_DIR = ROOT / "artifacts" / "reports"

TARGET = "failure_within_next_30min"
MACHINE_COLUMN = "machine_id"
TIME_COLUMN = "timestamp"
PURGE_TIMESTAMPS = 6

FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
    "temperature_c_diff_1",
    "temperature_c_rolling_mean_3",
    "temperature_c_rolling_mean_6",
    "temperature_c_rolling_mean_12",
    "temperature_c_deviation_from_past",
    "vibration_mm_s_diff_1",
    "vibration_mm_s_rolling_mean_3",
    "vibration_mm_s_rolling_mean_6",
    "vibration_mm_s_rolling_mean_12",
    "vibration_mm_s_deviation_from_past",
    "rotational_speed_rpm_diff_1",
    "rotational_speed_rpm_rolling_mean_3",
    "rotational_speed_rpm_rolling_mean_6",
    "rotational_speed_rpm_rolling_mean_12",
    "rotational_speed_rpm_deviation_from_past",
]


def evaluate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "rows": int(len(y_true)),
        "positive_labels": int(y_true.sum()),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "predicted_warnings": int(predictions.sum()),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, parse_dates=[TIME_COLUMN])
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], utc=True)
    data = data.sort_values([TIME_COLUMN, MACHINE_COLUMN]).reset_index(drop=True)

    required = {TIME_COLUMN, MACHINE_COLUMN, TARGET, *FEATURE_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    timestamps = pd.Index(data[TIME_COLUMN].drop_duplicates().sort_values())
    if len(timestamps) < 100:
        raise ValueError("Not enough unique timestamps for chronological evaluation.")

    train_cut = int(len(timestamps) * 0.60)
    validation_cut = int(len(timestamps) * 0.80)

    # Purge the final six timestamps of each earlier partition.
    train_times = timestamps[: train_cut - PURGE_TIMESTAMPS]
    validation_times = timestamps[train_cut : validation_cut - PURGE_TIMESTAMPS]
    test_times = timestamps[validation_cut:]

    train = data[data[TIME_COLUMN].isin(train_times)].copy()
    validation = data[data[TIME_COLUMN].isin(validation_times)].copy()
    test = data[data[TIME_COLUMN].isin(test_times)].copy()

    for name, part in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if part.empty or part[TARGET].nunique() < 2:
            raise ValueError(f"{name} partition is empty or does not contain both target classes.")

    # Confirm partition boundaries do not overlap.
    assert train[TIME_COLUMN].max() < validation[TIME_COLUMN].min()
    assert validation[TIME_COLUMN].max() < test[TIME_COLUMN].min()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train[FEATURE_COLUMNS], train[TARGET])

    validation_probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    test_probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    # Select the threshold on validation only.
    threshold_rows = []
    for threshold in np.round(np.arange(0.10, 0.951, 0.05), 2):
        threshold_rows.append(evaluate(validation[TARGET], validation_probabilities, threshold))

    threshold_table = pd.DataFrame(threshold_rows)

    # Highest validation F1; if tied, prefer the higher threshold.
    best_row = threshold_table.sort_values(
        ["f1_score", "threshold"], ascending=[False, False]
    ).iloc[0]
    selected_threshold = float(best_row["threshold"])

    # The final test is evaluated using the already selected threshold.
    test_metrics = evaluate(test[TARGET], test_probabilities, selected_threshold)

    report = {
        "evaluation": "chronological train/validation/test with purged boundaries",
        "dataset_type": "synthetic",
        "warning": (
            "Synthetic-data results do not establish real-world performance. "
            "The test period is independent of threshold selection, but this "
            "is still a single synthetic dataset."
        ),
        "purge_timestamps": PURGE_TIMESTAMPS,
        "purge_minutes": PURGE_TIMESTAMPS * 5,
        "split_boundaries": {
            "train_start": str(train[TIME_COLUMN].min()),
            "train_end": str(train[TIME_COLUMN].max()),
            "validation_start": str(validation[TIME_COLUMN].min()),
            "validation_end": str(validation[TIME_COLUMN].max()),
            "test_start": str(test[TIME_COLUMN].min()),
            "test_end": str(test[TIME_COLUMN].max()),
        },
        "partitions": {
            "train": {
                "rows": int(len(train)),
                "timestamps": int(train[TIME_COLUMN].nunique()),
                "positive_labels": int(train[TARGET].sum()),
            },
            "validation": {
                "rows": int(len(validation)),
                "timestamps": int(validation[TIME_COLUMN].nunique()),
                "positive_labels": int(validation[TARGET].sum()),
            },
            "test": {
                "rows": int(len(test)),
                "timestamps": int(test[TIME_COLUMN].nunique()),
                "positive_labels": int(test[TARGET].sum()),
            },
        },
        "threshold_selection": {
            "method": "maximum validation F1; ties prefer higher threshold",
            "selected_threshold": selected_threshold,
            "validation_metrics": evaluate(
                validation[TARGET],
                validation_probabilities,
                selected_threshold,
            ),
        },
        "final_test_metrics": test_metrics,
        "features": FEATURE_COLUMNS,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_path = REPORT_DIR / "three_way_validation_thresholds.csv"
    report_path = REPORT_DIR / "three_way_evaluation_summary.json"

    threshold_table.to_csv(threshold_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFORGE // INDUSTRIAL AI")
    print("=" * 58)
    print("Chronological three-way evaluation")
    print(f"Train rows:      {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Final test rows: {len(test):,}")
    print(f"Purge:           {PURGE_TIMESTAMPS * 5} minutes per boundary")
    print(f"Selected threshold (validation): {selected_threshold:.2f}")

    print("\nFINAL TEST METRICS — threshold selected on validation")
    for key, value in test_metrics.items():
        print(f"{key:24s}: {value}")

    print("\nSaved reports:")
    print(threshold_path.relative_to(ROOT))
    print(report_path.relative_to(ROOT))
    print("\nImportant: results are from synthetic data, not a real factory.")


if __name__ == "__main__":
    main()
