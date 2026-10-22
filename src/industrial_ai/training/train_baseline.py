from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_features.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "models"

TARGET = "failure_within_next_30min"
MACHINE_COLUMN = "machine_id"
TIME_COLUMN = "timestamp"
PURGE_TIMESTAMPS = 6
RANDOM_STATE = 42

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


def make_logistic_model() -> Pipeline:
    """Build a fresh balanced logistic-regression pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate probabilities at a fixed decision threshold."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_labels": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "predicted_warnings": int(predictions.sum()),
    }


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float | None]:
    """Calculate threshold-independent metrics."""
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": (float(roc_auc_score(y_true, probabilities)) if y_true.nunique() == 2 else None),
    }


def select_threshold(
    y_validation: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    """Select maximum validation F1; break ties toward higher thresholds."""
    candidates: list[dict[str, Any]] = []

    for threshold in np.round(np.arange(0.10, 1.00, 0.05), 2):
        candidates.append(
            evaluate_at_threshold(
                y_validation,
                probabilities,
                float(threshold),
            )
        )

    best = max(
        candidates,
        key=lambda result: (result["f1_score"], result["threshold"]),
    )
    return float(best["threshold"]), candidates


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}. Run prepare_dataset.py first."
        )

    data = pd.read_csv(INPUT_PATH)
    required = {
        *FEATURE_COLUMNS,
        TARGET,
        MACHINE_COLUMN,
        TIME_COLUMN,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], utc=True)
    if data[TIME_COLUMN].isna().any():
        raise ValueError("Dataset contains missing or invalid timestamps.")

    data = data.sort_values([TIME_COLUMN, MACHINE_COLUMN]).reset_index(drop=True)

    timestamps = pd.Index(data[TIME_COLUMN].drop_duplicates().sort_values())
    if len(timestamps) < 20:
        raise ValueError("Not enough unique timestamps for a purged split.")

    train_cut = int(len(timestamps) * 0.60)
    validation_cut = int(len(timestamps) * 0.80)

    if train_cut <= PURGE_TIMESTAMPS:
        raise ValueError("Training partition is too small for the purge.")

    train_times = timestamps[: train_cut - PURGE_TIMESTAMPS]
    validation_times = timestamps[train_cut : validation_cut - PURGE_TIMESTAMPS]
    test_times = timestamps[validation_cut:]

    if len(train_times) == 0 or len(validation_times) == 0 or len(test_times) == 0:
        raise ValueError("Chronological split produced an empty partition.")

    train = data.loc[data[TIME_COLUMN].isin(train_times)].copy()
    validation = data.loc[data[TIME_COLUMN].isin(validation_times)].copy()
    test = data.loc[data[TIME_COLUMN].isin(test_times)].copy()

    if any(part[TARGET].nunique() != 2 for part in (train, validation, test)):
        raise ValueError("Every partition must contain both target classes.")

    def get_x(partition: pd.DataFrame) -> pd.DataFrame:
        return partition[FEATURE_COLUMNS].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    x_train = get_x(train)
    x_validation = get_x(validation)
    x_test = get_x(test)

    y_train = train[TARGET].astype(int)
    y_validation = validation[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    # Train only on the training period while selecting the threshold.
    selection_model = make_logistic_model()
    selection_model.fit(x_train, y_train)

    validation_probabilities = selection_model.predict_proba(x_validation)[:, 1]

    threshold, validation_thresholds = select_threshold(
        y_validation,
        validation_probabilities,
    )

    validation_metrics = evaluate_at_threshold(
        y_validation,
        validation_probabilities,
        threshold,
    )

    # Final evaluation model remains trained only on the training period.
    test_probabilities = selection_model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_probabilities, threshold)
    test_metrics.update(evaluate_probabilities(y_test, test_probabilities))

    # A simple dummy baseline is evaluated on the same final test period.
    dummy_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    dummy_model.fit(x_train, y_train)
    dummy_probabilities = dummy_model.predict_proba(x_test)[:, 1]
    dummy_metrics = evaluate_at_threshold(y_test, dummy_probabilities, 0.50)
    dummy_metrics.update(evaluate_probabilities(y_test, dummy_probabilities))

    # Train the deployable model on all pre-test observations.
    pre_test = pd.concat([train, validation], ignore_index=True)
    deployment_model = make_logistic_model()
    deployment_model.fit(get_x(pre_test), pre_test[TARGET].astype(int))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "predictive_maintenance_logistic_regression.joblib"
    metadata_path = MODEL_DIR / "predictive_maintenance_metadata.json"
    joblib.dump(deployment_model, model_path)

    metadata = {
        "model_name": "logistic_regression_balanced",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "training_rows": int(len(pre_test)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "trained_at": datetime.now(UTC).isoformat(),
        "model_file": str(model_path.resolve()),
        "threshold": threshold,
        "threshold_selection": "maximum validation F1; ties prefer higher threshold",
        "dataset_type": "synthetic",
        "deployment_training": "training plus validation partitions",
        "evaluation_note": (
            "Final test metrics belong to the model trained on the training "
            "partition, not this refitted deployment artifact."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(INPUT_PATH.resolve()),
        "evaluation": "chronological three-way split with purged boundaries",
        "dataset_type": "synthetic",
        "purge_timestamps": PURGE_TIMESTAMPS,
        "purge_minutes": PURGE_TIMESTAMPS * 5,
        "partitions": {
            "train": {
                "rows": int(len(train)),
                "start": str(train[TIME_COLUMN].min()),
                "end": str(train[TIME_COLUMN].max()),
                "positive_labels": int(y_train.sum()),
            },
            "validation": {
                "rows": int(len(validation)),
                "start": str(validation[TIME_COLUMN].min()),
                "end": str(validation[TIME_COLUMN].max()),
                "positive_labels": int(y_validation.sum()),
            },
            "test": {
                "rows": int(len(test)),
                "start": str(test[TIME_COLUMN].min()),
                "end": str(test[TIME_COLUMN].max()),
                "positive_labels": int(y_test.sum()),
            },
        },
        "threshold_selection": {
            "method": "maximum validation F1; ties prefer higher threshold",
            "selected_threshold": threshold,
            "validation_metrics": validation_metrics,
            "threshold_candidates": validation_thresholds,
            "threshold_independent_metrics": evaluate_probabilities(
                y_validation,
                validation_probabilities,
            ),
        },
        "final_test_metrics": test_metrics,
        "dummy_baseline_test_metrics": dummy_metrics,
        "deployment_model": {
            "file": str(model_path.resolve()),
            "training_rows": int(len(pre_test)),
            "threshold": threshold,
        },
        "features": FEATURE_COLUMNS,
        "limitations": [
            "Synthetic data only; metrics do not establish real-world performance.",
            "One synthetic dataset and one chronological test period.",
            "Test metrics evaluate the training-only model, not the refitted deployment model.",
            "Thresholds need real-world calibration beyond synthetic validation data.",
        ],
    }

    report_path = REPORT_DIR / "baseline_model_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    thresholds_path = REPORT_DIR / "baseline_validation_thresholds.csv"
    pd.DataFrame(validation_thresholds).to_csv(
        thresholds_path,
        index=False,
    )

    print("\nFORGE // INDUSTRIAL AI")
    print("=" * 58)
    print("Purged chronological three-way model training")
    print(f"Training rows:       {len(train):,}")
    print(f"Validation rows:     {len(validation):,}")
    print(f"Final test rows:     {len(test):,}")
    print(f"Purge per boundary:  {PURGE_TIMESTAMPS * 5} minutes")
    print(f"Selected threshold:  {threshold:.2f}")
    print("\nFINAL TEST METRICS")
    for key, value in test_metrics.items():
        print(f"{key:24}: {value}")
    print("\nSaved deployment model and metadata.")
    print(f"Model:    {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")
    print(f"Report:   {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Thresholds: {thresholds_path.relative_to(PROJECT_ROOT)}")
    print("\nReminder: results use synthetic data.")


if __name__ == "__main__":
    main()
