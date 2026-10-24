from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "industrial_features.csv"
MODEL_PATH = ROOT / "artifacts" / "models" / "predictive_maintenance_logistic_regression.joblib"
OUTPUT_DIR = ROOT / "artifacts" / "reports"
TARGET = "failure_within_next_30min"
THRESHOLD = 0.95
PURGE = 6
RANDOM_SEED = 42

SENSORS = [
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
]

FEATURES = [
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


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Recalculate causal features after sensor corruption."""
    result = data.copy()
    result = result.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    for column in SENSORS:
        grouped = result.groupby("machine_id", sort=False)[column]

        result[f"{column}_diff_1"] = grouped.diff()

        for window in (3, 6, 12):
            result[f"{column}_rolling_mean_{window}"] = result.groupby("machine_id", sort=False)[
                column
            ].transform(
                lambda series, window=window: series.rolling(window=window, min_periods=1).mean()
            )

        previous_mean = result.groupby("machine_id", sort=False)[column].transform(
            lambda series: series.shift(1).rolling(window=12, min_periods=3).mean()
        )
        result[f"{column}_deviation_from_past"] = result[column] - previous_mean

    return result


def metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | None]:
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": THRESHOLD,
        "rows": int(len(labels)),
        "positive_labels": int(labels.sum()),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(labels, probabilities)), 4),
        "roc_auc": (
            round(float(roc_auc_score(labels, probabilities)), 4)
            if len(np.unique(labels)) == 2
            else None
        ),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "predicted_warnings": int(predictions.sum()),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    data = pd.read_csv(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["timestamp", "machine_id"]).reset_index(drop=True)

    required = {"timestamp", "machine_id", TARGET, *SENSORS, *FEATURES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    timestamps = pd.Index(data["timestamp"].drop_duplicates().sort_values())
    validation_cut = int(len(timestamps) * 0.80)
    test_times = timestamps[validation_cut:]

    # Match the independent test period used in the three-way evaluation.
    test = data[data["timestamp"].isin(test_times)].copy()
    if test.empty or test[TARGET].nunique() != 2:
        raise ValueError("Test period is empty or lacks one target class.")

    model = joblib.load(MODEL_PATH)
    labels = test[TARGET].astype(int).to_numpy()

    # Use the original, stored engineered features for the clean baseline.
    clean_x = test[FEATURES].replace([np.inf, -np.inf], np.nan)
    clean_probabilities = model.predict_proba(clean_x)[:, 1]

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    def evaluate_condition(
        name: str,
        corrupted_sensors: pd.DataFrame,
        corruption_description: str,
    ) -> None:
        engineered = make_features(corrupted_sensors)
        x = engineered[FEATURES].replace([np.inf, -np.inf], np.nan)
        probabilities = model.predict_proba(x)[:, 1]

        record = {
            "condition": name,
            "corruption": corruption_description,
            **metrics(labels, probabilities),
            "mean_absolute_probability_change": round(
                float(np.mean(np.abs(probabilities - clean_probabilities))), 6
            ),
            "changed_predictions_vs_clean": int(
                np.sum((probabilities >= THRESHOLD) != (clean_probabilities >= THRESHOLD))
            ),
        }
        results.append(record)

    # Baseline: unchanged sensor readings and existing features.
    results.append(
        {
            "condition": "clean_baseline",
            "corruption": "No corruption; stored features",
            **metrics(labels, clean_probabilities),
            "mean_absolute_probability_change": 0.0,
            "changed_predictions_vs_clean": 0,
        }
    )

    # Corruption conditions start from the same untouched test data.
    noisy = test.copy()
    for sensor in SENSORS:
        values = noisy[sensor].astype(float)
        scale = max(float(values.std()), 1e-9)
        noisy[sensor] = values + rng.normal(loc=0.0, scale=0.05 * scale, size=len(noisy))
    evaluate_condition(
        "sensor_noise_5pct",
        noisy,
        "Independent Gaussian noise; 5% of each sensor's test-set standard deviation",
    )

    missing_data = test.copy()
    missing_mask = rng.random((len(missing_data), len(SENSORS))) < 0.05
    for index, sensor in enumerate(SENSORS):
        missing_data.loc[missing_mask[:, index], sensor] = np.nan
    evaluate_condition(
        "missing_readings_5pct",
        missing_data,
        "Randomly missing 5% of readings per sensor; features recomputed",
    )

    spikes = test.copy()
    for sensor in SENSORS:
        count = max(1, int(round(len(spikes) * 0.01)))
        indices = rng.choice(spikes.index.to_numpy(), size=count, replace=False)
        scale = max(float(spikes[sensor].std()), 1e-9)
        signs = rng.choice([-1.0, 1.0], size=count)
        spikes.loc[indices, sensor] = (
            spikes.loc[indices, sensor].astype(float).to_numpy() + signs * 5.0 * scale
        )
    evaluate_condition(
        "sensor_spikes_1pct",
        spikes,
        "1% of rows per sensor receive a randomly signed 5-standard-deviation spike",
    )

    # Combined moderate corruption: noise, missingness and spikes.
    combined = test.copy()
    for sensor in SENSORS:
        values = combined[sensor].astype(float)
        scale = max(float(values.std()), 1e-9)
        combined[sensor] = values + rng.normal(0.0, 0.05 * scale, size=len(combined))

    combined_missing = rng.random((len(combined), len(SENSORS))) < 0.05
    for index, sensor in enumerate(SENSORS):
        combined.loc[combined_missing[:, index], sensor] = np.nan

        count = max(1, int(round(len(combined) * 0.01)))
        indices = rng.choice(combined.index.to_numpy(), size=count, replace=False)
        scale = max(float(test[sensor].std()), 1e-9)
        signs = rng.choice([-1.0, 1.0], size=count)
        combined.loc[indices, sensor] = (
            combined.loc[indices, sensor].fillna(float(test[sensor].median())).to_numpy()
            + signs * 5.0 * scale
        )

    evaluate_condition(
        "combined_corruption",
        combined,
        "5% Gaussian noise, 5% missing readings, and 1% large spikes per sensor",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "sensor_robustness_results.csv"
    json_path = OUTPUT_DIR / "sensor_robustness_summary.json"

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)

    report = {
        "evaluation": "sensor corruption robustness",
        "dataset_type": "synthetic",
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        "test_rows": int(len(test)),
        "positive_labels": int(labels.sum()),
        "threshold": THRESHOLD,
        "random_seed": RANDOM_SEED,
        "model_path": str(MODEL_PATH),
        "feature_handling": (
            "Recomputed causal features after sensor corruption. "
            "Missing values are handled by the saved model's imputer."
        ),
        "limitations": [
            "This is a synthetic corruption simulation, not field validation.",
            "Corruption rates and magnitudes are illustrative, not calibrated to real sensors.",
            "The clean baseline uses stored features; corrupted conditions recompute features.",
            "Threshold is fixed in advance; no threshold is selected using test results.",
        ],
        "results": results,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFORGE // INDUSTRIAL AI")
    print("=" * 60)
    print("Sensor robustness evaluation")
    print(f"Test rows: {len(test):,}")
    print(f"Positive labels: {int(labels.sum()):,}")
    print(f"Fixed threshold: {THRESHOLD:.2f}")
    print("\nRESULTS")
    print(
        results_df[
            [
                "condition",
                "precision",
                "recall",
                "f1_score",
                "false_positives",
                "false_negatives",
                "predicted_warnings",
                "mean_absolute_probability_change",
                "changed_predictions_vs_clean",
            ]
        ].to_string(index=False)
    )
    print("\nSaved reports:")
    print(csv_path.relative_to(ROOT))
    print(json_path.relative_to(ROOT))
    print("\nReminder: these results measure synthetic corruption robustness only.")


if __name__ == "__main__":
    main()
