from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
OUTPUT_DIR = ROOT / "artifacts" / "reports"

TARGET = "failure_within_next_30min"
TIME = "timestamp"
MACHINE = "machine_id"
SENSORS = ["temperature_c", "vibration_mm_s", "rotational_speed_rpm"]
PURGE = 6
SEED = 42

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


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the project's causal features using all available history."""
    result = data.sort_values([MACHINE, TIME]).copy()

    for column in SENSORS:
        grouped = result.groupby(MACHINE, sort=False)[column]

        result[f"{column}_diff_1"] = grouped.diff()

        for window in (3, 6, 12):
            result[f"{column}_rolling_mean_{window}"] = result.groupby(MACHINE, sort=False)[
                column
            ].transform(lambda s, window=window: s.rolling(window=window, min_periods=1).mean())

        previous_mean = result.groupby(MACHINE, sort=False)[column].transform(
            lambda s: s.shift(1).rolling(window=12, min_periods=3).mean()
        )
        result[f"{column}_deviation_from_past"] = result[column] - previous_mean

    return result.sort_values([TIME, MACHINE]).reset_index(drop=True)


def score_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, predicted, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y, probabilities)), 4),
        "roc_auc": (
            round(float(roc_auc_score(y, probabilities)), 4) if len(np.unique(y)) == 2 else None
        ),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "predicted_warnings": int(predicted.sum()),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    original = pd.read_csv(DATA_PATH)
    original[TIME] = pd.to_datetime(original[TIME], utc=True)
    original = original.sort_values([TIME, MACHINE]).reset_index(drop=True)

    required = {TIME, MACHINE, TARGET, *SENSORS}
    missing = sorted(required - set(original.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use the same chronological 60/20/20 boundaries as evaluate_three_way.py.
    timestamps = pd.Index(original[TIME].drop_duplicates().sort_values())
    train_cut = int(len(timestamps) * 0.60)
    validation_cut = int(len(timestamps) * 0.80)

    train_times = timestamps[: train_cut - PURGE]
    validation_times = timestamps[train_cut : validation_cut - PURGE]
    test_times = timestamps[validation_cut:]

    train = original[original[TIME].isin(train_times)].copy()
    validation = original[original[TIME].isin(validation_times)].copy()
    test = original[original[TIME].isin(test_times)].copy()

    if any(part.empty for part in (train, validation, test)):
        raise ValueError("One or more chronological partitions are empty.")

    if any(part[TARGET].nunique() != 2 for part in (train, validation, test)):
        raise ValueError("Each partition must contain both target classes.")

    # Fit only on training rows, matching the three-way evaluation.
    model = Pipeline(
        [
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
    model.fit(train[FEATURES], train[TARGET].astype(int))

    validation_probabilities = model.predict_proba(validation[FEATURES])[:, 1]

    # Threshold is selected on validation data only; final test stays untouched.
    candidates = np.round(np.arange(0.10, 0.951, 0.05), 2)
    threshold_scores = []
    for threshold in candidates:
        predicted = (validation_probabilities >= threshold).astype(int)
        threshold_scores.append(
            {
                "threshold": float(threshold),
                "f1_score": float(
                    f1_score(
                        validation[TARGET].astype(int),
                        predicted,
                        zero_division=0,
                    )
                ),
            }
        )

    selected = sorted(
        threshold_scores,
        key=lambda row: (row["f1_score"], row["threshold"]),
        reverse=True,
    )[0]
    threshold = selected["threshold"]

    # Recompute features over the complete ordered sensor history.
    # This preserves each machine's preceding readings at the test boundary.
    clean_engineered = engineer_features(original)
    test_mask = clean_engineered[TIME].isin(test_times)
    clean_test = clean_engineered.loc[test_mask].copy()

    expected_test_rows = len(test)
    if len(clean_test) != expected_test_rows:
        raise ValueError(f"Test row mismatch: expected {expected_test_rows}, got {len(clean_test)}")

    y_test = clean_test[TARGET].astype(int).to_numpy()
    clean_x = clean_test[FEATURES].replace([np.inf, -np.inf], np.nan)
    clean_probabilities = model.predict_proba(clean_x)[:, 1]

    rng = np.random.default_rng(SEED)
    results = []

    def evaluate_condition(name: str, corrupted: pd.DataFrame, description: str):
        engineered = engineer_features(corrupted)
        condition_test = engineered.loc[engineered[TIME].isin(test_times)].copy()

        if len(condition_test) != expected_test_rows:
            raise ValueError(f"{name}: test row count changed after corruption.")

        x = condition_test[FEATURES].replace([np.inf, -np.inf], np.nan)
        probabilities = model.predict_proba(x)[:, 1]

        results.append(
            {
                "condition": name,
                "description": description,
                **score_metrics(y_test, probabilities, threshold),
                "mean_absolute_probability_change": round(
                    float(np.mean(np.abs(probabilities - clean_probabilities))), 6
                ),
                "changed_predictions_vs_clean": int(
                    np.sum((probabilities >= threshold) != (clean_probabilities >= threshold))
                ),
            }
        )

    results.append(
        {
            "condition": "clean_baseline",
            "description": "Uncorrupted sensor history and causal features",
            **score_metrics(y_test, clean_probabilities, threshold),
            "mean_absolute_probability_change": 0.0,
            "changed_predictions_vs_clean": 0,
        }
    )

    # Corrupt only test-period sensor readings; preserve earlier context.
    noisy = original.copy()
    for sensor in SENSORS:
        mask = noisy[TIME].isin(test_times)
        scale = max(float(noisy.loc[mask, sensor].std()), 1e-9)
        noisy.loc[mask, sensor] = noisy.loc[mask, sensor].astype(float) + rng.normal(
            0.0, 0.05 * scale, size=int(mask.sum())
        )
    evaluate_condition(
        "sensor_noise_5pct",
        noisy,
        "Gaussian noise with standard deviation 5% of each sensor's test-period standard deviation",
    )

    missing_data = original.copy()
    test_indices = missing_data.index[missing_data[TIME].isin(test_times)].to_numpy()
    missing_mask = rng.random((len(test_indices), len(SENSORS))) < 0.05
    for i, sensor in enumerate(SENSORS):
        missing_data.loc[test_indices[missing_mask[:, i]], sensor] = np.nan
    evaluate_condition(
        "missing_readings_5pct",
        missing_data,
        "Random 5% missing readings per sensor in the test period",
    )

    spikes = original.copy()
    test_indices = spikes.index[spikes[TIME].isin(test_times)].to_numpy()
    for sensor in SENSORS:
        count = max(1, int(round(len(test_indices) * 0.01)))
        chosen = rng.choice(test_indices, size=count, replace=False)
        scale = max(float(spikes.loc[test_indices, sensor].std()), 1e-9)
        spikes.loc[chosen, sensor] = (
            spikes.loc[chosen, sensor].astype(float).to_numpy()
            + rng.choice([-1.0, 1.0], size=count) * 5.0 * scale
        )
    evaluate_condition(
        "sensor_spikes_1pct",
        spikes,
        "1% of test-period rows per sensor receive a randomly signed 5-standard-deviation spike",
    )

    combined = original.copy()
    test_indices = combined.index[combined[TIME].isin(test_times)].to_numpy()
    test_mask = combined[TIME].isin(test_times)

    for sensor in SENSORS:
        scale = max(float(combined.loc[test_mask, sensor].std()), 1e-9)
        combined.loc[test_mask, sensor] = combined.loc[test_mask, sensor].astype(
            float
        ) + rng.normal(0.0, 0.05 * scale, size=int(test_mask.sum()))

        missing_count = int(round(len(test_indices) * 0.05))
        missing_indices = rng.choice(test_indices, size=missing_count, replace=False)
        combined.loc[missing_indices, sensor] = np.nan

        spike_count = max(1, int(round(len(test_indices) * 0.01)))
        spike_indices = rng.choice(test_indices, size=spike_count, replace=False)
        fill_value = float(original.loc[test_mask, sensor].median())
        prior = combined.loc[spike_indices, sensor].fillna(fill_value).to_numpy()
        scale = max(float(original.loc[test_mask, sensor].std()), 1e-9)
        combined.loc[spike_indices, sensor] = (
            prior + rng.choice([-1.0, 1.0], size=spike_count) * 5.0 * scale
        )

    evaluate_condition(
        "combined_corruption",
        combined,
        "5% Gaussian noise, 5% missing readings, and 1% large spikes per sensor in test period",
    )

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "sensor_robustness_v2_results.csv"
    json_path = OUTPUT_DIR / "sensor_robustness_v2_summary.json"
    results_df.to_csv(csv_path, index=False)

    report = {
        "evaluation": "robustness test aligned with chronological three-way split",
        "dataset_type": "synthetic",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(clean_test)),
        "test_start": str(clean_test[TIME].min()),
        "test_end": str(clean_test[TIME].max()),
        "purge_minutes": PURGE * 5,
        "threshold_selection": "maximum validation F1; ties prefer higher threshold",
        "selected_threshold": float(threshold),
        "random_seed": SEED,
        "results": results,
        "limitations": [
            "Synthetic-data results do not establish field performance.",
            "Corruption magnitudes are illustrative rather than calibrated to industrial sensors.",
            (
                "Missing values are passed to the trained pipeline imputer "
                "after feature recomputation."
            ),
        ],
    }

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFORGE // INDUSTRIAL AI")
    print("=" * 60)
    print("Corrected sensor robustness evaluation")
    print(f"Train rows:      {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Final test rows: {len(clean_test):,}")
    print(f"Purge:           {PURGE * 5} minutes per boundary")
    print(f"Selected threshold (validation): {threshold:.2f}")
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
    print("\nReminder: this is a synthetic corruption simulation.")


if __name__ == "__main__":
    main()
