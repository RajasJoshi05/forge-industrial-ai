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

SEED = 42
PURGE = 6
DISCONNECTION_ROWS = 12  # 60 minutes at 5-minute sampling
STUCK_ROWS = 12  # 60 minutes
DRIFT_ROWS = 24  # 120 minutes
ABNORMAL_ROWS = 12  # 60 minutes
THRESHOLD_CANDIDATES = np.round(np.arange(0.10, 0.951, 0.05), 2)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild causal features using the complete available sensor history."""
    result = data.sort_values([MACHINE, TIME]).copy()

    for sensor in SENSORS:
        grouped = result.groupby(MACHINE, sort=False)[sensor]

        result[f"{sensor}_diff_1"] = grouped.diff()

        for window in (3, 6, 12):
            result[f"{sensor}_rolling_mean_{window}"] = result.groupby(MACHINE, sort=False)[
                sensor
            ].transform(
                lambda series, window=window: series.rolling(window=window, min_periods=1).mean()
            )

        previous_mean = result.groupby(MACHINE, sort=False)[sensor].transform(
            lambda series: series.shift(1).rolling(window=12, min_periods=3).mean()
        )
        result[f"{sensor}_deviation_from_past"] = result[sensor] - previous_mean

    return result.sort_values([TIME, MACHINE]).reset_index(drop=True)


def metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "roc_auc": (
            round(float(roc_auc_score(y_true, probabilities)), 4)
            if len(np.unique(y_true)) == 2
            else None
        ),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "predicted_warnings": int(predictions.sum()),
    }


def test_mask_for(machine_data: pd.DataFrame, test_times: pd.Index) -> np.ndarray:
    return machine_data[TIME].isin(test_times).to_numpy()


def choose_start(rng: np.random.Generator, length: int, duration: int) -> int:
    if length < duration:
        raise ValueError(f"Machine has only {length} test readings; {duration} required.")
    return int(rng.integers(0, length - duration + 1))


def apply_fault(
    data: pd.DataFrame,
    test_times: pd.Index,
    fault: str,
    seed: int,
) -> pd.DataFrame:
    """Inject one simulated fault segment per machine and sensor."""
    corrupted = data.copy()
    rng = np.random.default_rng(seed)

    for _, group in corrupted.groupby(MACHINE, sort=False):
        ordered_indices = group.loc[group[TIME].isin(test_times)].sort_values(TIME).index.to_numpy()

        if len(ordered_indices) == 0:
            continue

        for sensor in SENSORS:
            original_values = corrupted.loc[ordered_indices, sensor].astype(float).to_numpy()

            valid = original_values[np.isfinite(original_values)]
            if len(valid) == 0:
                continue

            sensor_scale = max(float(np.std(valid)), 1e-9)

            if fault == "disconnection":
                duration = DISCONNECTION_ROWS
                start = choose_start(rng, len(ordered_indices), duration)
                chosen = ordered_indices[start : start + duration]
                corrupted.loc[chosen, sensor] = np.nan

            elif fault == "stuck_sensor":
                duration = STUCK_ROWS
                start = choose_start(rng, len(ordered_indices), duration)
                chosen = ordered_indices[start : start + duration]
                frozen_value = original_values[start]
                corrupted.loc[chosen, sensor] = frozen_value

            elif fault == "gradual_drift":
                duration = DRIFT_ROWS
                start = choose_start(rng, len(ordered_indices), duration)
                chosen = ordered_indices[start : start + duration]

                # Drift rises gradually from zero to a two-SD offset.
                direction = float(rng.choice([-1.0, 1.0]))
                offsets = np.linspace(0.0, direction * 2.0 * sensor_scale, duration)
                corrupted.loc[chosen, sensor] = original_values[start : start + duration] + offsets

            elif fault == "persistent_abnormal":
                duration = ABNORMAL_ROWS
                start = choose_start(rng, len(ordered_indices), duration)
                chosen = ordered_indices[start : start + duration]

                # A persistent offset rather than a single-reading spike.
                direction = float(rng.choice([-1.0, 1.0]))
                abnormal_value = float(np.median(valid)) + direction * 5.0 * sensor_scale
                corrupted.loc[chosen, sensor] = abnormal_value

            else:
                raise ValueError(f"Unknown fault: {fault}")

    return corrupted


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data[TIME] = pd.to_datetime(data[TIME], utc=True)
    data = data.sort_values([TIME, MACHINE]).reset_index(drop=True)

    required = {TIME, MACHINE, TARGET, *SENSORS, *FEATURES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    # Identical chronological split to the corrected robustness experiment.
    timestamps = pd.Index(data[TIME].drop_duplicates().sort_values())
    train_cut = int(len(timestamps) * 0.60)
    validation_cut = int(len(timestamps) * 0.80)

    train_times = timestamps[: train_cut - PURGE]
    validation_times = timestamps[train_cut : validation_cut - PURGE]
    test_times = timestamps[validation_cut:]

    train = data[data[TIME].isin(train_times)]
    validation = data[data[TIME].isin(validation_times)]
    test = data[data[TIME].isin(test_times)]

    if any(part.empty for part in (train, validation, test)):
        raise ValueError("A chronological data partition is empty.")

    # Same model pipeline as the corrected three-way evaluation.
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

    # Select threshold on validation data only.
    candidates = []
    y_validation = validation[TARGET].astype(int).to_numpy()

    for threshold in THRESHOLD_CANDIDATES:
        predictions = (validation_probabilities >= threshold).astype(int)
        candidates.append(
            (
                float(f1_score(y_validation, predictions, zero_division=0)),
                float(threshold),
            )
        )

    # Maximum F1; ties prefer the higher threshold.
    _, threshold = max(candidates, key=lambda item: (item[0], item[1]))

    # Clean baseline and all fault scenarios use full sensor history.
    clean_features = engineer_features(data)
    clean_test = clean_features[clean_features[TIME].isin(test_times)].copy()

    if len(clean_test) != len(test):
        raise ValueError("Clean test row count does not match the split.")

    y_test = clean_test[TARGET].astype(int).to_numpy()
    clean_x = clean_test[FEATURES].replace([np.inf, -np.inf], np.nan)
    clean_probabilities = model.predict_proba(clean_x)[:, 1]

    results = [
        {
            "condition": "clean_baseline",
            "fault_duration": "None",
            **metrics(y_test, clean_probabilities, threshold),
            "mean_absolute_probability_change": 0.0,
            "changed_predictions_vs_clean": 0,
        }
    ]

    fault_specs = [
        (
            "sensor_disconnection",
            "disconnection",
            "60 minutes of missing readings per machine and sensor",
            101,
        ),
        (
            "stuck_sensor",
            "stuck_sensor",
            "60 minutes of readings frozen at the segment's first value",
            202,
        ),
        (
            "gradual_sensor_drift",
            "gradual_drift",
            "120-minute drift ramp ending at a 2-standard-deviation offset",
            303,
        ),
        (
            "persistent_abnormal_readings",
            "persistent_abnormal",
            "60 minutes held at median plus or minus 5 standard deviations",
            404,
        ),
    ]

    for name, fault, description, seed in fault_specs:
        corrupted = apply_fault(data, test_times, fault, seed)
        engineered = engineer_features(corrupted)
        fault_test = engineered[engineered[TIME].isin(test_times)].copy()

        if len(fault_test) != len(clean_test):
            raise ValueError(f"{name}: test row count changed.")

        x = fault_test[FEATURES].replace([np.inf, -np.inf], np.nan)
        probabilities = model.predict_proba(x)[:, 1]

        results.append(
            {
                "condition": name,
                "fault_duration": description,
                **metrics(y_test, probabilities, threshold),
                "mean_absolute_probability_change": round(
                    float(np.mean(np.abs(probabilities - clean_probabilities))),
                    6,
                ),
                "changed_predictions_vs_clean": int(
                    np.sum((probabilities >= threshold) != (clean_probabilities >= threshold))
                ),
            }
        )

    # Combined fault scenario: all four fault types, on separate
    # deterministic draws, with possible overlap between fault windows.
    combined = data.copy()
    for fault, seed in [
        ("disconnection", 501),
        ("stuck_sensor", 502),
        ("gradual_drift", 503),
        ("persistent_abnormal", 504),
    ]:
        combined = apply_fault(combined, test_times, fault, seed)

    combined_features = engineer_features(combined)
    combined_test = combined_features[combined_features[TIME].isin(test_times)].copy()

    combined_x = combined_test[FEATURES].replace([np.inf, -np.inf], np.nan)
    combined_probabilities = model.predict_proba(combined_x)[:, 1]

    results.append(
        {
            "condition": "combined_sensor_faults",
            "fault_duration": (
                "Disconnection, stuck readings, gradual drift, and "
                "persistent abnormal readings; windows may overlap"
            ),
            **metrics(y_test, combined_probabilities, threshold),
            "mean_absolute_probability_change": round(
                float(np.mean(np.abs(combined_probabilities - clean_probabilities))),
                6,
            ),
            "changed_predictions_vs_clean": int(
                np.sum((combined_probabilities >= threshold) != (clean_probabilities >= threshold))
            ),
        }
    )

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "sensor_failure_modes_results.csv"
    json_path = OUTPUT_DIR / "sensor_failure_modes_summary.json"

    results_df.to_csv(csv_path, index=False)

    report = {
        "evaluation": "chronological sensor failure stress test",
        "dataset_type": "synthetic",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(clean_test)),
        "test_start": str(clean_test[TIME].min()),
        "test_end": str(clean_test[TIME].max()),
        "purge_minutes": PURGE * 5,
        "selected_validation_threshold": float(threshold),
        "random_seeds": [101, 202, 303, 404, 501, 502, 503, 504],
        "results": results,
        "limitations": [
            "All faults are simulated, not calibrated to real sensor logs.",
            "Fault windows are injected into held-out sensor readings only.",
            "This experiment does not establish real-world predictive-maintenance reliability.",
            "The combined scenario can contain overlapping fault windows.",
        ],
    }

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFORGE // INDUSTRIAL AI")
    print("=" * 60)
    print("Sensor failure stress test")
    print(f"Train rows:      {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Final test rows: {len(clean_test):,}")
    print(f"Purge:           {PURGE * 5} minutes per boundary")
    print(f"Selected threshold (validation): {threshold:.2f}")

    print("\nRESULTS")
    columns = [
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
    print(results_df[columns].to_string(index=False))

    print("\nSaved reports:")
    print(csv_path.relative_to(ROOT))
    print(json_path.relative_to(ROOT))
    print("\nReminder: simulated sensor faults are not real-world validation.")


if __name__ == "__main__":
    main()
