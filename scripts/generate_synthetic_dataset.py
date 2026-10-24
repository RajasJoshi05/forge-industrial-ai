from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MACHINE_COUNT = 12
DAYS = 14
INTERVAL_MINUTES = 5
FAILURES_PER_MACHINE = 4
WARNING_WINDOW_MINUTES = 30
DEGRADATION_WINDOW_MINUTES = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "industrial_sensor_dataset.csv"


def generate_failure_events(
    rng: np.random.Generator,
    observation_count: int,
) -> list[int]:
    """Generate failure indices with at least 12 hours between events."""
    minimum_gap = 12 * 60 // INTERVAL_MINUTES
    candidates = rng.permutation(np.arange(minimum_gap, observation_count - 1))

    selected: list[int] = []

    for candidate in candidates:
        index = int(candidate)

        if all(abs(index - existing) >= minimum_gap for existing in selected):
            selected.append(index)

        if len(selected) == FAILURES_PER_MACHINE:
            break

    if len(selected) != FAILURES_PER_MACHINE:
        raise RuntimeError("Could not generate enough separated failure events.")

    return sorted(selected)


def is_failure_within_warning_window(
    observation_index: int,
    failure_indices: list[int],
    warning_window_minutes: int = WARNING_WINDOW_MINUTES,
    interval_minutes: int = INTERVAL_MINUTES,
) -> bool:
    """Return whether a failure occurs within the configured future warning window."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive.")

    if warning_window_minutes < 0:
        raise ValueError("warning_window_minutes cannot be negative.")

    warning_steps = warning_window_minutes // interval_minutes

    return any(
        0 < failure_index - observation_index <= warning_steps for failure_index in failure_indices
    )


def generate_dataset() -> pd.DataFrame:
    """Generate reproducible sensor observations and future-failure labels."""
    rng = np.random.default_rng(SEED)

    timestamps = pd.date_range(
        start="2026-09-01T00:00:00Z",
        periods=DAYS * 24 * 60 // INTERVAL_MINUTES,
        freq=f"{INTERVAL_MINUTES}min",
    )

    records: list[dict[str, object]] = []

    for machine_number in range(1, MACHINE_COUNT + 1):
        machine_id = f"M-{machine_number:03d}"
        observation_count = len(timestamps)

        failure_indices = generate_failure_events(rng, observation_count)

        temperature_baseline = rng.uniform(55.0, 70.0)
        vibration_baseline = rng.uniform(1.0, 2.5)
        speed_baseline = rng.uniform(1400.0, 1800.0)

        for index, timestamp in enumerate(timestamps):
            severity = 0.0

            for failure_index in failure_indices:
                minutes_until_failure = (failure_index - index) * INTERVAL_MINUTES

                if 0 < minutes_until_failure <= DEGRADATION_WINDOW_MINUTES:
                    severity = max(
                        severity,
                        1.0 - minutes_until_failure / DEGRADATION_WINDOW_MINUTES,
                    )

            temperature = temperature_baseline + 18.0 * severity + rng.normal(0.0, 1.5)

            vibration = vibration_baseline + 5.0 * severity + rng.normal(0.0, 0.25)

            rotational_speed = speed_baseline - 220.0 * severity + rng.normal(0.0, 20.0)

            failure_within_window = is_failure_within_warning_window(
                observation_index=index,
                failure_indices=failure_indices,
            )

            records.append(
                {
                    "machine_id": machine_id,
                    "timestamp": timestamp.isoformat(),
                    "temperature_c": round(float(temperature), 3),
                    "vibration_mm_s": round(float(max(vibration, 0.0)), 3),
                    "rotational_speed_rpm": round(
                        float(max(rotational_speed, 0.0)),
                        3,
                    ),
                    "failure_within_next_30min": int(failure_within_window),
                }
            )

    dataset = pd.DataFrame(records)

    return dataset.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)


def main() -> None:
    dataset = generate_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print("Synthetic industrial sensor dataset generated.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Machines: {dataset['machine_id'].nunique()}")
    print(f"Total observations: {len(dataset):,}")
    print(f"Positive failure-warning observations: {dataset['failure_within_next_30min'].sum():,}")
    print(f"Positive class percentage: {dataset['failure_within_next_30min'].mean() * 100:.3f}%")
    print(f"Missing values: {int(dataset.isna().sum().sum())}")


if __name__ == "__main__":
    main()
