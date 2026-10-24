from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = [
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
]

MACHINE_COLUMN = "machine_id"
TIME_COLUMN = "timestamp"

# Set these to verified limits from the actual machine/sensor specifications.
# None means the engineering limit has not yet been configured.
ENGINEERING_LIMITS: dict[str, tuple[float, float] | None] = {
    "temperature_c": None,
    "vibration_mm_s": None,
    "rotational_speed_rpm": None,
}

# Configure these from machine-specific sensor specifications or
# historical healthy-operation data before using drift alerts operationally.
DRIFT_THRESHOLDS: dict[str, float | None] = {
    "temperature_c": None,
    "vibration_mm_s": None,
    "rotational_speed_rpm": None,
}

FROZEN_WINDOW = 12  # 12 readings = 60 minutes at 5-minute intervals
DRIFT_WINDOW = 12  # Compare current reading with the preceding 60 minutes


def monitor_sensor_health(
    data: pd.DataFrame,
    engineering_limits: dict[str, tuple[float, float] | None] | None = None,
    drift_thresholds: dict[str, float | None] | None = None,
    frozen_window: int = FROZEN_WINDOW,
    drift_window: int = DRIFT_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assess sensor input quality without using the prediction target.

    Returns:
        row_results: original rows plus sensor-health flags and a row-level status.
        summary: one row per machine and sensor with fault counts.

    Drift is flagged when the absolute difference from the previous
    rolling median exceeds a configured sensor-specific threshold.
    """
    if frozen_window < 2 or drift_window < 2:
        raise ValueError("Window lengths must be at least 2.")

    required = {MACHINE_COLUMN, TIME_COLUMN, *SENSORS}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    limits = ENGINEERING_LIMITS if engineering_limits is None else engineering_limits
    drift_limits = DRIFT_THRESHOLDS if drift_thresholds is None else drift_thresholds

    result = data.copy()
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], utc=True)
    result["_original_order"] = np.arange(len(result))
    result = result.sort_values([MACHINE_COLUMN, TIME_COLUMN]).reset_index(drop=True)

    summary_rows: list[dict] = []
    all_flag_columns: list[str] = []

    for sensor in SENSORS:
        missing_flag = f"{sensor}_missing"
        frozen_flag = f"{sensor}_frozen"
        abnormal_flag = f"{sensor}_outside_limits"
        drift_flag = f"{sensor}_drift"

        all_flag_columns.extend([missing_flag, frozen_flag, abnormal_flag, drift_flag])

        result[sensor] = pd.to_numeric(result[sensor], errors="coerce")
        result[missing_flag] = result[sensor].isna()

        frozen_parts = []
        abnormal_parts = []
        drift_parts = []

        for _, group in result.groupby(MACHINE_COLUMN, sort=False):
            values = group[sensor]
            indices = group.index

            # Detect a full window of exactly identical, non-missing readings.
            frozen = (
                values.rolling(
                    window=frozen_window,
                    min_periods=frozen_window,
                )
                .apply(
                    lambda window: float(np.all(window == window[0])),
                    raw=True,
                )
                .eq(1.0)
            )
            frozen_parts.append(frozen.reindex(indices))

            # Engineering bounds are only applied when verified limits exist.
            bounds = limits.get(sensor)
            if bounds is None:
                abnormal = pd.Series(False, index=indices)
            else:
                lower, upper = bounds
                if lower >= upper:
                    raise ValueError(f"Invalid engineering limits for {sensor}: {bounds}")
                abnormal = values.notna() & ((values < lower) | (values > upper))
            abnormal_parts.append(abnormal.reindex(indices))

            # Use only preceding readings to create the drift baseline.
            baseline = (
                values.shift(1)
                .rolling(
                    window=drift_window,
                    min_periods=drift_window,
                )
                .median()
            )

            drift_limit = drift_limits.get(sensor)
            if drift_limit is None:
                drift = pd.Series(False, index=indices)
            else:
                if drift_limit <= 0:
                    raise ValueError(f"Drift threshold for {sensor} must be positive.")
                drift = (
                    values.notna() & baseline.notna() & ((values - baseline).abs() > drift_limit)
                )
            drift_parts.append(drift.reindex(indices))

        result[frozen_flag] = (
            pd.concat(frozen_parts).reindex(result.index).fillna(False).astype(bool)
        )
        result[abnormal_flag] = (
            pd.concat(abnormal_parts).reindex(result.index).fillna(False).astype(bool)
        )
        result[drift_flag] = pd.concat(drift_parts).reindex(result.index).fillna(False).astype(bool)

        for machine_id, group in result.groupby(MACHINE_COLUMN, sort=False):
            summary_rows.append(
                {
                    "machine_id": machine_id,
                    "sensor": sensor,
                    "readings": int(len(group)),
                    "missing_count": int(group[missing_flag].sum()),
                    "frozen_count": int(group[frozen_flag].sum()),
                    "outside_limits_count": int(group[abnormal_flag].sum()),
                    "drift_count": int(group[drift_flag].sum()),
                    "engineering_limits_configured": limits.get(sensor) is not None,
                    "drift_threshold_configured": drift_limits.get(sensor) is not None,
                }
            )

    result["sensor_health_issue_count"] = result[all_flag_columns].sum(axis=1)
    result["sensor_health_status"] = np.where(
        result["sensor_health_issue_count"] > 0,
        "CHECK_SENSOR",
        "INPUT_CHECKS_PASSED",
    )

    # Restore input ordering so callers can safely align rows with their data.
    result = (
        result.sort_values("_original_order")
        .drop(columns=["_original_order"])
        .reset_index(drop=True)
    )

    summary = pd.DataFrame(summary_rows)
    return result, summary


def main() -> None:
    """Run an initial sensor-health audit on the processed dataset."""
    root = Path(__file__).resolve().parents[3]
    data_path = root / "data" / "processed" / "industrial_features.csv"
    report_dir = root / "artifacts" / "reports"

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    checked, summary = monitor_sensor_health(data)

    report_dir.mkdir(parents=True, exist_ok=True)
    row_path = report_dir / "sensor_health_row_results.csv"
    summary_path = report_dir / "sensor_health_summary.csv"

    checked.to_csv(row_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nFORGE // INDUSTRIAL AI")
    print("=" * 60)
    print("Sensor-health audit")
    print(f"Rows checked: {len(checked):,}")
    print(f"Machines: {checked[MACHINE_COLUMN].nunique():,}")
    print(f"Rows flagged for review: {int((checked['sensor_health_issue_count'] > 0).sum()):,}")
    print("\nFAULT COUNTS")
    print(
        summary[
            [
                "machine_id",
                "sensor",
                "missing_count",
                "frozen_count",
                "outside_limits_count",
                "drift_count",
            ]
        ].to_string(index=False)
    )

    print("\nIMPORTANT CONFIGURATION STATUS")
    print("Engineering limits and drift thresholds are not configured by default.")
    print("Set verified values before relying on those two alert types.")
    print("\nSaved reports:")
    print(row_path.relative_to(root))
    print(summary_path.relative_to(root))


if __name__ == "__main__":
    main()
