from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "industrial_sensor_dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_features.csv"

SENSOR_COLUMNS = (
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
)
GROUP_COLUMNS = ("machine_id",)
ROLLING_WINDOWS = (3, 6, 12)
DEVIATION_WINDOW = 12


def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Create historical sensor features separately for each machine."""
    required_columns = [
        *GROUP_COLUMNS,
        "timestamp",
        *SENSOR_COLUMNS,
        "failure_within_next_30min",
    ]
    missing_columns = sorted(set(required_columns) - set(data.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(
        [*GROUP_COLUMNS, "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    for column in SENSOR_COLUMNS:
        grouped = data.groupby(list(GROUP_COLUMNS), sort=False)[column]

        data[f"{column}_diff_1"] = grouped.diff()

        for window in ROLLING_WINDOWS:
            data[f"{column}_rolling_mean_{window}"] = grouped.transform(
                lambda series, window=window: series.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )

        data[f"{column}_deviation_from_past"] = grouped.transform(
            lambda series: (
                series
                - series.shift(1)
                .rolling(
                    window=DEVIATION_WINDOW,
                    min_periods=3,
                )
                .mean()
            )
        )

    return data


def main() -> None:
    """Prepare and save the processed industrial sensor dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")

    data = pd.read_csv(RAW_DATA_PATH)
    processed_data = prepare_dataset(data)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)

    print("FORGE // INDUSTRIAL AI")
    print("=" * 48)
    print(f"Raw dataset:       {RAW_DATA_PATH}")
    print(f"Processed dataset: {PROCESSED_DATA_PATH}")
    print(f"Rows:              {len(processed_data):,}")
    print(f"Columns:           {len(processed_data.columns):,}")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
