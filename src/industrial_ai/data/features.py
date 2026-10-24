from __future__ import annotations

import pandas as pd

from industrial_ai.maintenance.contracts import SensorRecord

SENSOR_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
]

TIMESTAMP_COLUMN = "timestamp"


def build_sensor_features(
    records: list[dict],
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Build causal sensor features matching the trained model contract."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")

    if not records:
        return pd.DataFrame()

    validated = [SensorRecord.model_validate(record).model_dump(mode="json") for record in records]
    frame = pd.DataFrame(validated)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN],
        utc=True,
        errors="raise",
    )

    if frame.duplicated(["machine_id", TIMESTAMP_COLUMN]).any():
        raise ValueError("Duplicate machine/timestamp pairs are not allowed")

    frame = frame.sort_values(
        ["machine_id", TIMESTAMP_COLUMN],
        kind="stable",
    ).reset_index(drop=True)

    grouped = frame.groupby("machine_id", sort=False)

    for column in SENSOR_COLUMNS:
        sensor = grouped[column]

        frame[f"{column}_change"] = sensor.diff()

        frame[f"{column}_rolling_mean"] = sensor.transform(
            lambda values: values.rolling(
                window=rolling_window,
                min_periods=1,
            ).mean()
        )

        frame[f"{column}_rolling_std"] = sensor.transform(
            lambda values: values.rolling(
                window=rolling_window,
                min_periods=2,
            ).std()
        )

        frame[f"{column}_diff_1"] = sensor.diff()

        for window in (3, 6, 12):
            frame[f"{column}_rolling_mean_{window}"] = sensor.transform(
                lambda values, window=window: values.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )

        past_mean = sensor.transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=12,
                    min_periods=3,
                )
                .mean()
            )
        )
        frame[f"{column}_deviation_from_past"] = frame[column] - past_mean

    return frame
