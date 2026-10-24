import numpy as np
import pandas as pd
import pytest

from industrial_ai.monitoring.sensor_health import (
    SENSORS,
    monitor_sensor_health,
)


def make_sensor_data(rows: int = 30) -> pd.DataFrame:
    """Create deterministic, healthy synthetic sensor readings."""
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="5min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "machine_id": ["M001"] * rows,
            "timestamp": timestamps,
            "temperature_c": np.linspace(60.0, 62.0, rows),
            "vibration_mm_s": np.linspace(2.0, 2.2, rows),
            "rotational_speed_rpm": np.linspace(1500.0, 1510.0, rows),
        }
    )


@pytest.mark.parametrize("sensor", SENSORS)
def test_missing_sensor_reading_is_detected(sensor: str) -> None:
    data = make_sensor_data()
    data.loc[15, sensor] = np.nan

    checked, _ = monitor_sensor_health(data)

    assert checked.loc[15, f"{sensor}_missing"]
    assert checked.loc[15, "sensor_health_status"] == "CHECK_SENSOR"


@pytest.mark.parametrize("sensor", SENSORS)
def test_frozen_sensor_readings_are_detected(sensor: str) -> None:
    data = make_sensor_data()

    # Twelve consecutive identical readings should trigger detection.
    data.loc[10:21, sensor] = 100.0

    checked, _ = monitor_sensor_health(data, frozen_window=12)

    assert checked.loc[21, f"{sensor}_frozen"]


@pytest.mark.parametrize(
    ("sensor", "limits", "abnormal_value"),
    [
        ("temperature_c", (0.0, 100.0), 150.0),
        ("vibration_mm_s", (0.0, 10.0), 15.0),
        ("rotational_speed_rpm", (500.0, 3000.0), 4000.0),
    ],
)
def test_engineering_limit_violations_are_detected(
    sensor: str,
    limits: tuple[float, float],
    abnormal_value: float,
) -> None:
    data = make_sensor_data()
    data.loc[20, sensor] = abnormal_value

    checked, _ = monitor_sensor_health(
        data,
        engineering_limits={sensor: limits},
    )

    assert checked.loc[20, f"{sensor}_outside_limits"]


@pytest.mark.parametrize(
    ("sensor", "threshold", "abnormal_value"),
    [
        ("temperature_c", 5.0, 100.0),
        ("vibration_mm_s", 1.0, 10.0),
        ("rotational_speed_rpm", 50.0, 2000.0),
    ],
)
def test_sensor_drift_is_detected(
    sensor: str,
    threshold: float,
    abnormal_value: float,
) -> None:
    data = make_sensor_data()
    data.loc[20, sensor] = abnormal_value

    checked, _ = monitor_sensor_health(
        data,
        drift_thresholds={sensor: threshold},
    )

    assert checked.loc[20, f"{sensor}_drift"]
