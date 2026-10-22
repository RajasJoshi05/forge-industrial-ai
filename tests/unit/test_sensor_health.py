import numpy as np
import pandas as pd
import pytest

from industrial_ai.monitoring.sensor_health import monitor_sensor_health


def make_sensor_data(rows: int = 24) -> pd.DataFrame:
    """Create varying sensor readings with no intentional faults."""
    index = np.arange(rows)

    return pd.DataFrame(
        {
            "machine_id": ["M-001"] * rows,
            "timestamp": pd.date_range(
                "2026-09-01 00:00:00",
                periods=rows,
                freq="5min",
                tz="UTC",
            ),
            "temperature_c": 40.0 + np.sin(index * 0.7),
            "vibration_mm_s": 2.0 + 0.2 * np.sin(index * 1.3),
            "rotational_speed_rpm": 1500.0 + 20.0 * np.sin(index * 0.9),
        }
    )


def test_clean_data_has_no_sensor_health_flags():
    data = make_sensor_data()

    checked, summary = monitor_sensor_health(data)

    assert len(checked) == len(data)
    assert (checked["sensor_health_issue_count"] == 0).all()
    assert (checked["sensor_health_status"] == "INPUT_CHECKS_PASSED").all()
    assert summary["missing_count"].sum() == 0
    assert summary["frozen_count"].sum() == 0
    assert summary["outside_limits_count"].sum() == 0
    assert summary["drift_count"].sum() == 0


def test_missing_reading_is_flagged():
    data = make_sensor_data()
    data.loc[5, "temperature_c"] = np.nan

    checked, summary = monitor_sensor_health(data)

    assert bool(checked.loc[5, "temperature_c_missing"])
    assert checked.loc[5, "sensor_health_status"] == "CHECK_SENSOR"

    temperature_summary = summary[summary["sensor"] == "temperature_c"]
    assert temperature_summary["missing_count"].sum() == 1


def test_frozen_sensor_is_detected():
    data = make_sensor_data()
    data.loc[4:15, "temperature_c"] = 55.0

    checked, summary = monitor_sensor_health(data, frozen_window=12)

    assert checked["temperature_c_frozen"].any()

    temperature_summary = summary[summary["sensor"] == "temperature_c"]
    assert temperature_summary["frozen_count"].sum() > 0


def test_verified_engineering_limits_flag_abnormal_reading():
    data = make_sensor_data()
    data.loc[15, "temperature_c"] = 200.0

    limits = {
        "temperature_c": (0.0, 100.0),
        "vibration_mm_s": (0.0, 10.0),
        "rotational_speed_rpm": (500.0, 3000.0),
    }

    checked, summary = monitor_sensor_health(
        data,
        engineering_limits=limits,
    )

    assert bool(checked.loc[15, "temperature_c_outside_limits"])
    assert checked.loc[15, "sensor_health_status"] == "CHECK_SENSOR"

    temperature_summary = summary[summary["sensor"] == "temperature_c"]
    assert temperature_summary["outside_limits_count"].sum() == 1
    assert temperature_summary["engineering_limits_configured"].all()


def test_configured_drift_threshold_detects_large_change():
    data = make_sensor_data()
    data.loc[18, "temperature_c"] = 90.0

    thresholds = {
        "temperature_c": 5.0,
        "vibration_mm_s": 1.0,
        "rotational_speed_rpm": 100.0,
    }

    checked, summary = monitor_sensor_health(
        data,
        drift_thresholds=thresholds,
    )

    assert bool(checked.loc[18, "temperature_c_drift"])

    temperature_summary = summary[summary["sensor"] == "temperature_c"]
    assert temperature_summary["drift_count"].sum() >= 1
    assert temperature_summary["drift_threshold_configured"].all()


def test_input_row_order_is_preserved():
    data = make_sensor_data()
    data = data.iloc[[5, 0, 9, 2, 12, 1]].reset_index(drop=True)

    checked, _ = monitor_sensor_health(data)

    pd.testing.assert_series_equal(
        checked["timestamp"],
        data["timestamp"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        checked["machine_id"],
        data["machine_id"],
        check_names=False,
    )


def test_missing_required_column_raises_error():
    data = make_sensor_data().drop(columns=["vibration_mm_s"])

    with pytest.raises(ValueError, match="Missing required columns"):
        monitor_sensor_health(data)


@pytest.mark.parametrize(
    ("frozen_window", "drift_window"),
    [
        (1, 12),
        (12, 1),
        (0, 0),
    ],
)
def test_invalid_window_lengths_raise_error(frozen_window, drift_window):
    data = make_sensor_data()

    with pytest.raises(ValueError, match="Window lengths"):
        monitor_sensor_health(
            data,
            frozen_window=frozen_window,
            drift_window=drift_window,
        )


def test_invalid_engineering_limits_raise_error():
    data = make_sensor_data()

    with pytest.raises(ValueError, match="Invalid engineering limits"):
        monitor_sensor_health(
            data,
            engineering_limits={"temperature_c": (100.0, 0.0)},
        )


def test_invalid_drift_threshold_raises_error():
    data = make_sensor_data()

    with pytest.raises(ValueError, match="must be positive"):
        monitor_sensor_health(
            data,
            drift_thresholds={"temperature_c": 0.0},
        )
