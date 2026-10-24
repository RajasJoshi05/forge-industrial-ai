import pandas as pd
import pytest

from industrial_ai.data.features import build_sensor_features


def make_record(
    machine_id: str,
    timestamp: str,
    temperature: float,
    vibration: float = 2.0,
    speed: float = 1500.0,
) -> dict:
    return {
        "machine_id": machine_id,
        "timestamp": timestamp,
        "temperature_c": temperature,
        "vibration_mm_s": vibration,
        "rotational_speed_rpm": speed,
    }


def test_features_sort_chronologically_and_calculate_rolling_mean() -> None:
    records = [
        make_record("M-001", "2026-10-18T09:02:00+00:00", 70.0),
        make_record("M-001", "2026-10-18T09:00:00+00:00", 60.0),
        make_record("M-001", "2026-10-18T09:01:00+00:00", 65.0),
    ]

    result = build_sensor_features(records, rolling_window=2)

    assert result["timestamp"].is_monotonic_increasing
    assert result["temperature_c_rolling_mean"].tolist() == [60.0, 62.5, 67.5]
    assert pd.isna(result.loc[0, "temperature_c_change"])
    assert result.loc[1, "temperature_c_change"] == 5.0


def test_rolling_features_do_not_mix_machines() -> None:
    records = [
        make_record("M-001", "2026-10-18T09:00:00+00:00", 60.0),
        make_record("M-002", "2026-10-18T09:00:00+00:00", 100.0),
        make_record("M-001", "2026-10-18T09:01:00+00:00", 70.0),
    ]

    result = build_sensor_features(records, rolling_window=2)
    machine_one = result[result["machine_id"] == "M-001"].reset_index(drop=True)

    assert machine_one["temperature_c_rolling_mean"].tolist() == [60.0, 65.0]


def test_duplicate_machine_timestamps_are_rejected() -> None:
    record = make_record("M-001", "2026-10-18T09:00:00+00:00", 60.0)

    with pytest.raises(ValueError, match="Duplicate"):
        build_sensor_features([record, record])


def test_invalid_rolling_window_is_rejected() -> None:
    record = make_record("M-001", "2026-10-18T09:00:00+00:00", 60.0)

    with pytest.raises(ValueError, match="rolling_window"):
        build_sensor_features([record], rolling_window=1)


def test_empty_input_returns_empty_dataframe() -> None:
    result = build_sensor_features([])

    assert isinstance(result, pd.DataFrame)
    assert result.empty
