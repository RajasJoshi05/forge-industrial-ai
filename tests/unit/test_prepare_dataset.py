import pandas as pd
import pytest

from industrial_ai.training.prepare_dataset import prepare_dataset


def make_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": ["M-001"] * 7 + ["M-002"] * 2,
            "timestamp": [
                "2026-10-18T08:45:00Z",
                "2026-10-18T08:50:00Z",
                "2026-10-18T08:55:00Z",
                "2026-10-18T09:00:00Z",
                "2026-10-18T09:05:00Z",
                "2026-10-18T09:10:00Z",
                "2026-10-18T09:15:00Z",
                "2026-10-18T09:00:00Z",
                "2026-10-18T09:05:00Z",
            ],
            "temperature_c": [45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 100.0, 110.0],
            "vibration_mm_s": [1.0, 1.2, 1.4, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0],
            "rotational_speed_rpm": [
                1520.0,
                1510.0,
                1505.0,
                1500.0,
                1490.0,
                1480.0,
                1470.0,
                1600.0,
                1580.0,
            ],
            "failure_within_next_30min": [0, 0, 0, 0, 0, 1, 1, 0, 1],
        }
    )


def test_features_use_only_current_and_past_observations() -> None:
    result = prepare_dataset(make_training_data())
    machine_one = result[result["machine_id"] == "M-001"].reset_index(drop=True)

    assert machine_one["temperature_c_rolling_mean_3"].tolist() == [
        45.0,
        47.5,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
    ]
    assert pd.isna(machine_one.loc[0, "temperature_c_diff_1"])
    assert machine_one.loc[1, "temperature_c_diff_1"] == 5.0
    assert machine_one.loc[2, "temperature_c_diff_1"] == 5.0


def test_features_do_not_mix_machines() -> None:
    result = prepare_dataset(make_training_data())
    machine_two = result[result["machine_id"] == "M-002"].reset_index(drop=True)

    assert machine_two["temperature_c_rolling_mean_3"].tolist() == [
        100.0,
        105.0,
    ]
    assert pd.isna(machine_two.loc[0, "temperature_c_diff_1"])


def test_future_sensor_changes_do_not_change_earlier_features() -> None:
    data = make_training_data()
    original = prepare_dataset(data)

    changed_data = data.copy()
    future_mask = (changed_data["machine_id"] == "M-001") & (
        changed_data["timestamp"] == "2026-10-18T09:15:00Z"
    )
    changed_data.loc[future_mask, "temperature_c"] = 999.0
    changed = prepare_dataset(changed_data)

    target_mask = (original["machine_id"] == "M-001") & (
        original["timestamp"] == "2026-10-18T09:10:00Z"
    )
    original_row = original.loc[target_mask].iloc[0]
    changed_row = changed.loc[target_mask].iloc[0]

    assert (
        original_row["temperature_c_rolling_mean_3"] == changed_row["temperature_c_rolling_mean_3"]
    )
    assert pd.notna(original_row["temperature_c_deviation_from_past"])
    assert pd.notna(changed_row["temperature_c_deviation_from_past"])
    assert (
        original_row["temperature_c_deviation_from_past"]
        == changed_row["temperature_c_deviation_from_past"]
    )


def test_missing_required_columns_are_rejected() -> None:
    data = make_training_data().drop(columns=["temperature_c"])

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_dataset(data)
