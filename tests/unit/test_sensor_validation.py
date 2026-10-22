from datetime import datetime

import pytest
from pydantic import ValidationError

from industrial_ai.maintenance.contracts import SensorRecord
from industrial_ai.maintenance.validation import validate_sensor_record


def valid_sensor_record() -> dict:
    return {
        "machine_id": "M-001",
        "timestamp": "2026-10-18T09:00:00+05:30",
        "temperature_c": 72.5,
        "vibration_mm_s": 2.4,
        "rotational_speed_rpm": 1500,
    }


def test_valid_sensor_record_is_accepted():
    assert validate_sensor_record(valid_sensor_record()) == []


def test_negative_vibration_is_rejected():
    record = valid_sensor_record()
    record["vibration_mm_s"] = -1

    assert validate_sensor_record(record)


def test_missing_machine_id_is_rejected():
    record = valid_sensor_record()
    del record["machine_id"]

    assert validate_sensor_record(record)


def test_unknown_sensor_field_is_rejected():
    record = valid_sensor_record()
    record["unexpected"] = True

    assert validate_sensor_record(record)


@pytest.mark.parametrize(
    "field",
    [
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_sensor_values_are_rejected(field, value):
    record = valid_sensor_record()
    record[field] = value

    with pytest.raises(ValidationError):
        SensorRecord.model_validate(record)


def test_naive_timestamp_is_rejected():
    record = valid_sensor_record()
    record["timestamp"] = datetime(2026, 10, 18, 9, 0)

    with pytest.raises(ValidationError, match="timezone"):
        SensorRecord.model_validate(record)


def test_timezone_aware_timestamp_is_normalized_to_utc():
    sensor = SensorRecord.model_validate(valid_sensor_record())

    assert sensor.timestamp.utcoffset().total_seconds() == 0
