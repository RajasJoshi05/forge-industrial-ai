from industrial_ai.maintenance.batch_validation import validate_sensor_batch


def valid_record(timestamp: str = "2026-10-18T09:00:00+05:30") -> dict:
    return {
        "machine_id": "M-001",
        "timestamp": timestamp,
        "temperature_c": 72.5,
        "vibration_mm_s": 2.4,
        "rotational_speed_rpm": 1500,
    }


def test_valid_batch_has_no_errors():
    records = [
        valid_record(),
        valid_record("2026-10-18T09:01:00+05:30"),
    ]

    assert validate_sensor_batch(records) == []


def test_duplicate_machine_timestamp_is_detected():
    records = [valid_record(), valid_record()]

    errors = validate_sensor_batch(records)

    assert any("duplicate machine/timestamp pair" in error for error in errors)


def test_invalid_record_is_reported_with_row_number():
    record = valid_record()
    record["vibration_mm_s"] = -1

    errors = validate_sensor_batch([valid_record(), record])

    assert any(error.startswith("Row 1:") for error in errors)


def test_same_timestamp_for_different_machines_is_allowed():
    first = valid_record()
    second = valid_record()
    second["machine_id"] = "M-002"

    assert validate_sensor_batch([first, second]) == []
