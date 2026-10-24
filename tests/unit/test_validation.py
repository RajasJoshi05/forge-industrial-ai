from industrial_ai.common.validation import validate_required_fields


def test_valid_record_has_no_errors():
    record = {"machine_id": "M-001", "temperature": 72.5}

    assert validate_required_fields(record, {"machine_id", "temperature"}) == []


def test_missing_field_is_reported():
    record = {"machine_id": "M-001"}

    assert validate_required_fields(record, {"machine_id", "temperature"}) == [
        "Missing required field: temperature"
    ]


def test_empty_string_is_reported():
    record = {"machine_id": "   "}

    assert validate_required_fields(record, {"machine_id"}) == [
        "Required field is empty: machine_id"
    ]


def test_none_value_is_reported():
    record = {"machine_id": None}

    assert validate_required_fields(record, {"machine_id"}) == [
        "Required field is None: machine_id"
    ]
