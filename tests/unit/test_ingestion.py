from industrial_ai.maintenance.ingestion import ingest_sensor_batch


def make_record(machine_id: str = "M-001", vibration: float = 2.4) -> dict:
    return {
        "machine_id": machine_id,
        "timestamp": "2026-10-18T09:00:00+05:30",
        "temperature_c": 72.5,
        "vibration_mm_s": vibration,
        "rotational_speed_rpm": 1500,
    }


def test_valid_record_is_accepted():
    result = ingest_sensor_batch([make_record()])

    assert len(result["accepted"]) == 1
    assert result["rejected"] == []


def test_invalid_record_is_rejected_with_reason():
    result = ingest_sensor_batch([make_record(vibration=-1)])

    assert result["accepted"] == []
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["row_index"] == 0
    assert result["rejected"][0]["errors"]


def test_all_duplicate_records_are_rejected_with_audit_details():
    record = make_record()
    result = ingest_sensor_batch([record, record])

    assert result["accepted"] == []
    assert len(result["rejected"]) == 2

    for rejected_record in result["rejected"]:
        assert any(
            "duplicate machine/timestamp pair" in error for error in rejected_record["errors"]
        )


def test_empty_batch_returns_empty_results():
    assert ingest_sensor_batch([]) == {"accepted": [], "rejected": []}
