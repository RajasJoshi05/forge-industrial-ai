from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from industrial_ai.maintenance.contracts import SensorRecord
from industrial_ai.vision.contracts import InspectionRecord


def test_valid_inspection_record():
    record = InspectionRecord(
        image_id="IMG-001",
        image_path="data/raw/sample.png",
        machine_id="M-001",
        defect_label="surface_scratch",
        inspection_timestamp="2026-10-17T10:00:00Z",
    )

    assert record.image_id == "IMG-001"


def test_inspection_record_rejects_missing_label():
    with pytest.raises(ValidationError):
        InspectionRecord(
            image_id="IMG-001",
            image_path="data/raw/sample.png",
            machine_id="M-001",
            inspection_timestamp="2026-10-17T10:00:00Z",
        )


def test_valid_sensor_record():
    record = SensorRecord(
        machine_id="M-001",
        timestamp=datetime(2026, 10, 17, tzinfo=UTC),
        temperature_c=72.5,
        vibration_mm_s=2.1,
        rotational_speed_rpm=1500,
    )

    assert record.vibration_mm_s == 2.1


def test_sensor_record_rejects_negative_vibration():
    with pytest.raises(ValidationError):
        SensorRecord(
            machine_id="M-001",
            timestamp=datetime(2026, 10, 17, tzinfo=UTC),
            temperature_c=72.5,
            vibration_mm_s=-1,
            rotational_speed_rpm=1500,
        )
