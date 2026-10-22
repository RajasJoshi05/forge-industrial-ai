import pytest
from pydantic import ValidationError

from industrial_ai.common.config import MachineConfig


def test_valid_machine_configuration():
    config = MachineConfig(
        machine_id="M-001",
        temperature_warning_c=85,
        vibration_warning_mm_s=7.1,
        inspection_interval_minutes=30,
    )

    assert config.machine_id == "M-001"


def test_temperature_must_be_within_configured_range():
    with pytest.raises(ValidationError):
        MachineConfig(
            machine_id="M-001",
            temperature_warning_c=300,
            vibration_warning_mm_s=7.1,
            inspection_interval_minutes=30,
        )


def test_unknown_configuration_fields_are_rejected():
    with pytest.raises(ValidationError):
        MachineConfig(
            machine_id="M-001",
            temperature_warning_c=85,
            vibration_warning_mm_s=7.1,
            inspection_interval_minutes=30,
            unknown_setting=True,
        )
