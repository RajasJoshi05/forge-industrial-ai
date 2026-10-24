from pydantic import BaseModel, ConfigDict, Field


class MachineConfig(BaseModel):
    """Validated configuration for an industrial machine."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    machine_id: str = Field(min_length=1)
    temperature_warning_c: float = Field(gt=0, le=250)
    vibration_warning_mm_s: float = Field(gt=0, le=100)
    inspection_interval_minutes: int = Field(gt=0, le=1440)
