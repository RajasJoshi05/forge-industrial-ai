import math
from datetime import UTC, datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class SensorRecord(BaseModel):
    """One timestamped sensor observation from an industrial machine."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    machine_id: str = Field(min_length=1)
    timestamp: datetime
    temperature_c: float
    vibration_mm_s: float = Field(ge=0)
    rotational_speed_rpm: float = Field(ge=0)

    @field_validator("temperature_c", "vibration_mm_s", "rotational_speed_rpm")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        """Reject NaN and infinite sensor readings."""
        if not math.isfinite(value):
            raise ValueError("Sensor values must be finite")
        return value

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        """Require timestamps with an explicit timezone."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp must include a timezone")
        return value.astimezone(UTC)
