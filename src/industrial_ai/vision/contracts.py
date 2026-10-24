from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class InspectionRecord(BaseModel):
    """Metadata describing one industrial visual inspection."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    image_id: str = Field(min_length=1)
    image_path: Path
    machine_id: str = Field(min_length=1)
    defect_label: str = Field(min_length=1)
    inspection_timestamp: str = Field(min_length=1)
