from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    random_seed: int = Field(ge=0)


class DataSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_version: str = Field(min_length=1)


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_fraction: float = Field(gt=0, lt=1)
    batch_size: int = Field(gt=0, le=4096)
    epochs: int = Field(gt=0, le=10000)


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentSettings
    data: DataSettings
    training: TrainingSettings


def load_settings(config_path: str | Path) -> ProjectSettings:
    """Load YAML settings and validate their structure and values."""
    path = Path(config_path)

    with path.open("r", encoding="utf-8-sig") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    return ProjectSettings.model_validate(raw_config)
