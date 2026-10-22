import pytest
from pydantic import ValidationError

from industrial_ai.common.settings import (
    ProjectSettings,
    load_settings,
)


def test_experiment_yaml_loads_correctly():
    settings = load_settings("configs/experiment.yaml")

    assert settings.experiment.name == "industrial-baseline"
    assert settings.experiment.random_seed == 42
    assert settings.data.dataset_version == "local-v1"
    assert settings.training.validation_fraction == 0.2
    assert settings.training.batch_size == 32
    assert settings.training.epochs == 10


def test_invalid_validation_fraction_is_rejected():
    with pytest.raises(ValidationError):
        ProjectSettings.model_validate(
            {
                "experiment": {"name": "test", "random_seed": 42},
                "data": {"dataset_version": "v1"},
                "training": {
                    "validation_fraction": 1.0,
                    "batch_size": 32,
                    "epochs": 10,
                },
            }
        )


def test_unknown_configuration_fields_are_rejected():
    with pytest.raises(ValidationError):
        ProjectSettings.model_validate(
            {
                "experiment": {
                    "name": "test",
                    "random_seed": 42,
                    "unexpected": True,
                },
                "data": {"dataset_version": "v1"},
                "training": {
                    "validation_fraction": 0.2,
                    "batch_size": 32,
                    "epochs": 10,
                },
            }
        )


def test_missing_configuration_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")
