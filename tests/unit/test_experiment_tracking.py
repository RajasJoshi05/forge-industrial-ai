import json

import pytest

from industrial_ai.common.settings import load_settings
from industrial_ai.training.experiment_tracking import record_experiment


def test_experiment_metadata_is_saved(tmp_path):
    path = record_experiment(
        experiment_name="baseline",
        model_name="example_classifier",
        dataset_version="dataset-v1",
        random_seed=42,
        metrics={"accuracy": 0.91, "f1": 0.88},
        output_dir=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["experiment_name"] == "baseline"
    assert payload["model_name"] == "example_classifier"
    assert payload["dataset_version"] == "dataset-v1"
    assert payload["random_seed"] == 42
    assert payload["metrics"]["accuracy"] == 0.91
    assert payload["run_id"]
    assert payload["settings"] is None


def test_empty_experiment_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="experiment_name"):
        record_experiment(
            experiment_name=" ",
            model_name="example_classifier",
            dataset_version="dataset-v1",
            random_seed=42,
            metrics={"accuracy": 0.91},
            output_dir=tmp_path,
        )


def test_non_numeric_metric_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be numeric"):
        record_experiment(
            experiment_name="baseline",
            model_name="example_classifier",
            dataset_version="dataset-v1",
            random_seed=42,
            metrics={"accuracy": "high"},
            output_dir=tmp_path,
        )


def test_experiment_saves_configuration_snapshot(tmp_path):
    settings = load_settings("configs/experiment.yaml")

    path = record_experiment(
        experiment_name=settings.experiment.name,
        model_name="baseline_classifier",
        dataset_version=settings.data.dataset_version,
        random_seed=settings.experiment.random_seed,
        metrics={"accuracy": 0.85},
        output_dir=tmp_path,
        settings=settings,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["settings"]["experiment"]["name"] == "industrial-baseline"
    assert payload["settings"]["experiment"]["random_seed"] == 42
    assert payload["settings"]["training"]["batch_size"] == 32
