import json

import pytest

import industrial_ai.maintenance.predictor as predictor_module
from industrial_ai.maintenance.predictor import PredictiveMaintenancePredictor


class FakeModel:
    """Small deterministic model for testing predictor decision logic."""

    def __init__(self, classes, probabilities):
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, rows):
        return [self.probabilities]


def build_predictor(tmp_path, monkeypatch, probabilities, classes=(0, 1)):
    """Create a predictor with temporary metadata and a mocked model."""
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    model_path.write_bytes(b"test model placeholder")
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "test_model",
                "target": "failure_within_next_30min",
                "features": ["temperature_c", "vibration_mm_s"],
                "threshold": 0.95,
                "dataset_type": "synthetic",
            }
        ),
        encoding="utf-8",
    )

    fake_model = FakeModel(classes, probabilities)
    monkeypatch.setattr(
        predictor_module.joblib,
        "load",
        lambda _path: fake_model,
    )

    return PredictiveMaintenancePredictor(
        model_path=model_path,
        metadata_path=metadata_path,
    )


def test_probability_equal_to_threshold_produces_warning(tmp_path, monkeypatch):
    predictor = build_predictor(
        tmp_path,
        monkeypatch,
        probabilities=[0.05, 0.95],
    )

    result = predictor.predict({"temperature_c": 90.0, "vibration_mm_s": 8.0})

    assert result["warning_probability"] == 0.95
    assert result["threshold"] == 0.95
    assert result["prediction"] == "WARNING"


def test_probability_below_threshold_produces_normal(tmp_path, monkeypatch):
    predictor = build_predictor(
        tmp_path,
        monkeypatch,
        probabilities=[0.050001, 0.949999],
    )

    result = predictor.predict({"temperature_c": 90.0, "vibration_mm_s": 8.0})

    assert result["warning_probability"] == 0.949999
    assert result["prediction"] == "NORMAL"


def test_missing_required_features_raise_value_error(tmp_path, monkeypatch):
    predictor = build_predictor(
        tmp_path,
        monkeypatch,
        probabilities=[0.1, 0.9],
    )

    with pytest.raises(ValueError, match="vibration_mm_s"):
        predictor.predict({"temperature_c": 90.0})


def test_probability_for_class_one_is_used_when_classes_are_reversed(
    tmp_path,
    monkeypatch,
):
    predictor = build_predictor(
        tmp_path,
        monkeypatch,
        probabilities=[0.96, 0.04],
        classes=[1, 0],
    )

    result = predictor.predict({"temperature_c": 90.0, "vibration_mm_s": 8.0})

    assert result["warning_probability"] == 0.96
    assert result["prediction"] == "WARNING"
