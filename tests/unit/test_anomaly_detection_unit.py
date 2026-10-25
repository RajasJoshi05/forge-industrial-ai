from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from industrial_ai.monitoring.anomaly_detection import (
    SENSORS,
    fit_anomaly_detector,
    load_anomaly_detector,
    predict_sensor_anomalies,
    save_anomaly_detector,
)


@pytest.fixture
def training_data() -> pd.DataFrame:
    """Create reproducible synthetic sensor readings."""
    rng = np.random.default_rng(42)
    rows = 200

    return pd.DataFrame(
        {
            "temperature_c": rng.normal(70, 3, rows),
            "vibration_mm_s": rng.normal(2.5, 0.3, rows),
            "rotational_speed_rpm": rng.normal(1500, 40, rows),
        }
    )


def test_fit_returns_pipeline(training_data: pd.DataFrame) -> None:
    model = fit_anomaly_detector(training_data)

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "detector" in model.named_steps


def test_predictions_have_expected_columns(
    training_data: pd.DataFrame,
) -> None:
    model = fit_anomaly_detector(training_data)
    results = predict_sensor_anomalies(model, training_data.head(10))

    assert len(results) == 10
    assert {"anomaly_label", "anomaly_score", "is_anomaly"}.issubset(results.columns)
    assert set(results["anomaly_label"].unique()).issubset({"NORMAL", "ANOMALY"})
    assert results["is_anomaly"].dtype == bool
    assert np.isfinite(results["anomaly_score"]).all()


def test_extreme_reading_is_flagged(training_data: pd.DataFrame) -> None:
    model = fit_anomaly_detector(training_data)

    unusual = pd.DataFrame(
        {
            "temperature_c": [180.0],
            "vibration_mm_s": [18.0],
            "rotational_speed_rpm": [3500.0],
        }
    )

    result = predict_sensor_anomalies(model, unusual)

    assert bool(result.iloc[0]["is_anomaly"])


@pytest.mark.parametrize(
    "invalid_data",
    [
        pd.DataFrame(
            {
                "temperature_c": [70.0],
                "vibration_mm_s": [np.nan],
                "rotational_speed_rpm": [1500.0],
            }
        ),
        pd.DataFrame(
            {
                "temperature_c": [70.0],
                "vibration_mm_s": [np.inf],
                "rotational_speed_rpm": [1500.0],
            }
        ),
        pd.DataFrame(
            {
                "temperature_c": ["invalid"],
                "vibration_mm_s": [2.5],
                "rotational_speed_rpm": [1500.0],
            }
        ),
    ],
)
def test_invalid_sensor_values_are_rejected(
    invalid_data: pd.DataFrame,
    training_data: pd.DataFrame,
) -> None:
    model = fit_anomaly_detector(training_data)

    with pytest.raises(ValueError):
        predict_sensor_anomalies(model, invalid_data)


def test_missing_sensor_column_is_rejected(
    training_data: pd.DataFrame,
) -> None:
    model = fit_anomaly_detector(training_data)
    incomplete = training_data.drop(columns=[SENSORS[0]])

    with pytest.raises(ValueError, match="Missing required sensor columns"):
        predict_sensor_anomalies(model, incomplete)


def test_training_requires_at_least_ten_rows(
    training_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="At least 10 training rows"):
        fit_anomaly_detector(training_data.head(9))


def test_training_rejects_constant_sensor(training_data: pd.DataFrame) -> None:
    training_data["temperature_c"] = 70.0

    with pytest.raises(ValueError, match="Every sensor must vary"):
        fit_anomaly_detector(training_data)


def test_save_and_load_preserves_predictions(
    training_data: pd.DataFrame,
    tmp_path,
) -> None:
    model = fit_anomaly_detector(training_data)
    sample = training_data.head(20)

    before = predict_sensor_anomalies(model, sample)
    model_path = tmp_path / "models" / "anomaly_detector.joblib"

    saved_path = save_anomaly_detector(model, model_path)
    loaded_model, metadata = load_anomaly_detector(saved_path)
    after = predict_sensor_anomalies(loaded_model, sample)

    assert saved_path.is_file()
    assert metadata["sensor_columns"] == SENSORS
    assert before["anomaly_label"].tolist() == after["anomaly_label"].tolist()
    np.testing.assert_allclose(
        before["anomaly_score"].to_numpy(),
        after["anomaly_score"].to_numpy(),
    )


def test_load_missing_model_raises_error(tmp_path) -> None:
    missing_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Anomaly model not found"):
        load_anomaly_detector(missing_path)
