from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SENSORS = [
    "temperature_c",
    "vibration_mm_s",
    "rotational_speed_rpm",
]

DEFAULT_CONTAMINATION = 0.02
RANDOM_STATE = 42


def _validate_sensor_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and return numeric sensor features."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must contain at least one row.")

    missing = sorted(set(SENSORS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")

    features = data[SENSORS].apply(pd.to_numeric, errors="coerce")

    if features.isna().any().any():
        raise ValueError("Sensor features cannot contain missing or non-numeric values.")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Sensor features must contain only finite values.")

    return features.astype(float)


def build_anomaly_detector(
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create a reproducible sensor anomaly-detection pipeline."""
    if not 0.0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    contamination=contamination,
                    random_state=random_state,
                    n_estimators=200,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_anomaly_detector(
    training_data: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Fit using training-period sensor data only."""
    features = _validate_sensor_data(training_data)

    if len(features) < 10:
        raise ValueError("At least 10 training rows are required.")

    if (features.nunique() < 2).any():
        raise ValueError("Every sensor must vary in the training data.")

    model = build_anomaly_detector(
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(features)
    return model


def predict_sensor_anomalies(
    model: Pipeline,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return anomaly labels and scores without modifying the input."""
    features = _validate_sensor_data(data)

    labels = model.predict(features)
    scores = model.decision_function(features)

    result = data.copy()
    result["anomaly_label"] = np.where(labels == -1, "ANOMALY", "NORMAL")
    result["anomaly_score"] = -scores
    result["is_anomaly"] = labels == -1

    return result


def save_anomaly_detector(
    model: Pipeline,
    model_path: str | Path,
) -> Path:
    """Save a fitted model and basic artifact metadata."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "sensor_columns": SENSORS,
            "contamination": model.named_steps["detector"].contamination,
            "random_state": RANDOM_STATE,
        },
        path,
    )
    return path


def load_anomaly_detector(
    model_path: str | Path,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load an anomaly model artifact and validate its schema."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Anomaly model not found: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Invalid anomaly model artifact.")

    if artifact.get("sensor_columns") != SENSORS:
        raise ValueError("Model sensor schema does not match the current schema.")

    model = artifact["model"]
    if not isinstance(model, Pipeline):
        raise ValueError("Artifact does not contain a scikit-learn Pipeline.")

    metadata = {key: value for key, value in artifact.items() if key != "model"}
    return model, metadata
