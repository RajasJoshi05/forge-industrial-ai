from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "predictive_maintenance_logistic_regression.joblib"
)

METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "predictive_maintenance_metadata.json"


class PredictiveMaintenancePredictor:
    """Load a trained model and score engineered sensor observations."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        metadata_path: str | Path = METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")

        self.metadata: dict[str, Any] = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        self.features: list[str] = self.metadata["features"]
        self.target: str = self.metadata["target"]
        self.threshold: float = float(self.metadata["threshold"])

        self.model = joblib.load(self.model_path)

    def predict(self, observation: dict[str, object]) -> dict[str, object]:
        """Return a warning probability and a threshold-based prediction."""
        missing = [feature for feature in self.features if feature not in observation]

        if missing:
            raise ValueError(f"Observation is missing required features: {missing}")

        row = pd.DataFrame(
            [{feature: observation[feature] for feature in self.features}],
            columns=self.features,
        )

        probabilities = self.model.predict_proba(row)[0]
        classes = list(self.model.classes_)
        positive_index = classes.index(1)

        probability = float(probabilities[positive_index])
        predicted_warning = probability >= self.threshold

        return {
            "model_name": self.metadata["model_name"],
            "target": self.target,
            "warning_probability": round(probability, 6),
            "threshold": self.threshold,
            "prediction": ("WARNING" if predicted_warning else "NORMAL"),
            "dataset_type": self.metadata.get("dataset_type", "unknown"),
            "interpretation": (
                "The model score exceeds the configured warning threshold."
                if predicted_warning
                else "The model score is below the configured warning threshold."
            ),
        }
