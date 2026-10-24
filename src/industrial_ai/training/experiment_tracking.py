import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from industrial_ai.common.settings import ProjectSettings


def record_experiment(
    experiment_name: str,
    model_name: str,
    dataset_version: str,
    random_seed: int,
    metrics: dict[str, float],
    output_dir: str | Path = "artifacts/reports/experiments",
    settings: ProjectSettings | None = None,
) -> Path:
    """Save experiment metadata, metrics, and optional validated settings."""
    if not experiment_name.strip():
        raise ValueError("experiment_name must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not dataset_version.strip():
        raise ValueError("dataset_version must not be empty")

    for name, value in metrics.items():
        if not name.strip():
            raise ValueError("Metric names must not be empty")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Metric '{name}' must be numeric")

    payload: dict[str, Any] = {
        "run_id": str(uuid4()),
        "experiment_name": experiment_name,
        "model_name": model_name,
        "dataset_version": dataset_version,
        "random_seed": random_seed,
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump() if settings is not None else None,
    }

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"{payload['run_id']}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_path
