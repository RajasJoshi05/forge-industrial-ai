import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def save_ingestion_report(
    result: dict[str, list[dict[str, Any]]],
    output_dir: str | Path = "artifacts/reports/ingestion",
) -> Path:
    """Save a JSON summary of accepted and rejected sensor records."""
    accepted = result["accepted"]
    rejected = result["rejected"]

    report = {
        "report_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "total_records": len(accepted) + len(rejected),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejection_details": [
            {
                "row_index": item["row_index"],
                "errors": item["errors"],
            }
            for item in rejected
        ],
    }

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    report_path = directory / f"{report['report_id']}.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report_path
