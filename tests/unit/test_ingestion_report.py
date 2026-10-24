import json

from industrial_ai.maintenance.ingestion_report import save_ingestion_report


def test_ingestion_report_counts_accepted_and_rejected(tmp_path):
    result = {
        "accepted": [{"machine_id": "M-001"}],
        "rejected": [
            {
                "row_index": 1,
                "record": {"machine_id": "M-002"},
                "errors": ["Invalid vibration value"],
            }
        ],
    }

    path = save_ingestion_report(result, output_dir=tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["total_records"] == 2
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 1
    assert report["rejection_details"][0]["row_index"] == 1
    assert report["report_id"]


def test_empty_ingestion_report_is_supported(tmp_path):
    path = save_ingestion_report(
        {"accepted": [], "rejected": []},
        output_dir=tmp_path,
    )

    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["total_records"] == 0
    assert report["accepted_count"] == 0
    assert report["rejected_count"] == 0
