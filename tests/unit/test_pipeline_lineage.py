from pathlib import Path

from industrial_ai.data.pipeline import (
    PIPELINE_VERSION,
    calculate_sha256,
    run_sensor_pipeline,
)


def test_sha256_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("industrial-ai", encoding="utf-8")

    first_hash = calculate_sha256(source)
    second_hash = calculate_sha256(source)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_pipeline_report_contains_lineage_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sensors.csv"
    source.write_text(
        "machine_id,timestamp,temperature_c,vibration_mm_s,rotational_speed_rpm\n"
        "M-001,2026-10-18T09:00:00+00:00,60,2.0,1500\n"
        "M-001,2026-10-18T09:01:00+00:00,62,2.2,1500\n",
        encoding="utf-8",
    )

    result = run_sensor_pipeline(
        input_csv=source,
        output_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        rolling_window=2,
    )

    report = result["summary"]

    assert report["run_id"]
    assert report["pipeline_version"] == PIPELINE_VERSION
    assert len(report["input_sha256"]) == 64
    assert len(report["output_sha256"]) == 64
    assert report["feature_rows"] == 2
    assert Path(result["report_json"]).is_file()
