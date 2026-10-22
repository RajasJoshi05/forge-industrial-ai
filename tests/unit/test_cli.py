from pathlib import Path

import pytest

from industrial_ai.data import cli


def test_cli_help_displays_supported_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--input" in output
    assert "--output-dir" in output
    assert "--report-dir" in output
    assert "--rolling-window" in output


def test_cli_rejects_rolling_window_below_two() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--input", "sensors.csv", "--rolling-window", "1"])

    assert exc_info.value.code == 2


def test_cli_returns_failure_for_missing_input(tmp_path: Path) -> None:
    exit_code = cli.main(["--input", str(tmp_path / "missing.csv")])

    assert exit_code == 1


def test_cli_successfully_runs_pipeline(monkeypatch, capsys, tmp_path: Path) -> None:
    output_csv = tmp_path / "processed.csv"
    pipeline_report = tmp_path / "pipeline_report.json"
    ingestion_report = tmp_path / "ingestion_report.json"

    def fake_pipeline(**kwargs):
        assert kwargs["rolling_window"] == 3
        return {
            "output_csv": output_csv,
            "report_json": pipeline_report,
            "ingestion_report": ingestion_report,
            "summary": {
                "total_records": 5,
                "accepted_records": 4,
                "rejected_records": 1,
            },
        }

    monkeypatch.setattr(cli, "run_sensor_pipeline", fake_pipeline)

    exit_code = cli.main(
        [
            "--input",
            "sensors.csv",
            "--rolling-window",
            "3",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Sensor pipeline completed successfully." in output
    assert "Accepted: 4" in output
    assert "Rejected: 1" in output
