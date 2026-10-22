import argparse
import logging
import sys

from industrial_ai.data.pipeline import run_sensor_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate industrial sensor CSV data and build features."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source sensor CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed datasets.",
    )
    parser.add_argument(
        "--report-dir",
        default="artifacts/reports/pipeline",
        help="Directory for pipeline and ingestion reports.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Number of observations used for rolling features (minimum 2).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the sensor pipeline from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.rolling_window < 2:
        parser.error("--rolling-window must be at least 2")

    try:
        result = run_sensor_pipeline(
            input_csv=args.input,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            rolling_window=args.rolling_window,
        )
    except (OSError, ValueError) as exc:
        logging.error("Sensor pipeline failed: %s", exc)
        return 1

    summary = result["summary"]
    print("Sensor pipeline completed successfully.")
    print(f"Input records: {summary['total_records']}")
    print(f"Accepted: {summary['accepted_records']}")
    print(f"Rejected: {summary['rejected_records']}")
    print(f"Processed dataset: {result['output_csv']}")
    print(f"Pipeline report: {result['report_json']}")
    print(f"Ingestion report: {result['ingestion_report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
