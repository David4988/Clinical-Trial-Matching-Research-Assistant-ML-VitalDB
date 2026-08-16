"""CLI entry point for the VitalDB metadata audit."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vitaldb_audit.run",
        description="Metadata-only reconnaissance of the VitalDB open dataset. "
                    "Downloads no signal data and trains no model.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download raw metadata even if cached copies exist.",
    )
    parser.add_argument(
        "--probe-signals",
        action="store_true",
        help="Measure observed sampling rates for the selected candidate cases. "
             "Reads at most 200 KB per track. Off by default.",
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=None,
        help="Number of candidate cases to select (default: 8, valid 5-10).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args and run the pipeline. Placeholder until Task 8."""
    args = build_parser().parse_args(argv)
    print(f"VitalDB audit CLI (args: {args})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
