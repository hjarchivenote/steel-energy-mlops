"""Command-line entry points shared with future application and orchestration layers."""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.preparation import prepare_dataset


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Steel Energy MLOps data preparation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="Validate data and create chronological feature partitions")
    prepare.add_argument("--input", type=Path, default=Path("data/coursework/Steel_industry_data_5000.csv"))
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    prepare.add_argument("--expected-rows", type=int, default=5000)
    prepare.add_argument("--train-fraction", type=float, default=0.70)
    prepare.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args(arguments)
    try:
        manifest = prepare_dataset(
            args.input, args.output, args.expected_rows, args.train_fraction, args.validation_fraction
        )
    except (DataValidationError, OSError, UnicodeError, csv.Error) as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
