"""Command-line entry points shared with future application and orchestration layers."""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from mlflow.exceptions import MlflowException

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.modeling import LEARNED_FAMILIES
from steel_energy_mlops.prediction import predict_consumption
from steel_energy_mlops.preparation import prepare_dataset
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE
from steel_energy_mlops.training import train_models


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Steel Energy MLOps preparation, training, and prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="Validate data and create chronological feature partitions")
    prepare.add_argument("--input", type=Path, default=Path("data/coursework/Steel_industry_data_5000.csv"))
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    prepare.add_argument("--expected-rows", type=int, default=5000)
    prepare.add_argument("--train-fraction", type=float, default=0.70)
    prepare.add_argument("--validation-fraction", type=float, default=0.15)
    train = commands.add_parser("train", help="Compare train-only CV candidates and apply the validation gate")
    train.add_argument("--data", type=Path, default=Path("data/processed"))
    train.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    train.add_argument("--model-name", default=DEFAULT_MODEL)
    train.add_argument("--families", nargs="+", choices=LEARNED_FAMILIES, default=list(LEARNED_FAMILIES))
    train.add_argument("--forest-trees", type=int, default=100)
    train.add_argument("--minimum-improvement", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict", help="Predict with one pinned, validated champion version")
    predict.add_argument("--timestamp", type=datetime.fromisoformat, action="append", required=True)
    predict.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    predict.add_argument("--model-name", default=DEFAULT_MODEL)
    args = parser.parse_args(arguments)
    try:
        if args.command == "prepare-data":
            result = prepare_dataset(
                args.input, args.output, args.expected_rows, args.train_fraction, args.validation_fraction
            )
        elif args.command == "train":
            result = train_models(args.data, args.workspace, args.model_name, tuple(args.families), args.forest_trees,
                                  args.minimum_improvement, args.seed)
        else:
            result = predict_consumption(tuple(args.timestamp), args.workspace, args.model_name)
    except (DataValidationError, ValueError, OSError, UnicodeError, csv.Error, MlflowException) as exc:
        operations = {"prepare-data": "Data preparation", "train": "Model training", "predict": "Prediction"}
        operation = operations[args.command]
        print(f"{operation} failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
