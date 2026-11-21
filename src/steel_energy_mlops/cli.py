"""Command-line entry points shared with future application and orchestration layers."""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from mlflow.exceptions import MlflowException
import uvicorn

from steel_energy_mlops.api import create_app
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.evaluation import evaluate_champion
from steel_energy_mlops.input_audit import audit_inputs
from steel_energy_mlops.modeling import LEARNED_FAMILIES
from steel_energy_mlops.monitoring import monitor_predictions
from steel_energy_mlops.prediction import predict_consumption
from steel_energy_mlops.preparation import prepare_dataset
from steel_energy_mlops.promotion import MINIMUM_IMPROVEMENT
from steel_energy_mlops.runtime import export_champion, initialize_runtime
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE
from steel_energy_mlops.training import train_models


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return port


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Steel Energy MLOps preparation, training, prediction and diagnostics")
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
    train.add_argument("--minimum-improvement", type=float, default=MINIMUM_IMPROVEMENT,
                       help="Required relative MAE improvement; at least 0.01 (1%%), and less than 1")
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict", help="Predict with one pinned, validated champion version")
    predict.add_argument("--timestamp", type=datetime.fromisoformat, action="append", required=True)
    predict.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    predict.add_argument("--model-name", default=DEFAULT_MODEL)
    monitor = commands.add_parser("monitor", help="Report version-scoped request and prediction distribution changes")
    monitor.add_argument("--data", type=Path, default=Path("data/processed"))
    monitor.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    monitor.add_argument("--model-name", default=DEFAULT_MODEL)
    monitor.add_argument("--lookback-hours", type=int, default=168)
    monitor.add_argument("--max-runs", type=int, default=100)
    evaluate = commands.add_parser("evaluate", help="Compare the pinned champion with schedule references")
    evaluate.add_argument("--data", type=Path, default=Path("data/processed"))
    evaluate.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    evaluate.add_argument("--model-name", default=DEFAULT_MODEL)
    audit = commands.add_parser("audit-inputs", help="Audit training inputs and a causal 15-minute forecast contract")
    audit.add_argument("--input", type=Path, default=Path("data/coursework/Steel_industry_data_5000.csv"))
    audit.add_argument("--data", type=Path, default=Path("data/processed"))
    audit.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    audit.add_argument("--reporting-delay-minutes", type=int, required=True,
                       help="Explicit assumed measurement reporting delay for the offline audit")
    serve = commands.add_parser("serve", help="Run the local prediction HTTP service and interactive API docs")
    serve.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    serve.add_argument("--model-name", default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8000)
    serve.add_argument("--data", type=Path, default=Path("data/processed"))
    export = commands.add_parser("export-model", help="Export the existing champion as a portable model seed")
    export.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    export.add_argument("--model-name", default=DEFAULT_MODEL)
    export.add_argument("--output", type=Path, default=Path("artifacts/deployment/champion"))
    initialize = commands.add_parser("init-runtime", help="Copy prepared data and restore an existing model seed")
    initialize.add_argument("--prepared-data", type=Path, required=True)
    initialize.add_argument("--data", type=Path, required=True)
    initialize.add_argument("--workspace", type=Path, required=True)
    initialize.add_argument("--model-seed", type=Path, required=True)
    initialize.add_argument("--model-name", default=DEFAULT_MODEL)
    initialize.add_argument("--expected-rows", type=int, default=5000)
    args = parser.parse_args(arguments)
    try:
        if args.command == "prepare-data":
            result = prepare_dataset(
                args.input, args.output, args.expected_rows, args.train_fraction, args.validation_fraction
            )
        elif args.command == "train":
            result = train_models(args.data, args.workspace, args.model_name, tuple(args.families), args.forest_trees,
                                  args.minimum_improvement, args.seed)
        elif args.command == "evaluate":
            result = evaluate_champion(args.data, args.workspace, args.model_name)
        elif args.command == "monitor":
            result = monitor_predictions(args.data, args.workspace, args.model_name, args.lookback_hours, args.max_runs)
        elif args.command == "audit-inputs":
            result = audit_inputs(args.input, args.data, args.reporting_delay_minutes, args.workspace)
        elif args.command == "serve":
            uvicorn.run(create_app(args.workspace, args.model_name, args.data),
                        host=args.host, port=args.port, workers=1)
            return 0
        elif args.command == "export-model":
            result = export_champion(args.workspace, args.output, args.model_name)
        elif args.command == "init-runtime":
            result = initialize_runtime(args.prepared_data, args.data, args.workspace, args.model_seed,
                                        args.model_name, args.expected_rows)
        else:
            result = predict_consumption(tuple(args.timestamp), args.workspace, args.model_name)
    except (DataValidationError, ValueError, OSError, UnicodeError, csv.Error, MlflowException) as exc:
        operations = {"prepare-data": "Data preparation", "train": "Model training", "predict": "Prediction",
                      "evaluate": "Validation diagnostics", "serve": "API startup",
                      "export-model": "Model export", "init-runtime": "Runtime initialization",
                      "monitor": "Prediction monitoring",
                      "audit-inputs": "Input audit"}
        operation = operations[args.command]
        print(f"{operation} failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
