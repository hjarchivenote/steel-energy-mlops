"""Replay the frozen history experiment; never select models, score final test, or deploy."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import SOURCE_COLUMNS, DataValidationError, _parse_record
from steel_energy_mlops.forecast_inputs import AvailableObservation, build_forecast_inputs
from steel_energy_mlops.tracking import open_store
from steel_energy_mlops.training_data import load_training_data

HERE = Path(__file__).resolve().parent
DELAY = timedelta(minutes=15)


def is_midnight(stamp):
    return stamp.hour == 0 and stamp.minute == 0


def context_inputs(source, partitions, expected_sha256):
    """Read numerical context only through the last requested target, preserving original dates."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise DataValidationError("Original context source checksum differs from the frozen plan.")
    targets = tuple(stamp for partition in partitions for stamp in partition.timestamps)
    records = {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != list(SOURCE_COLUMNS):
            raise DataValidationError("Original context must retain the 11-column source schema.")
        for line, values in enumerate(reader, 2):
            if len(values) != len(SOURCE_COLUMNS):
                raise DataValidationError("Original context must have exactly 11 cells per row.")
            stamp = datetime.strptime(values[0], "%d/%m/%Y %H:%M")
            if stamp > targets[-1]:
                continue
            if stamp in records:
                raise DataValidationError("Original context contains a duplicate timestamp.")
            records[stamp] = _parse_record(dict(zip(SOURCE_COLUMNS, values, strict=True)), line)
    if any(stamp not in records for stamp in targets):
        raise DataValidationError("Original context is missing a fixed target observation.")
    expected = np.concatenate([partition.target for partition in partitions])
    if not np.allclose([records[stamp].usage_kwh for stamp in targets], expected, rtol=0, atol=1e-9):
        raise DataValidationError("Original context target values differ from the fixed sample.")
    history = tuple(AvailableObservation(row, row.timestamp + DELAY) for row in records.values()
                    if not is_midnight(row.timestamp))
    return build_forecast_inputs(targets, history)


def metrics(target, output, high, mask):
    errors = output[mask] - target[mask]
    high_errors = (output - target)[mask & high]
    return {"rows": int(mask.sum()), "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "p95_absolute_error": float(np.quantile(np.abs(errors), 0.95)),
            "high_usage_rows": int(len(high_errors)),
            "high_usage_mae": float(np.mean(np.abs(high_errors))) if high_errors.size else None,
            "high_usage_bias": float(np.mean(high_errors)) if high_errors.size else None}


def acceptance_checks(scores):
    checks = {}
    selected = scores["candidate"]
    for name in ("calendar", "persistence"):
        control = scores[name]
        checks.update({name + "_mae": selected["mae"] <= control["mae"] * 0.99,
                       name + "_rmse": selected["rmse"] <= control["rmse"],
                       name + "_p95": selected["p95_absolute_error"] <= control["p95_absolute_error"],
                       name + "_high_mae": (selected["high_usage_mae"] is not None
                                            and selected["high_usage_mae"] <= control["high_usage_mae"]),
                       name + "_high_bias": (selected["high_usage_bias"] is not None
                                             and abs(selected["high_usage_bias"])
                                             <= abs(control["high_usage_bias"]) * 0.9)})
    return checks


def replay(source, directory, workspace, plan):
    """Repeat one fixed configuration for reproducibility, with the original 15-minute delay."""
    _, experiment = open_store(workspace, "steel-energy-dense-history-screen")
    with threadpool_limits(limits=1), mlflow.start_run(
        experiment_id=experiment, run_name="dense-history-frozen-validation-replay"
    ) as run:
        # Log the selection and acceptance criteria before reading validation observations.
        mlflow.log_dict(plan, "frozen_validation_plan.json")
        mlflow.log_artifact(str(Path(__file__)), "source")
        data = load_training_data(directory)
        train, validation = data.train, data.validation
        if train.sha256 != plan["train_sha256"]:
            raise DataValidationError("Training checksum differs from the frozen selection.")
        if validation.sha256 != plan["validation_sha256"]:
            raise DataValidationError("Validation checksum differs from the frozen validation.")
        inputs = context_inputs(source, (train, validation), plan["context_source_sha256"])
        features = np.asarray([row.values for row in inputs])
        train_features, validation_features = features[:len(train.target)], features[len(train.target):]
        fit_mask = np.asarray([not is_midnight(stamp) for stamp in train.timestamps])
        midnight = np.asarray([is_midnight(stamp) for stamp in validation.timestamps])
        if not fit_mask.any() or any(stamp + DELAY > validation.timestamps[0] - DELAY
                                     for stamp, keep in zip(train.timestamps, fit_mask) if keep):
            raise DataValidationError("Fitted labels must be available by the first validation forecast origin.")
        calendar = RandomForestRegressor(n_estimators=100, criterion="absolute_error", max_depth=6,
                                         min_samples_leaf=2, random_state=42, n_jobs=1)
        calendar.fit(train.features[fit_mask], train.target[fit_mask])
        calendar_prediction = np.maximum(calendar.predict(validation.features), 0)
        persistence = np.where(validation_features[:, -1] == 1, validation_features[:, 7],
                               np.median(train.target[fit_mask]))
        persistence[midnight] = calendar_prediction[midnight]
        candidate = HistGradientBoostingRegressor(**plan["candidate"])
        candidate.fit(train_features[fit_mask], train.target[fit_mask])
        prediction = np.maximum(candidate.predict(validation_features), 0)
        prediction[midnight] = calendar_prediction[midnight]
        cutoff = float(np.quantile(train.target[fit_mask], 0.75))
        high = validation.target > cutoff
        all_rows = np.ones(len(validation.target), dtype=bool)
        outputs = {"calendar": calendar_prediction, "persistence": persistence, "candidate": prediction}
        scores = {name: metrics(validation.target, values, high, all_rows) for name, values in outputs.items()}
        checks = acceptance_checks(scores)
        midpoint = len(validation.target) // 2
        masks = {"first_half": np.arange(len(validation.target)) < midpoint,
                 "second_half": np.arange(len(validation.target)) >= midpoint, "non_midnight": ~midnight}
        result = {"run_id": run.info.run_id, "plan": plan, "validation_sha256": validation.sha256,
                  "fitted_label_rows": int(fit_mask.sum()), "target_rows": len(validation.target),
                  "midnight_fallback_targets": int(midnight.sum()), "high_usage_cutoff": cutoff,
                  "scores": scores, "checks": checks, "eligible_for_further_development": all(checks.values()),
                  "diagnostics": {label: {name: metrics(validation.target, values, high, mask)
                                          for name, values in outputs.items()} for label, mask in masks.items()},
                  "final_test_evaluated": False, "registry_mutated": False, "is_replay": True}
        trace = [{"target": row.target_timestamp.isoformat(), "origin": row.forecast_origin.isoformat(),
                  "history_timestamp": None if row.source_timestamp is None else row.source_timestamp.isoformat(),
                  "history_available_at": (None if row.source_available_at is None
                                           else row.source_available_at.isoformat()),
                  "actual_kwh": float(validation.target[index]),
                  **{name + "_kwh": float(values[index]) for name, values in outputs.items()},
                  "midnight_calendar_fallback": bool(midnight[index])}
                 for index, row in enumerate(inputs[len(train.target):])]
        mlflow.log_dict({"rows": trace}, "validation_predictions_and_availability.json")
        mlflow.log_dict(result, "validation_summary.json")
        mlflow.set_tags({"purpose": "frozen_validation_replay", "test_evaluated": "false",
                         "registry_mutated": "false"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/Steel_industry_data.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--workspace", type=Path, default=Path("artifacts/history-validation-replay"))
    args = parser.parse_args()
    snapshot = json.loads((HERE / "validation.json").read_text())
    plan = {**snapshot["plan"], "validation_sha256": snapshot["validation_sha256"]}
    print(json.dumps(replay(args.source, args.data, args.workspace, plan), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
