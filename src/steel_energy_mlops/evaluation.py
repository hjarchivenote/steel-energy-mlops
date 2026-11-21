"""Read-only champion diagnostics on the fixed validation partition; no test scoring or promotion."""

from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.sklearn import load_model
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import DataValidationError, WEEKDAYS
from steel_energy_mlops.modeling import calendar_folds, error_metrics
from steel_energy_mlops.seasonal import PROFILE_STATISTICS, fit_schedule_profile
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE, champion_version, open_store
from steel_energy_mlops.tracking import require_compatible_version, validate_model_name
from steel_energy_mlops.training_data import Partition, load_training_data


def _take(partition: Partition, indices: NDArray[np.int64]) -> Partition:
    return Partition(partition.features[indices], partition.target[indices],
                     tuple(partition.timestamps[int(index)] for index in indices), partition.sha256)


def schedule_cv(train: Partition) -> dict[str, dict[str, Any]]:
    """Use exactly the existing three elapsed-time windows and refit every reference inside each fold."""
    scores: dict[str, list[float]] = {statistic: [] for statistic in PROFILE_STATISTICS}
    for left, right in calendar_folds(train.timestamps):
        history, future = _take(train, left), _take(train, right)
        for statistic in PROFILE_STATISTICS:
            prediction, _ = fit_schedule_profile(history, statistic).predict(future.timestamps)
            scores[statistic].append(error_metrics(future.target, prediction).mae)
    return {statistic: {"cv_mae": float(np.mean(values)), "fold_mae": values}
            for statistic, values in scores.items()}


def error_summary(target: NDArray[np.float64], predictions: NDArray[np.float64]) -> dict[str, Any]:
    metrics = error_metrics(target, predictions)
    absolute = np.abs(predictions - target)
    return {"row_count": int(target.size), **metrics.as_dict(),
            "bias_kwh": float(np.mean(predictions - target)),
            "p95_absolute_error_kwh": float(np.quantile(absolute, 0.95)),
            "maximum_absolute_error_kwh": float(np.max(absolute))}


def error_slices(
    validation: Partition, train_target: NDArray[np.float64], predictions: dict[str, NDArray[np.float64]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Observed usage bands are offline diagnostics, never predictors or deployable alert rules."""
    error_metrics(train_target, train_target)
    lower, upper = (float(value) for value in np.quantile(train_target, [0.25, 0.75]))
    frame = pd.DataFrame({"timestamp": [stamp.isoformat() for stamp in validation.timestamps],
                          "actual_kwh": validation.target})
    groups: dict[str, list[str]] = {
        "hour": [f"{stamp.hour:02d}" for stamp in validation.timestamps],
        "weekday": [WEEKDAYS[stamp.weekday()] for stamp in validation.timestamps],
        "day_type": ["weekend" if stamp.weekday() >= 5 else "weekday" for stamp in validation.timestamps],
        "month": [stamp.strftime("%Y-%m") for stamp in validation.timestamps],
        "date": [stamp.date().isoformat() for stamp in validation.timestamps],
        "observed_usage_band": ["low" if value <= lower else "high" if value > upper else "middle"
                                for value in validation.target],
    }
    slices = []
    for name, values in predictions.items():
        error_metrics(validation.target, values)
        frame[f"{name}_predicted_kwh"] = values
        frame[f"{name}_absolute_error_kwh"] = np.abs(values - validation.target)
        for dimension, labels in groups.items():
            for label in sorted(set(labels)):
                mask = np.asarray([item == label for item in labels])
                slices.append({"model": name, "dimension": dimension, "group": label,
                               **error_summary(validation.target[mask], values[mask])})
    return frame, pd.DataFrame(slices), {"train_q25_kwh": lower, "train_q75_kwh": upper}


def evaluate_champion(
    directory: Path = Path("data/processed"), workspace: Path = DEFAULT_WORKSPACE, model_name: str = DEFAULT_MODEL
) -> dict[str, Any]:
    validate_model_name(model_name)
    data = load_training_data(directory)
    client, experiment_id = open_store(workspace, "steel-energy-diagnostics")
    version = champion_version(client, model_name)
    if version is None:
        raise DataValidationError("No validated champion is available for diagnostics.")
    require_compatible_version(version)
    for key, expected in (("train_sha256", data.train.sha256), ("validation_sha256", data.validation.sha256),
                          ("train_last_timestamp", data.train.timestamps[-1].isoformat())):
        if version.tags.get(key) != expected:
            raise DataValidationError("Champion provenance must match these fixed training and validation partitions.")
    model_uri = f"models:/{model_name}/{version.version}"
    model = load_model(model_uri)
    if not isinstance(model, Pipeline):
        raise DataValidationError("Registered champion must contain a fitted sklearn Pipeline.")
    with threadpool_limits(limits=1):
        predictions = {"champion": np.asarray(model.predict(data.validation.features), dtype=np.float64)}
    comparison = [{"model": "champion", **error_summary(data.validation.target, predictions["champion"])}]
    cv = schedule_cv(data.train)
    for statistic in PROFILE_STATISTICS:
        name = f"weekday_hour_{statistic}"
        predictions[name], support = fit_schedule_profile(data.train, statistic).predict(data.validation.timestamps)
        comparison.append({"model": name, **error_summary(data.validation.target, predictions[name]),
                           **cv[statistic], "prediction_support": support})
    selected_statistic = min(PROFILE_STATISTICS, key=lambda item: cv[item]["cv_mae"])
    reference_name = f"weekday_hour_{selected_statistic}"
    reference = next(row for row in comparison if row["model"] == reference_name)
    champion = comparison[0]
    delta = np.abs(predictions["champion"] - data.validation.target) - np.abs(
        predictions[reference_name] - data.validation.target
    )
    frame, slices, bands = error_slices(data.validation, data.train.target, predictions)
    frame["champion_minus_reference_absolute_error_kwh"] = delta
    with mlflow.start_run(experiment_id=experiment_id, run_name="validation-diagnostics") as run:
        summary = {
            "run_id": run.info.run_id, "model_name": model_name, "model_version": str(version.version),
            "model_uri": model_uri, "training_run_id": version.run_id,
            "train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
            "comparison": comparison, "reference_selected_by": "training_elapsed_time_cv_mae",
            "selected_reference": reference_name, "observed_usage_thresholds": bands,
            "champion_minus_reference_mae_kwh": float(np.mean(delta)),
            "champion_beats_reference_mae": champion["mae"] < reference["mae"],
            "champion_beats_reference_rmse": champion["rmse"] < reference["rmse"],
            "test_evaluated": False, "registry_mutated": False,
            "interpretation": "Validation diagnostics only; no significance, energy savings, or root-cause claim. "
                              "Usage bands require observed targets; daily groups contain sampled observations, "
                              "not complete daily energy totals. Sparse groups are descriptive.",
        }
        mlflow.log_params({"model_name": model_name, "model_version": str(version.version),
                           "train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
                           "selected_reference": reference_name, "profile_granularity": "weekday_hour",
                           "profile_backoff": "weekday_hour,day_type_hour,hour,global"})
        for row in comparison:
            mlflow.log_metrics({f"{row['model']}_{metric}": row[metric] for metric in
                                ("mae", "rmse", "bias_kwh", "p95_absolute_error_kwh")})
        mlflow.log_dict(summary, "validation_summary.json")
        mlflow.log_dict(data.manifest, "data/preparation_manifest.json")
        mlflow.log_text(frame.to_csv(index=False), "validation_predictions.csv")
        mlflow.log_text(slices.to_csv(index=False), "error_slices.csv")
        largest = frame.nlargest(20, "champion_absolute_error_kwh")
        mlflow.log_text(largest.to_csv(index=False), "largest_champion_errors.csv")
        for source in sorted(Path(__file__).parent.glob("*.py")):
            mlflow.log_text(source.read_text(encoding="utf-8"), f"source/{source.name}")
        mlflow.set_tags({"purpose": "validation_diagnostics", "test_evaluated": "false",
                         "registry_mutated": "false", "evaluated_at": datetime.now().isoformat()})
    return summary
