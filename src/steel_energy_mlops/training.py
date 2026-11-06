"""Train, record, validate, and conditionally promote model versions."""

import subprocess
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import mlflow
from mlflow.sklearn import load_model, log_model
import numpy as np
import pandas as pd
import skops.io
from filelock import FileLock, Timeout
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES
from steel_energy_mlops.modeling import BASELINES, LEARNED_FAMILIES, ErrorMetrics, error_metrics, fit_candidates
from steel_energy_mlops.modeling import promotion_decision
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE, champion_version, open_store
from steel_energy_mlops.tracking import require_compatible_version, validate_model_name
from steel_energy_mlops.training_data import FEATURE_CONTRACT, TrainingData, load_training_data


def train_models(
    directory: Path = Path("data/processed"),
    workspace: Path = DEFAULT_WORKSPACE,
    model_name: str = DEFAULT_MODEL,
    families: tuple[str, ...] = LEARNED_FAMILIES,
    forest_trees: int = 100,
    minimum_improvement: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    validate_model_name(model_name)
    if not 0 <= minimum_improvement < 1:
        raise DataValidationError("Minimum improvement must be a fraction in [0, 1).")
    data = load_training_data(directory)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(workspace / "training.lock", timeout=0), threadpool_limits(limits=1):
            return _train_locked(
                data, directory, workspace, model_name, families, forest_trees, minimum_improvement, seed
            )
    except Timeout as exc:
        raise DataValidationError("Another training job holds this workspace lock; retry after it finishes.") from exc


def _git_provenance() -> dict[str, str | bool]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    except OSError:
        return {"git_revision": "unavailable", "git_dirty": "unknown"}
    return {"git_revision": revision.stdout.strip() or "unavailable",
            "git_dirty": "unknown" if revision.returncode else bool(status.stdout.strip())}


def _train_locked(
    data: TrainingData, directory: Path, workspace: Path, model_name: str,
    families: tuple[str, ...], forest_trees: int, minimum_improvement: float, seed: int
) -> dict[str, Any]:
    client, experiment_id = open_store(workspace, "steel-energy-training")
    incumbent_version = champion_version(client, model_name)
    incumbent_metrics = None
    if incumbent_version is not None:
        require_compatible_version(incumbent_version)
        training_cutoff = incumbent_version.tags.get("train_last_timestamp")
        if not training_cutoff or datetime.fromisoformat(training_cutoff) >= data.validation.timestamps[0]:
            raise DataValidationError("Incumbent training data overlaps this validation window or lacks provenance.")
        incumbent_model = load_model(f"models:/{model_name}/{incumbent_version.version}")
        if not isinstance(incumbent_model, Pipeline):
            raise DataValidationError("Registered champion must contain a fitted sklearn Pipeline.")
        predictions = np.asarray(incumbent_model.predict(data.validation.features), dtype=np.float64)
        incumbent_metrics = error_metrics(data.validation.target, predictions)

    with mlflow.start_run(experiment_id=experiment_id, run_name="train-and-validate") as run:
        mlflow.log_params({
            "feature_contract": FEATURE_CONTRACT, "seed": seed, "forest_trees": forest_trees,
            "minimum_improvement": minimum_improvement, "candidate_families": ",".join(families),
            "train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
            **_git_provenance(),
            "source_sha256": data.manifest["source"]["sha256"],
        })
        mlflow.log_dict(data.manifest, "data/preparation_manifest.json")
        mlflow.log_artifact(str(directory / "train.csv"), artifact_path="data")
        for source in sorted(Path(__file__).parent.glob("*.py")):
            mlflow.log_text(source.read_text(encoding="utf-8"), f"source/{source.name}")
        candidates, windows = fit_candidates(data.train, families, forest_trees, seed)
        mlflow.log_dict({"elapsed_time_cv_windows": windows}, "validation/cross_validation.json")
        selected = min((item for item in candidates if item.name not in BASELINES), key=lambda item: item.cv_mae)
        metrics: dict[str, ErrorMetrics] = {}
        models: dict[str, str] = {}
        for candidate in candidates:
            with mlflow.start_run(run_name=candidate.name, nested=True):
                predictions = np.asarray(candidate.model.predict(data.validation.features), dtype=np.float64)
                result = error_metrics(data.validation.target, predictions)
                metrics[candidate.name] = result
                mlflow.log_params({"family": candidate.name, **candidate.parameters})
                mlflow.log_metrics({"cv_mae": candidate.cv_mae, "validation_mae": result.mae,
                                    "validation_rmse": result.rmse})
                mlflow.log_dict({"trials": candidate.trials}, "cross_validation.json")
                output = pd.DataFrame({
                    "timestamp": [stamp.isoformat() for stamp in data.validation.timestamps],
                    "actual_kwh": data.validation.target, "predicted_kwh": predictions,
                    "absolute_error_kwh": np.abs(data.validation.target - predictions),
                })
                mlflow.log_text(output.to_csv(index=False), "validation_predictions.csv")
                trusted = skops.io.get_untrusted_types(data=skops.io.dumps(candidate.model))
                model = log_model(
                    candidate.model, name="model", input_example=data.train.features[:3],
                    skops_trusted_types=trusted,
                    pip_requirements=[f"{name}=={package_version(name)}" for name in
                                      ("mlflow", "scikit-learn", "numpy", "scipy", "pandas", "skops")],
                    metadata={"feature_contract": FEATURE_CONTRACT, "feature_names": list(FEATURE_NAMES)},
                )
                models[candidate.name] = model.model_uri
        decision = promotion_decision(
            metrics[selected.name], min(metrics[name].mae for name in BASELINES), metrics["dummy_mean"].rmse,
            incumbent_metrics, minimum_improvement,
        )
        mlflow.log_metrics({
            "selected_cv_mae": selected.cv_mae,
            "selected_validation_mae": metrics[selected.name].mae,
            "selected_validation_rmse": metrics[selected.name].rmse,
            "baseline_mean_validation_mae": metrics["dummy_mean"].mae,
            "baseline_median_validation_mae": metrics["dummy_median"].mae,
        })
        if incumbent_metrics is not None:
            mlflow.log_metrics({"incumbent_validation_mae": incumbent_metrics.mae,
                                "incumbent_validation_rmse": incumbent_metrics.rmse})
        registered = mlflow.register_model(models[selected.name], model_name, tags={
            "feature_contract": FEATURE_CONTRACT,
            "train_last_timestamp": data.train.timestamps[-1].isoformat(),
            "train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
            "selection": "training_elapsed_time_cv_mae", "gate_reason": decision.reason,
        })
        summary = {
            "run_id": run.info.run_id, "model_name": model_name, "selected_family": selected.name,
            "selection_basis": "training_elapsed_time_cv_mae", "candidate_version": str(registered.version),
            "previous_champion_version": None if incumbent_version is None else str(incumbent_version.version),
            "promotion_eligible": decision.eligible, "reason": decision.reason,
            "incumbent_validation": None if incumbent_metrics is None else incumbent_metrics.as_dict(),
            "comparison": [{"family": item.name, "cv_mae": item.cv_mae,
                            "validation": metrics[item.name].as_dict(), "parameters": item.parameters,
                            "model_uri": models[item.name]}
                           for item in candidates],
            "test_evaluated": False,
        }
        mlflow.log_dict(summary, "training_summary.json")
        mlflow.set_tags({"selected_family": selected.name, "promotion_eligible": str(decision.eligible),
                         "gate_reason": decision.reason, "test_evaluated": "false"})
    # All training and artifact writes finish before the serving alias can change.
    if decision.eligible:
        client.set_registered_model_alias(model_name, "champion", registered.version)
    summary["promoted"] = decision.eligible
    return summary
