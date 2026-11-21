"""Persist and pin a history estimator together with its matched calendar fallback."""

import hashlib
import json
import re
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
import mlflow
from mlflow.sklearn import load_model, log_model
import numpy as np
from numpy.typing import NDArray
import skops.io
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES
from steel_energy_mlops.forecast_inputs import FORECAST_FEATURE_CONTRACT, FORECAST_FEATURE_NAMES
from steel_energy_mlops.forecast_routing import FORECAST_ROUTING_POLICY
from steel_energy_mlops.model_io import load_checked_model
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE, champion_version, open_store
from steel_energy_mlops.tracking import store_client, validate_model_name

DEFAULT_FORECAST_MODEL = "steel-energy-history-15min"
FORECAST_PUBLICATION_POLICY = "history-frozen-validation-v1"
FROZEN_HISTORY_PARAMETERS = {"loss": "squared_error", "max_iter": 200, "max_leaf_nodes": 15,
                             "l2_regularization": 1., "early_stopping": False, "random_state": 42}
FROZEN_CALENDAR_PARAMETERS = {"n_estimators": 100, "criterion": "absolute_error", "max_depth": 6,
                              "min_samples_leaf": 2, "random_state": 42, "n_jobs": 1}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _require_evidence(evidence: dict[str, Any]) -> None:
    if not isinstance(evidence, dict):
        raise DataValidationError("Frozen validation evidence must be a JSON object.")
    plan = evidence.get("plan", {})
    names = {f"{control}_{metric}" for control in ("calendar", "persistence")
             for metric in ("mae", "rmse", "p95", "high_mae", "high_bias")}
    if (evidence.get("eligible_for_further_development") is not True
            or not isinstance(evidence.get("checks"), dict) or set(evidence["checks"]) != names
            or any(value is not True for value in evidence["checks"].values())
            or evidence.get("final_test_evaluated") is not False
            or not isinstance(plan, dict) or plan.get("candidate") != FROZEN_HISTORY_PARAMETERS
            or plan.get("reporting_delay_minutes") != 15 or plan.get("forecast_horizon_minutes") != 15
            or plan.get("availability_is_simulated") is not True):
        raise DataValidationError("Forecast publication requires the complete passed frozen validation evidence.")
    hashes = (plan.get("train_sha256"), plan.get("context_source_sha256"), evidence.get("validation_sha256"))
    if (any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in hashes)
            or re.fullmatch(r"[0-9a-f]{32}", str(evidence.get("run_id"))) is None):
        raise DataValidationError("Frozen evidence must identify its data and validation run.")
    _require_scores(evidence)


def _require_scores(evidence: dict[str, Any]) -> None:
    """Recompute the frozen criteria instead of trusting saved pass flags."""
    scores = evidence.get("scores")
    metrics = ("mae", "rmse", "p95_absolute_error", "high_usage_mae", "high_usage_bias")
    if not isinstance(scores, dict) or set(scores) != {"candidate", "calendar", "persistence"}:
        raise DataValidationError("Frozen evidence must contain all three validation scores.")
    for row in scores.values():
        if (not isinstance(row, dict) or type(row.get("rows")) is not int or row["rows"] != 750
                or type(row.get("high_usage_rows")) is not int or not 1 <= row["high_usage_rows"] <= 750
                or any(type(row.get(key)) not in (int, float) or not np.isfinite(row[key])
                       or (key != "high_usage_bias" and row[key] < 0) for key in metrics)):
            raise DataValidationError("Frozen validation scores require finite metrics for the fixed 750 targets.")
    candidate = scores["candidate"]
    for name in ("calendar", "persistence"):
        control = scores[name]
        if (candidate["high_usage_rows"] != control["high_usage_rows"]
                or candidate["mae"] > control["mae"] * 0.99
                or candidate["rmse"] > control["rmse"]
                or candidate["p95_absolute_error"] > control["p95_absolute_error"]
                or candidate["high_usage_mae"] > control["high_usage_mae"]
                or abs(candidate["high_usage_bias"]) > abs(control["high_usage_bias"]) * 0.9):
            raise DataValidationError("Saved pass flags disagree with the frozen validation criteria.")


def _require_model(model: Any, width: int) -> Pipeline:
    if not isinstance(model, Pipeline) or getattr(model, "n_features_in_", None) != width:
        raise DataValidationError("Forecast bundle requires fitted pipelines with the declared input widths.")
    return model


def _cutoff(value: Any) -> datetime:
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Forecast bundle must identify its training cutoff.") from exc
    if stamp.tzinfo is not None or stamp.minute % 15 or stamp.second or stamp.microsecond:
        raise DataValidationError("Forecast training cutoff must use the naive 15-minute grid.")
    return stamp


def publish_forecast_pair(
    history: Pipeline, calendar: Pipeline, example: NDArray[np.float64], evidence: dict[str, Any],
    workspace: Path = DEFAULT_WORKSPACE, model_name: str = DEFAULT_FORECAST_MODEL, *, training_cutoff: str,
) -> dict[str, Any]:
    """Publish one fixed pair after saved-model round trips; never replace the calendar champion."""
    validate_model_name(model_name)
    if model_name == DEFAULT_MODEL:
        raise DataValidationError("The history model must have a separate registered model name.")
    _require_evidence(evidence)
    _cutoff(training_cutoff)
    _require_model(history, len(FORECAST_FEATURE_NAMES))
    _require_model(calendar, len(FEATURE_NAMES))
    for pipeline, expected in ((history, HistGradientBoostingRegressor(**FROZEN_HISTORY_PARAMETERS)),
                               (calendar, RandomForestRegressor(**FROZEN_CALENDAR_PARAMETERS))):
        if (list(pipeline.named_steps) != ["model"]
                or not isinstance(pipeline["model"], type(expected))
                or pipeline["model"].get_params() != expected.get_params()):
            raise DataValidationError("Forecast estimator configuration differs from the frozen experiment.")
    if (example.ndim != 2 or example.shape[1] != len(FORECAST_FEATURE_NAMES) or len(example) < 1
            or not np.isfinite(example).all()):
        raise DataValidationError("Provide history feature examples for saved-model round-trip checks.")
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(workspace / "training.lock", timeout=0), threadpool_limits(limits=1):
            return _publish_pair(history, calendar, example, evidence, workspace, model_name, training_cutoff)
    except Timeout as exc:
        raise DataValidationError("Another publication or training job holds the workspace lock.") from exc


def _publish_pair(history, calendar, example, evidence, workspace, model_name, training_cutoff) -> dict[str, Any]:
    client, experiment = open_store(workspace, "steel-energy-forecast-publication")
    if champion_version(client, model_name) is not None:
        raise DataValidationError("This forecast model already has a champion; new validation is required.")
    with mlflow.start_run(experiment_id=experiment, run_name="publish-frozen-history-pair") as run:
        mlflow.log_dict(evidence, "frozen_validation.json")
        requirements = [f"{name}=={package_version(name)}" for name in
                        ("mlflow", "scikit-learn", "numpy", "scipy", "pandas", "skops")]
        uris = {}
        for role, model, sample, names in (
            ("history", history, example, FORECAST_FEATURE_NAMES),
            ("calendar", calendar, example[:, :len(FEATURE_NAMES)], FEATURE_NAMES),
        ):
            trusted = skops.io.get_untrusted_types(data=skops.io.dumps(model))
            info = log_model(model, name=role, input_example=sample, serialization_format="skops",
                             skops_trusted_types=trusted, pip_requirements=requirements,
                             metadata={"role": role, "feature_names": list(names)})
            restored = _require_model(load_checked_model(load_model, info.model_uri), len(names))
            if not np.allclose(model.predict(sample), restored.predict(sample), rtol=0, atol=1e-12):
                raise DataValidationError("Saved forecast model predictions differ from their training instance.")
            uris[role] = info.model_uri
        manifest = {"format_version": 1, "feature_contract": FORECAST_FEATURE_CONTRACT,
                    "routing_policy": FORECAST_ROUTING_POLICY, "publication_policy": FORECAST_PUBLICATION_POLICY,
                    "models": uris, "evidence_sha256": _digest(evidence), "evidence_run_id": evidence["run_id"],
                    "train_last_timestamp": training_cutoff,
                    "train_sha256": evidence["plan"]["train_sha256"],
                    "validation_sha256": evidence["validation_sha256"]}
        mlflow.log_dict(manifest, "forecast_bundle.json")
        for path in Path(__file__).parent.glob("*.py"):
            mlflow.log_artifact(str(path), "source")
        mlflow.set_tags({"purpose": "frozen_forecast_publication", "test_evaluated": "false",
                         "routing_policy": FORECAST_ROUTING_POLICY})
        registered = mlflow.register_model(uris["history"], model_name, tags={
            "feature_contract": FORECAST_FEATURE_CONTRACT, "routing_policy": FORECAST_ROUTING_POLICY,
            "bundle_sha256": _digest(manifest), "publication_policy": FORECAST_PUBLICATION_POLICY,
            "evidence_run_id": evidence["run_id"],
        })
    client.set_registered_model_alias(model_name, "champion", registered.version)
    return {"model_name": model_name, "model_version": str(registered.version), "run_id": run.info.run_id,
            "evidence_run_id": evidence["run_id"], "test_evaluated": False}


def load_forecast_pair(workspace: Path, model_name: str = DEFAULT_FORECAST_MODEL):
    """Resolve the alias once and verify both model references against one finished publication run."""
    validate_model_name(model_name)
    client = store_client(workspace)
    version = champion_version(client, model_name)
    if version is None or version.tags.get("feature_contract") != FORECAST_FEATURE_CONTRACT:
        raise DataValidationError("No compatible history forecast champion is available.")
    run_id = version.run_id
    if run_id is None or client.get_run(run_id).info.status != "FINISHED":
        raise DataValidationError("Forecast publication must finish before serving.")
    manifest = json.loads(Path(client.download_artifacts(run_id, "forecast_bundle.json")).read_text())
    if (not isinstance(manifest, dict) or _digest(manifest) != version.tags.get("bundle_sha256")
            or manifest.get("format_version") != 1 or manifest.get("feature_contract") != FORECAST_FEATURE_CONTRACT
            or manifest.get("routing_policy") != FORECAST_ROUTING_POLICY
            or manifest.get("publication_policy") != FORECAST_PUBLICATION_POLICY):
        raise DataValidationError("Forecast bundle metadata or checksums are incompatible.")
    evidence = json.loads(Path(client.download_artifacts(run_id, "frozen_validation.json")).read_text())
    if _digest(evidence) != manifest.get("evidence_sha256"):
        raise DataValidationError("Forecast validation evidence differs from its publication.")
    _require_evidence(evidence)
    if (manifest.get("evidence_run_id") != evidence["run_id"]
            or manifest.get("train_sha256") != evidence["plan"]["train_sha256"]
            or manifest.get("validation_sha256") != evidence["validation_sha256"]
            or version.tags.get("evidence_run_id") != evidence["run_id"]
            or version.tags.get("routing_policy") != FORECAST_ROUTING_POLICY
            or version.tags.get("publication_policy") != FORECAST_PUBLICATION_POLICY):
        raise DataValidationError("Forecast bundle provenance differs from its validation or registry tags.")
    _cutoff(manifest.get("train_last_timestamp"))
    models = manifest.get("models", {})
    if (not isinstance(models, dict) or set(models) != {"history", "calendar"}
            or any(re.fullmatch(r"models:/m-[0-9a-f]{32}", str(uri)) is None for uri in models.values())):
        raise DataValidationError("Forecast bundle requires immutable named-model references.")
    # Both logged models must belong to the same publication; a mutable alias cannot supply the fallback.
    if any(client.get_logged_model(uri.split("/")[-1]).source_run_id != version.run_id for uri in models.values()):
        raise DataValidationError("Forecast estimators must belong to the same publication run.")
    if models["history"] != version.source or models["history"] == models["calendar"]:
        raise DataValidationError("Forecast history reference must match the registry and use a distinct fallback.")
    history = _require_model(load_checked_model(load_model, models["history"]), len(FORECAST_FEATURE_NAMES))
    calendar = _require_model(load_checked_model(load_model, models["calendar"]), len(FEATURE_NAMES))
    metadata = {"model_name": model_name, "model_version": str(version.version),
                "feature_contract": FORECAST_FEATURE_CONTRACT, "routing_policy": FORECAST_ROUTING_POLICY,
                "publication_policy": FORECAST_PUBLICATION_POLICY, "evidence_run_id": manifest["evidence_run_id"],
                "publication_run_id": version.run_id, "unit": "kWh", "availability_is_assumed": True,
                "train_last_timestamp": manifest["train_last_timestamp"],
                "forecast_horizon_minutes": 15, "minimum_history_delay_minutes": 15,
                "maximum_history_age_at_origin_minutes": 30}
    return history, calendar, metadata
