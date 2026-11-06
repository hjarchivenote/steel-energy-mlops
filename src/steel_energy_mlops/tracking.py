"""Local SQLite tracking and explicit versioned model references."""

import os
import re
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.training_data import FEATURE_CONTRACT

DEFAULT_WORKSPACE = Path("artifacts/mlflow")
DEFAULT_MODEL = "steel-energy-calendar"


def open_store(workspace: Path, experiment_name: str) -> tuple[MlflowClient, str]:
    if mlflow.active_run() is not None:
        raise DataValidationError("Finish the active MLflow run before starting another application operation.")
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + str(workspace / "mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    parent = os.environ.get("MLFLOW_RUN_ID")
    if parent:
        return client, client.get_run(parent).info.experiment_id
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return client, client.create_experiment(experiment_name, artifact_location=(workspace / "artifacts").as_uri())
    return client, experiment.experiment_id


def validate_model_name(name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name) is None:
        raise DataValidationError("Model name must contain only letters, digits, dots, underscores, and hyphens.")


def champion_version(client: MlflowClient, name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, "champion")
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None


def require_compatible_version(version: ModelVersion) -> None:
    if version.tags.get("feature_contract") != FEATURE_CONTRACT:
        raise DataValidationError("Registered model has an incompatible or missing feature contract.")
