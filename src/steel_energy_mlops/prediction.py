"""Pin one champion version per request and log the inputs and outputs used."""

from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.sklearn import load_model
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.model_io import load_checked_model
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE, champion_version, open_store
from steel_energy_mlops.tracking import require_compatible_version, store_client, validate_model_name


def _load_champion(client: MlflowClient, model_name: str) -> tuple[ModelVersion, Pipeline]:
    version = champion_version(client, model_name)
    if version is None:
        raise DataValidationError("No validated champion model is available; train an eligible model first.")
    require_compatible_version(version)
    model = load_checked_model(load_model, f"models:/{model_name}/{version.version}")
    if not isinstance(model, Pipeline):
        raise DataValidationError("Registered champion must contain a fitted sklearn Pipeline.")
    return version, model


def _model_metadata(version: ModelVersion) -> dict[str, str]:
    return {"model_name": version.name, "model_version": str(version.version),
            "promotion_policy": version.tags.get("promotion_policy", "legacy-v1"),
            "feature_contract": version.tags["feature_contract"], "unit": "kWh"}


def model_status(workspace: Path = DEFAULT_WORKSPACE, model_name: str = DEFAULT_MODEL) -> dict[str, str]:
    """Check registry compatibility and model deserialization without creating a tracking run."""
    validate_model_name(model_name)
    version, _ = _load_champion(store_client(workspace), model_name)
    return _model_metadata(version)


def predict_consumption(
    timestamps: tuple[datetime, ...], workspace: Path = DEFAULT_WORKSPACE, model_name: str = DEFAULT_MODEL,
    request_id: str | None = None,
) -> dict[str, Any]:
    validate_model_name(model_name)
    if not 1 <= len(timestamps) <= 1000:
        raise DataValidationError("Provide between 1 and 1000 timestamps per prediction request.")
    if any(stamp.minute % 15 or stamp.second or stamp.microsecond for stamp in timestamps):
        raise DataValidationError("Prediction timestamps must lie on the source's 15-minute grid.")
    features = np.asarray([calendar_features(stamp) for stamp in timestamps], dtype=np.float64)
    client, experiment_id = open_store(workspace, "steel-energy-prediction")
    with threadpool_limits(limits=1):
        version, model = _load_champion(client, model_name)
        values = np.asarray(model.predict(features), dtype=np.float64)
    if values.shape != (len(timestamps),) or not np.isfinite(values).all() or np.any(values < 0):
        raise DataValidationError("Model returned invalid consumption predictions.")
    with mlflow.start_run(experiment_id=experiment_id, run_name="predict") as run:
        response = {"run_id": run.info.run_id, **_model_metadata(version), "request_id": request_id,
                    "predictions": [{"timestamp": stamp.isoformat(), "usage_kwh": float(value)}
                                    for stamp, value in zip(timestamps, values, strict=True)]}
        mlflow.log_params({"model_name": model_name, "model_version": version.version, "records": len(timestamps)})
        if request_id is not None:
            mlflow.set_tag("http_request_id", request_id)
        mlflow.log_dict({"timestamps": [stamp.isoformat() for stamp in timestamps], "request_id": request_id},
                        "request.json")
        mlflow.log_dict(response, "response.json")
        frame = pd.DataFrame({name: features[:, index] for index, name in enumerate(FEATURE_NAMES)})
        frame.insert(0, "timestamp", np.asarray([stamp.isoformat() for stamp in timestamps]))
        frame["prediction_kwh"] = values
        mlflow.log_text(frame.to_csv(index=False), "predictions.csv")
    return response
