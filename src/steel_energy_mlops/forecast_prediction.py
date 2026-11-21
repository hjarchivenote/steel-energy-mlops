"""Logged HTTP-ready predictions using a single pinned history/calendar model pair."""

from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.forecast_inputs import AvailableObservation
from steel_energy_mlops.forecast_models import DEFAULT_FORECAST_MODEL, load_forecast_pair
from steel_energy_mlops.forecast_routing import build_forecast_routes
from steel_energy_mlops.forecast_routing import MINIMUM_HISTORY_DELAY
from steel_energy_mlops.tracking import DEFAULT_WORKSPACE, open_store


class ForecastRequestError(DataValidationError):
    """A validly shaped request predates the pinned model's training-label availability."""


def forecast_status(workspace: Path, model_name: str = DEFAULT_FORECAST_MODEL) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        return load_forecast_pair(workspace, model_name)[2]


def predict_forecast(
    timestamps: tuple[datetime, ...], observations: tuple[AvailableObservation, ...],
    workspace: Path = DEFAULT_WORKSPACE, model_name: str = DEFAULT_FORECAST_MODEL, request_id: str | None = None,
) -> dict[str, Any]:
    if not 1 <= len(timestamps) <= 1000 or len(observations) > 10000:
        raise DataValidationError("Provide 1-1000 forecast targets and at most 10000 historical observations.")
    routes = build_forecast_routes(timestamps, observations)
    with threadpool_limits(limits=1):
        history_model, calendar_model, metadata = load_forecast_pair(workspace, model_name)
        ready_at = datetime.fromisoformat(metadata["train_last_timestamp"]) + MINIMUM_HISTORY_DELAY
        if any(row.forecast_origin < ready_at for row in routes):
            raise ForecastRequestError("Forecast origin precedes the pinned model's training-label availability.")
        values = np.empty(len(routes), dtype=np.float64)
        for use_history, model in ((True, history_model), (False, calendar_model)):
            indices = [index for index, row in enumerate(routes) if (row.history is not None) == use_history]
            if not indices:
                continue
            selected = [routes[index] for index in indices]
            features = np.asarray([row.history.values if row.history is not None else row.calendar_values
                                   for row in selected], dtype=np.float64)
            output = np.asarray(model.predict(features), dtype=np.float64)
            if output.shape != (len(indices),) or not np.isfinite(output).all():
                raise DataValidationError("Forecast model returned invalid prediction values.")
            values[indices] = np.maximum(output, 0.)  # Match the frozen experiment's non-negative clipping.
    _, experiment = open_store(workspace, "steel-energy-forecast-prediction")
    with mlflow.start_run(experiment_id=experiment, run_name="forecast") as run:
        predictions = []
        for route, value in zip(routes, values, strict=True):
            history = route.history
            predictions.append({"timestamp": route.target_timestamp.isoformat(), "usage_kwh": float(value),
                                "forecast_origin": route.forecast_origin.isoformat(),
                                "method": route.method, "reason": route.reason,
                                "source_timestamp": (None if history is None or history.source_timestamp is None
                                                     else history.source_timestamp.isoformat()),
                                "source_available_at": (None if history is None or history.source_available_at is None
                                                        else history.source_available_at.isoformat()),
                                "history_age_minutes": None if history is None else history.values[-2]})
        result = {**metadata, "request_id": request_id, "run_id": run.info.run_id, "predictions": predictions}
        mlflow.log_dict({"timestamps": [stamp.isoformat() for stamp in timestamps],
                         "observations": [{"timestamp": row.record.timestamp.isoformat(),
                                           "available_at": row.available_at.isoformat(),
                                           "usage_kwh": row.record.usage_kwh,
                                           "electrical": vars(row.record.electrical)} for row in observations],
                         "request_id": request_id}, "request.json")
        mlflow.log_dict(result, "response.json")
        mlflow.log_dict({"rows": [{"timestamp": route.target_timestamp.isoformat(), "method": route.method,
                                   "reason": route.reason,
                                   "features": (list(route.history.values) if route.history is not None
                                                else list(route.calendar_values))} for route in routes]}, "inputs.json")
        mlflow.set_tags({"http_request_id": request_id or "unavailable", "model_name": model_name,
                         "model_version": metadata["model_version"], "routing_policy": metadata["routing_policy"],
                         "publication_run_id": metadata["publication_run_id"]})
    return result
