"""A bounded polling service for saved monitoring reports; it never trains or promotes models."""

import json
import logging
from pathlib import Path
import signal
from threading import Event
import time
from typing import Any
from uuid import uuid4

from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.monitoring import monitor_predictions, validate_monitor_options

LOGGER = logging.getLogger(__name__)


def _write_status(workspace: Path, state: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "monitoring_service.json"
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps({"updated_at": time.time(), **state}, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def monitoring_service_healthy(workspace: Path, interval_seconds: int = 60) -> bool:
    try:
        state = json.loads((workspace / "monitoring_service.json").read_text(encoding="utf-8"))
        age = time.time() - state["updated_at"]
        return (type(state["updated_at"]) in (int, float) and 0 <= age <= max(180, interval_seconds * 3)
                and state["status"] in ("updated", "skipped_unchanged", "skipped_busy"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def watch_monitoring(
    directory: Path, workspace: Path, model_name: str, lookback_hours: int = 168, max_runs: int = 100,
    interval_seconds: int = 60, stop: Event | None = None,
) -> dict[str, str]:
    validate_monitor_options(model_name, lookback_hours, max_runs)
    if type(interval_seconds) is not int or not 1 <= interval_seconds <= 3600:
        raise DataValidationError("Monitoring interval must be between 1 and 3600 seconds.")
    install_handlers = stop is None
    stop = Event() if stop is None else stop
    previous_handlers = {}
    if install_handlers:
        for name in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[name] = signal.signal(name, lambda *_: stop.set())
    try:
        while not stop.is_set():
            try:
                result = monitor_predictions(directory, workspace, model_name, lookback_hours, max_runs,
                                             only_if_changed=True)
                status = result["status"] if result["status"].startswith("skipped_") else "updated"
                if result["status"] == "retry_wait":
                    status = "retry_wait"
                state = {"status": status, "model_name": model_name, "monitor_run_id": result.get("run_id")}
            except (OSError, ValueError, MlflowException, SQLAlchemyError):
                LOGGER.exception("Automatic monitoring failed")
                state = {"status": "error", "model_name": model_name}
            _write_status(workspace, state)
            stop.wait(interval_seconds)
    finally:
        for name, previous in previous_handlers.items():
            signal.signal(name, previous)
        _write_status(workspace, {"status": "stopped", "model_name": model_name})
    return {"status": "stopped"}
