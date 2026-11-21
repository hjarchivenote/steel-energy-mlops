"""Version-scoped request distribution monitoring, without target-error or retraining claims."""

from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.sklearn import load_model
from filelock import FileLock, Timeout
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE, champion_version, open_store
from steel_energy_mlops.tracking import require_compatible_version, validate_model_name
from steel_energy_mlops.training_data import load_training_partition

MONITORED_COLUMNS = (*FEATURE_NAMES, "prediction_kwh")
MINIMUM_ROWS = 100
DRIFT_THRESHOLD = 0.1
REPORT_DECIMAL_PLACES = 12
INTERPRETATION = (
    "Calendar-only inputs: drift describes request schedule and prediction distributions, not measured factory load. "
    "Seasonality and deliberate request selection can explain changes. No observed targets are available, so model "
    "accuracy and energy savings are not assessed. A drift signal requests investigation, never automatic retraining. "
    "No detected drift is not proof of model quality. The minimum sample size is an operational guard, "
    "not a power test."
)


def _logged_frame(path: Path, records: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"timestamp": str}, keep_default_na=False)
    if list(frame.columns) != ["timestamp", *MONITORED_COLUMNS] or not 1 <= len(frame) <= 1000:
        raise DataValidationError("Prediction artifact has an incompatible schema or row count.")
    if len(frame) != int(records):
        raise DataValidationError("Prediction artifact row count differs from its logged request.")
    stamps = tuple(datetime.fromisoformat(value) for value in frame["timestamp"])
    expected = np.asarray([calendar_features(stamp) for stamp in stamps])
    if any(stamp.minute % 15 or stamp.second or stamp.microsecond for stamp in stamps):
        raise DataValidationError("Prediction artifact timestamps must lie on the 15-minute grid.")
    values = frame[list(MONITORED_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values[:, -1] < 0):
        raise DataValidationError("Prediction artifact values must be finite and predictions non-negative.")
    if not np.allclose(values[:, :-1], expected, rtol=0, atol=1e-12):
        raise DataValidationError("Prediction artifact features differ from their calendar timestamps.")
    frame["timestamp"] = [stamp.isoformat() for stamp in stamps]
    return frame


def _prediction_runs(
    client: MlflowClient, model_name: str, version: str, since: datetime, as_of: datetime, max_runs: int,
) -> list[Run]:
    experiment = client.get_experiment_by_name("steel-energy-prediction")
    if experiment is None:
        return []
    filters = (f"params.model_name = '{model_name}' AND params.model_version = '{version}' "
               f"AND attributes.status = 'FINISHED' AND attributes.start_time >= {int(since.timestamp() * 1000)} "
               f"AND attributes.start_time <= {int(as_of.timestamp() * 1000)}")
    return list(client.search_runs([experiment.experiment_id], filter_string=filters, max_results=max_runs,
                                   order_by=["attributes.start_time DESC", "attributes.run_id ASC"]))


def _current(client: MlflowClient, runs: list[Run]) -> tuple[pd.DataFrame, list[str], int]:
    frames, run_ids = [], []
    for run in runs:
        path = Path(client.download_artifacts(run.info.run_id, "predictions.csv"))
        frames.append(_logged_frame(path, run.data.params.get("records", "0")))
        run_ids.append(run.info.run_id)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        {name: [] for name in ("timestamp", *MONITORED_COLUMNS)}
    )
    raw_rows = len(frame)
    # Repeated API calls for the same target timestamp are not independent observations.
    frame = frame.drop_duplicates("timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
    return frame, run_ids, raw_rows


def _drift_report(current: pd.DataFrame, reference: pd.DataFrame, output: Path) -> dict[str, Any]:
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift

    schema = DataDefinition(numerical_columns=[name for name in MONITORED_COLUMNS if name != "is_weekend"],
                            categorical_columns=["is_weekend"])
    report = Report([
        ValueDrift(column=name, threshold=DRIFT_THRESHOLD,
                   method="jensenshannon" if name == "is_weekend" else "wasserstein")
        for name in MONITORED_COLUMNS
    ], include_tests=True)
    # Collapse CSV/circular-feature roundoff before binning nearly constant columns.
    # Only monitoring copies are rounded; serving values and stored source artifacts stay intact.
    snapshot = report.run(
        Dataset.from_pandas(current.reindex(columns=pd.Index(MONITORED_COLUMNS)).round(REPORT_DECIMAL_PLACES),
                            data_definition=schema),
        Dataset.from_pandas(reference.reindex(columns=pd.Index(MONITORED_COLUMNS)).round(REPORT_DECIMAL_PLACES),
                            data_definition=schema),
    )
    snapshot.save_html(str(output))
    return snapshot.dict()


def validate_monitor_options(model_name: str, lookback_hours: int, max_runs: int) -> None:
    validate_model_name(model_name)
    if type(lookback_hours) is not int or not 1 <= lookback_hours <= 720:
        raise DataValidationError("Monitoring lookback must be between 1 and 720 hours.")
    if type(max_runs) is not int or not 1 <= max_runs <= 1000:
        raise DataValidationError("Monitoring run limit must be between 1 and 1000.")


def monitor_predictions(
    directory: Path = Path("data/processed"), workspace: Path = DEFAULT_WORKSPACE, model_name: str = DEFAULT_MODEL,
    lookback_hours: int = 168, max_runs: int = 100, only_if_changed: bool = False,
) -> dict[str, Any]:
    validate_monitor_options(model_name, lookback_hours, max_runs)
    if type(only_if_changed) is not bool:
        raise DataValidationError("Monitoring change detection must be a boolean.")
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    lease = FileLock(workspace / "monitoring.lock", timeout=0)
    try:
        lease.acquire()
    except Timeout:
        return {"status": "skipped_busy", "model_name": model_name}
    try:
        return _monitor_locked(directory, workspace, model_name, lookback_hours, max_runs, only_if_changed)
    finally:
        lease.release()


def _refresh_decision(previous: Run | None, fingerprint: str, now: datetime) -> str | None:
    if previous is None or previous.data.params.get("monitor_fingerprint") != fingerprint:
        return None
    age = now.timestamp() - (previous.info.end_time or previous.info.start_time or 0) / 1000
    if previous.info.status == "FINISHED" and 0 <= age < 86400:
        return "skipped_unchanged"
    if previous.info.status in ("FAILED", "KILLED") and 0 <= age < 300:
        return "retry_wait"
    return None


def _monitor_locked(
    directory: Path, workspace: Path, model_name: str, lookback_hours: int, max_runs: int, only_if_changed: bool,
) -> dict[str, Any]:
    client, experiment_id = open_store(workspace, "steel-energy-monitoring")
    version = champion_version(client, model_name)
    if version is None:
        raise DataValidationError("No validated champion is available for monitoring.")
    require_compatible_version(version)
    as_of = datetime.now(timezone.utc)
    since = as_of - timedelta(hours=lookback_hours)
    runs = _prediction_runs(client, model_name, str(version.version), since, as_of, max_runs)
    reference_hashes = {}
    for name in ("train.csv", "manifest.json"):
        try:
            reference_hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        except OSError:
            # Start a failed run below rather than hiding an unreadable reference behind an old snapshot.
            reference_hashes[name] = None
    fingerprint = hashlib.sha256(json.dumps({
        "model_name": model_name, "model_version": str(version.version),
        "train_sha256": version.tags.get("train_sha256", ""), "lookback_hours": lookback_hours, "max_runs": max_runs,
        "source_run_ids": sorted(run.info.run_id for run in runs),
        "reference_files": reference_hashes,
        "report_policy": [MINIMUM_ROWS, DRIFT_THRESHOLD, REPORT_DECIMAL_PLACES, package_version("evidently")],
    }, sort_keys=True).encode()).hexdigest()
    previous = client.search_runs([experiment_id], max_results=1,
                                  filter_string=f"params.model_name = '{model_name}' "
                                                f"AND params.model_version = '{version.version}' "
                                                f"AND params.train_sha256 = '{version.tags.get('train_sha256', '')}' "
                                                "AND tags.operation = 'request_distribution_monitoring'",
                                  order_by=["attributes.start_time DESC", "attributes.run_id ASC"])
    latest = previous[0] if previous else None
    # Owning the shared lock proves any older RUNNING attempt has lost its monitor process.
    if latest is not None and latest.info.status == "RUNNING":
        client.set_terminated(latest.info.run_id, status="KILLED")
        latest = None
    decision = _refresh_decision(latest, fingerprint, as_of) if only_if_changed else None
    if decision is not None and latest is not None:
        return {"status": decision, "model_name": model_name, "run_id": latest.info.run_id}
    with mlflow.start_run(experiment_id=experiment_id, run_name="request-distribution-monitoring") as run:
        mlflow.set_tag("operation", "request_distribution_monitoring")
        mlflow.log_params({"model_name": model_name, "model_version": str(version.version),
                           "train_sha256": version.tags.get("train_sha256", ""),
                           "lookback_hours": lookback_hours, "max_runs": max_runs, "monitor_fingerprint": fingerprint})
        # Record the attempt before reading inputs so failures cannot leave an older verdict looking current.
        train = load_training_partition(directory)
        if version.tags.get("train_sha256") != train.sha256:
            raise DataValidationError("Monitoring reference must match the pinned champion's training checksum.")
        current, run_ids, raw_rows = _current(client, runs)
        summary: dict[str, Any] = {
            "run_id": run.info.run_id, "model_name": model_name, "model_version": str(version.version),
            "feature_contract": version.tags["feature_contract"],
            "reference": "training_partition", "train_sha256": train.sha256, "reference_rows": len(train.target),
            "since_request_time": since.isoformat(), "as_of_request_time": as_of.isoformat(),
            "max_runs": max_runs, "run_limit_reached": len(run_ids) == max_runs, "source_run_ids": run_ids,
            "raw_rows": raw_rows, "unique_rows": len(current), "duplicate_rows_removed": raw_rows - len(current),
            "current_first_timestamp": None if current.empty else current["timestamp"].iloc[0],
            "current_last_timestamp": None if current.empty else current["timestamp"].iloc[-1],
            "minimum_rows_per_dataset": MINIMUM_ROWS, "drift_threshold": DRIFT_THRESHOLD,
            "report_decimal_places": REPORT_DECIMAL_PLACES,
            "status": "insufficient_data", "drifted_columns": [], "column_scores": {}, "html_artifact": None,
            "performance_evaluated": False, "test_evaluated": False, "registry_mutated": False,
            "interpretation": INTERPRETATION,
        }
        if len(current) >= MINIMUM_ROWS and len(train.target) >= MINIMUM_ROWS:
            model = load_model(f"models:/{model_name}/{version.version}")
            if not isinstance(model, Pipeline):
                raise DataValidationError("Monitoring reference model must contain a fitted sklearn Pipeline.")
            with threadpool_limits(limits=1):
                predictions = np.asarray(model.predict(train.features), dtype=np.float64)
            if predictions.shape != train.target.shape or not np.isfinite(predictions).all() or np.any(predictions < 0):
                raise DataValidationError("Monitoring reference model returned invalid predictions.")
            reference = pd.DataFrame({name: train.features[:, index] for index, name in enumerate(FEATURE_NAMES)})
            reference["prediction_kwh"] = predictions
            with TemporaryDirectory() as temporary:
                output = Path(temporary) / "drift_report.html"
                result = _drift_report(current, reference, output)
                mlflow.log_artifact(str(output))
            summary["column_scores"] = {item["config"]["column"]: float(item["value"]) for item in result["metrics"]}
            summary["drifted_columns"] = [item["metric_config"]["params"]["column"] for item in result["tests"]
                                          if item["status"] == "FAIL"]
            summary["status"] = "distribution_changed" if summary["drifted_columns"] else "no_drift_detected"
            summary["html_artifact"] = f"runs:/{run.info.run_id}/drift_report.html"
            mlflow.log_dict(result, "evidently.json")
        mlflow.log_metrics({"reference_rows": len(train.target), "unique_rows": len(current),
                            "duplicate_rows_removed": raw_rows - len(current)})
        mlflow.log_dict(summary, "monitoring_summary.json")
        mlflow.log_text(current.to_csv(index=False), "current_predictions.csv")
    return summary
