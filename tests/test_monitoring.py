import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from steel_energy_mlops import monitoring
from steel_energy_mlops.cli import main
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES
from steel_energy_mlops.prediction import predict_consumption
from steel_energy_mlops.tracking import champion_version, store_client
from steel_energy_mlops.training import train_models
from steel_energy_mlops.training_data import load_training_partition
from tests.helpers import make_training_data

MODEL = "monitor-model"


@pytest.fixture
def monitored(tmp_path):
    directory = make_training_data(tmp_path)
    workspace = tmp_path / "tracking"
    assert train_models(directory, workspace, MODEL, families=("ridge",))["promoted"]
    train = load_training_partition(directory)
    (directory / "test.csv").unlink()
    (directory / "validation.csv").unlink()
    return directory, workspace, train, store_client(workspace)


def test_monitor_logs_real_evidently_artifacts_without_validation_or_test_data(monitored, capsys) -> None:
    directory, workspace, train, client = monitored
    prediction = predict_consumption(train.timestamps, workspace, MODEL)
    assert main(["monitor", "--data", str(directory), "--workspace", str(workspace), "--model-name", MODEL]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "no_drift_detected" and not result["drifted_columns"]
    assert result["unique_rows"] == result["reference_rows"] == len(train.target)
    assert set(result["column_scores"]) == set(monitoring.MONITORED_COLUMNS)
    np.testing.assert_allclose(list(result["column_scores"].values()), 0, rtol=0, atol=1e-9)
    assert result["source_run_ids"] == [prediction["run_id"]]
    assert not any(result[key] for key in ("test_evaluated", "performance_evaluated", "registry_mutated"))
    assert str(champion_version(client, MODEL).version) == result["model_version"]
    saved = Path(client.download_artifacts(result["run_id"], "monitoring_summary.json"))
    assert json.loads(saved.read_text()) == result
    html = Path(client.download_artifacts(result["run_id"], "drift_report.html")).read_text()
    assert "<html" in html.lower() and "</html>" in html.lower() and "<script" in html.lower()
    raw = json.loads(Path(client.download_artifacts(result["run_id"], "evidently.json")).read_text())
    assert len(raw["metrics"]) == len(FEATURE_NAMES) + 1
    assert {test["status"] for test in raw["tests"]} == {"SUCCESS"}


def test_shifted_request_schedule_produces_review_signal_without_promoting_a_model(monitored) -> None:
    directory, workspace, train, client = monitored
    predict_consumption(tuple(stamp + timedelta(days=31) for stamp in train.timestamps), workspace, MODEL)
    result = monitoring.monitor_predictions(directory, workspace, MODEL)
    assert result["status"] == "distribution_changed"
    assert {"month_sin", "month_cos"} <= set(result["drifted_columns"])
    assert "not measured factory load" in result["interpretation"]
    assert result["html_artifact"] == f"runs:/{result['run_id']}/drift_report.html"
    assert str(champion_version(client, MODEL).version) == "1"


def test_repeated_requests_do_not_inflate_the_monitoring_sample(monitored) -> None:
    directory, workspace, train, client = monitored
    predict_consumption((train.timestamps[0],) * 100, workspace, MODEL)
    result = monitoring.monitor_predictions(directory, workspace, MODEL)
    assert result["raw_rows"] == 100 and result["unique_rows"] == 1 and result["duplicate_rows_removed"] == 99
    assert result["status"] == "insufficient_data" and result["html_artifact"] is None
    assert "drift_report.html" not in [item.path for item in client.list_artifacts(result["run_id"])]


def test_missing_prediction_experiment_is_reported_as_insufficient_data(monitored) -> None:
    directory, workspace, _, _ = monitored
    result = monitoring.monitor_predictions(directory, workspace, MODEL)
    assert result["status"] == "insufficient_data" and result["source_run_ids"] == []
    assert result["current_first_timestamp"] is None and result["current_last_timestamp"] is None


def test_only_finished_requests_for_the_pinned_version_inside_the_request_window_are_selected(monitored) -> None:
    directory, workspace, train, client = monitored
    prediction = predict_consumption(train.timestamps, workspace, MODEL)
    experiment = client.get_experiment_by_name("steel-energy-prediction")
    now = datetime.now(timezone.utc)
    for model, version, stamp, status in (
        (MODEL, "2", now, "FINISHED"), ("another-model", "1", now, "FINISHED"),
        (MODEL, "1", now - timedelta(days=8), "FINISHED"), (MODEL, "1", now + timedelta(days=1), "FINISHED"),
        (MODEL, "1", now, "FAILED"), (MODEL, "1", now, "RUNNING"),
    ):
        run = client.create_run(experiment.experiment_id, start_time=int(stamp.timestamp() * 1000))
        client.log_param(run.info.run_id, "model_name", model)
        client.log_param(run.info.run_id, "model_version", version)
        if status != "RUNNING":
            client.set_terminated(run.info.run_id, status)
    result = monitoring.monitor_predictions(directory, workspace, MODEL)
    assert result["source_run_ids"] == [prediction["run_id"]]


def test_latest_run_limit_is_explicit_and_does_not_turn_sparse_requests_green(monitored) -> None:
    directory, workspace, train, _ = monitored
    predict_consumption(train.timestamps, workspace, MODEL)
    latest = predict_consumption(train.timestamps[:2], workspace, MODEL)
    result = monitoring.monitor_predictions(directory, workspace, MODEL, max_runs=1)
    assert result["source_run_ids"] == [latest["run_id"]] and result["run_limit_reached"]
    assert result["status"] == "insufficient_data" and result["unique_rows"] == 2


@pytest.mark.parametrize("key,value", [
    ("lookback_hours", 0), ("lookback_hours", 721), ("lookback_hours", True),
    ("max_runs", 0), ("max_runs", 1001), ("max_runs", True),
])
def test_monitor_rejects_unbounded_or_invalid_windows(tmp_path, key, value) -> None:
    with pytest.raises(DataValidationError, match="Monitoring"):
        monitoring.monitor_predictions(tmp_path, tmp_path, **{key: value})


def test_cli_reports_monitoring_configuration_errors(capsys) -> None:
    assert main(["monitor", "--max-runs", "0"]) == 2
    assert "Prediction monitoring failed:" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["missing", "contract", "checksum"])
def test_monitor_requires_a_compatible_champion_with_matching_reference_provenance(monitored, failure) -> None:
    directory, workspace, _, client = monitored
    if failure == "missing":
        client.delete_registered_model_alias(MODEL, "champion")
    else:
        key = "feature_contract" if failure == "contract" else "train_sha256"
        client.set_model_version_tag(MODEL, "1", key, "incompatible")
    with pytest.raises(DataValidationError, match="champion|contract|checksum"):
        monitoring.monitor_predictions(directory, workspace, MODEL)


@pytest.mark.parametrize("failure", ["not-pipeline", "negative-predictions"])
def test_invalid_reference_models_cannot_generate_monitoring_results(monitored, monkeypatch, failure) -> None:
    directory, workspace, train, _ = monitored
    predict_consumption(train.timestamps, workspace, MODEL)
    model = object() if failure == "not-pipeline" else Pipeline([
        ("regressor", DummyRegressor(strategy="constant", constant=-1))
    ]).fit(train.features, train.target)
    monkeypatch.setattr(monitoring, "load_model", lambda uri: model)
    with pytest.raises(DataValidationError, match="Monitoring reference model"):
        monitoring.monitor_predictions(directory, workspace, MODEL)


@pytest.mark.parametrize("failure", [
    "missing-column", "empty", "too-many", "count", "timestamp", "missing-timestamp",
    "timezone", "off-grid", "feature", "nan", "negative", "text",
])
def test_malformed_prediction_artifacts_are_rejected(prepared_training, tmp_path, failure) -> None:
    frame = pd.read_csv(prepared_training / "train.csv").rename(columns={"Usage_kWh": "prediction_kwh"})
    records = str(len(frame))
    if failure == "missing-column":
        frame = frame.drop(columns=["is_weekend"])
    elif failure == "empty":
        frame = frame.iloc[:0]
    elif failure == "too-many":
        frame = pd.concat([frame] * 10, ignore_index=True)
    elif failure == "count":
        records = "0"
    elif failure in ("timestamp", "missing-timestamp", "timezone", "off-grid"):
        frame.loc[0, "timestamp"] = {"timestamp": "invalid", "missing-timestamp": "",
                                     "timezone": "2018-01-01T00:00:00+00:00",
                                     "off-grid": "2018-01-01T00:01:00"}[failure]
    elif failure == "feature":
        frame.loc[0, "time_of_day_sin"] = 100
    else:
        frame["prediction_kwh"] = frame["prediction_kwh"].astype(object)
        frame.loc[0, "prediction_kwh"] = {"nan": np.nan, "negative": -1, "text": "invalid"}[failure]
    path = tmp_path / "logged.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        monitoring._logged_frame(path, records)
