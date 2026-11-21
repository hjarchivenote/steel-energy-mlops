from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import mlflow
import pytest
from streamlit.testing.v1 import AppTest

from steel_energy_mlops import dashboard
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.monitoring import MONITORED_COLUMNS, monitor_predictions
from steel_energy_mlops.prediction import predict_consumption
from steel_energy_mlops.tracking import store_client
from steel_energy_mlops.training import train_models
from steel_energy_mlops.training_data import load_training_partition
from tests.helpers import make_training_data

MODEL = "dashboard-model"
APP = Path(__file__).resolve().parents[1] / "dashboard/app.py"


def summary_for(state="insufficient_data"):
    now = datetime.now(timezone.utc)
    sufficient = state != "insufficient_data"
    scores = {name: 0.0 for name in MONITORED_COLUMNS} if sufficient else {}
    drifted = []
    if state == "distribution_changed":
        scores["month_sin"] = 0.5
        drifted = ["month_sin"]
    return {
        "run_id": "a" * 32, "model_name": MODEL, "model_version": "1", "train_sha256": "b" * 64,
        "status": state, "minimum_rows_per_dataset": 100, "drift_threshold": 0.1,
        "raw_rows": 104 if sufficient else 8, "unique_rows": 100 if sufficient else 4,
        "duplicate_rows_removed": 4, "reference_rows": 3500, "column_scores": scores, "drifted_columns": drifted,
        "html_artifact": "runs:/" + "a" * 32 + "/drift_report.html" if sufficient else None,
        "performance_evaluated": False, "test_evaluated": False, "registry_mutated": False,
        "since_request_time": (now - timedelta(days=7)).isoformat(), "as_of_request_time": now.isoformat(),
        "run_limit_reached": False, "max_runs": 100,
    }


@pytest.fixture
def saved_store(tmp_path):
    directory = make_training_data(tmp_path)
    workspace = tmp_path / "tracking"
    assert train_models(directory, workspace, MODEL, families=("ridge",))["promoted"]
    return directory, workspace, store_client(workspace)


def test_reader_does_not_create_a_missing_store_or_change_global_tracking_state(tmp_path) -> None:
    assert dashboard.load_dashboard(tmp_path / "missing", MODEL).state == "no_store"
    assert not (tmp_path / "missing").exists()
    store_client(tmp_path)
    original_uri = mlflow.get_tracking_uri()
    assert dashboard.load_dashboard(tmp_path, MODEL).state == "no_model"
    assert mlflow.get_tracking_uri() == original_uri


def test_reader_handles_no_snapshot_and_reloads_saved_results(saved_store) -> None:
    directory, workspace, _ = saved_store
    assert dashboard.load_dashboard(workspace, MODEL).state == "no_snapshot"
    summary = monitor_predictions(directory, workspace, MODEL)
    view = dashboard.load_dashboard(workspace, MODEL)
    assert view.state == "insufficient_data" and view.summary == summary and view.report_html is None


def test_reader_loads_evidently_and_can_display_an_old_snapshot(saved_store) -> None:
    directory, workspace, client = saved_store
    predict_consumption(load_training_partition(directory).timestamps, workspace, MODEL)
    summary = monitor_predictions(directory, workspace, MODEL)
    path = Path(client.download_artifacts(summary["run_id"], "monitoring_summary.json"))
    summary["as_of_request_time"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    path.write_text(json.dumps(summary))
    client.log_artifact(summary["run_id"], str(path))
    # The dashboard reads stored artifacts only, without any prepared data or model inference.
    for path in directory.glob("*.csv"):
        path.unlink()
    view = dashboard.load_dashboard(workspace, MODEL)
    assert view.stale and view.state == "no_drift_detected" and "<html" in view.report_html.lower()
    html = Path(client.download_artifacts(summary["run_id"], "drift_report.html"))
    html.write_text("incomplete")
    client.log_artifact(summary["run_id"], str(html))
    with pytest.raises(DataValidationError, match="HTML"):
        dashboard.load_dashboard(workspace, MODEL)


@pytest.mark.parametrize("status,state", [("FAILED", "monitoring_failed"), ("RUNNING", "monitoring_running")])
def test_latest_failed_or_running_attempt_does_not_expose_an_old_verdict(saved_store, status, state) -> None:
    directory, workspace, client = saved_store
    old = monitor_predictions(directory, workspace, MODEL)
    experiment = client.get_experiment_by_name("steel-energy-monitoring")
    run = client.create_run(experiment.experiment_id, tags={"operation": "request_distribution_monitoring"})
    for key in ("model_name", "model_version", "train_sha256"):
        client.log_param(run.info.run_id, key, old[key])
    if status != "RUNNING":
        client.set_terminated(run.info.run_id, status)
    view = dashboard.load_dashboard(workspace, MODEL)
    assert view.state == state and view.summary is None and view.report_html is None


def test_reader_filters_other_models_versions_and_reference_data(saved_store) -> None:
    directory, workspace, client = saved_store
    original = monitor_predictions(directory, workspace, MODEL)
    experiment = client.get_experiment_by_name("steel-energy-monitoring")
    for key in ("model_name", "model_version", "train_sha256"):
        run = client.create_run(experiment.experiment_id, tags={"operation": "request_distribution_monitoring"})
        for name in ("model_name", "model_version", "train_sha256"):
            client.log_param(run.info.run_id, name, "other" if name == key else original[name])
        client.set_terminated(run.info.run_id, "FAILED")
    assert dashboard.load_dashboard(workspace, MODEL).summary == original
    client.set_model_version_tag(MODEL, "1", "train_sha256", "c" * 64)
    assert dashboard.load_dashboard(workspace, MODEL).state == "no_snapshot"


def test_prediction_artifact_failure_is_recorded_before_an_old_snapshot_can_be_shown(saved_store) -> None:
    directory, workspace, client = saved_store
    monitor_predictions(directory, workspace, MODEL)
    stamps = load_training_partition(directory).timestamps[:2]
    prediction = predict_consumption(stamps, workspace, MODEL)
    path = Path(client.download_artifacts(prediction["run_id"], "predictions.csv"))
    path.write_text("broken")
    client.log_artifact(prediction["run_id"], str(path))
    with pytest.raises(DataValidationError, match="schema"):
        monitor_predictions(directory, workspace, MODEL)
    assert dashboard.load_dashboard(workspace, MODEL).state == "monitoring_failed"


@pytest.mark.parametrize("mutation", [
    {"run_id": "other"}, {"model_name": "other"}, {"model_version": "2"}, {"train_sha256": "other"},
    {"raw_rows": True}, {"unique_rows": -1}, {"duplicate_rows_removed": 0}, {"minimum_rows_per_dataset": 1},
    {"performance_evaluated": True}, {"status": "no_drift_detected"}, {"column_scores": {"x": 1}},
    {"run_limit_reached": "invalid"}, {"max_runs": True},
    {"as_of_request_time": "bad"}, {"as_of_request_time": "2018-01-01T00:00:00"},
    {"as_of_request_time": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
])
def test_invalid_summaries_cannot_display_a_successful_verdict(mutation) -> None:
    summary = summary_for()
    summary.update(mutation)
    with pytest.raises(DataValidationError, match="summary"):
        dashboard._validate_summary(summary, "a" * 32, MODEL, "1", "b" * 64)


@pytest.mark.parametrize("mutation", [
    {"column_scores": {"x": 0}}, {"column_scores": {name: float("nan") for name in MONITORED_COLUMNS}},
    {"drifted_columns": ["month_sin"]}, {"html_artifact": "runs:/different/drift_report.html"},
    {"status": "unsupported"},
])
def test_inconsistent_evaluated_summaries_are_rejected(mutation) -> None:
    summary = summary_for("no_drift_detected")
    summary.update(mutation)
    with pytest.raises(DataValidationError):
        dashboard._validate_summary(summary, "a" * 32, MODEL, "1", "b" * 64)


@pytest.mark.parametrize("field", ["run_id", "run_limit_reached", "max_runs"])
def test_missing_summary_fields_are_rejected(field) -> None:
    summary = summary_for()
    del summary[field]
    with pytest.raises(DataValidationError):
        dashboard._validate_summary(summary, "a" * 32, MODEL, "1", "b" * 64)


@pytest.mark.parametrize("state,message", [
    ("no_store", "No saved project results"), ("no_model", "No approved prediction model"),
    ("no_snapshot", "No monitoring snapshot"), ("monitoring_running", "being generated"),
    ("monitoring_failed", "latest monitoring attempt failed"),
])
def test_dashboard_explains_empty_or_unfinished_states(monkeypatch, state, message) -> None:
    monkeypatch.setattr(dashboard, "load_dashboard", lambda *args: dashboard.DashboardView(MODEL, None, state))
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception and not app.metric
    assert message in (app.error[0].value if state == "monitoring_failed" else app.info[0].value)


@pytest.mark.parametrize("state", ["insufficient_data", "distribution_changed", "no_drift_detected"])
def test_dashboard_renders_counts_status_and_report_then_refreshes_saved_results(monkeypatch, state) -> None:
    summary = summary_for(state)
    summary["run_limit_reached"] = True
    html = None if state == "insufficient_data" else "<html><body>Test report</body></html>"
    loads = []

    def load(*args):
        loads.append(True)
        return dashboard.DashboardView(MODEL, "1", state, deepcopy(summary), html, stale=True)

    monkeypatch.setattr(dashboard, "load_dashboard", load)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.metric[0].value == str(summary["unique_rows"])
    assert any("24 hours old" in message.value for message in app.warning)
    assert any("newest 100" in message.value for message in app.warning)
    if state == "insufficient_data":
        assert any("Not enough data" in message.value for message in app.warning)
        assert not app.dataframe and not app.get("iframe")
    else:
        assert len(app.dataframe[0].value) == len(MONITORED_COLUMNS) and len(app.get("iframe")) == 1
    app.button(key="refresh").click().run()
    assert len(loads) == 2 and not app.exception


def test_dashboard_handles_storage_errors_without_a_false_drift_verdict(monkeypatch) -> None:
    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(dashboard, "load_dashboard", fail)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception and not app.metric
    assert "results are unavailable" in app.error[0].value
