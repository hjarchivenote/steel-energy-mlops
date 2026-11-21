from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

from fastapi.testclient import TestClient
from filelock import FileLock
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from steel_energy_mlops import api, forecast_models as models, forecast_prediction as prediction
from steel_energy_mlops.api import create_app
from steel_energy_mlops.data import DataValidationError, ElectricalReadings, EnergyRecord, load_dataset
from steel_energy_mlops.forecast_inputs import AvailableObservation, build_forecast_inputs
from steel_energy_mlops.forecast_routing import build_forecast_routes
from steel_energy_mlops.tracking import champion_version, store_client
from steel_energy_mlops.training_data import load_training_data
from tests.helpers import make_training_data


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("forecast-pair")
    directory = make_training_data(root)
    data = load_training_data(directory)
    source = root / "training_source.csv"
    observations = tuple(AvailableObservation(row, row.timestamp + timedelta(minutes=15))
                         for row in load_dataset(source).records if row.timestamp.hour or row.timestamp.minute)
    features = np.asarray([row.values for row in build_forecast_inputs(data.train.timestamps, observations)])
    mask = np.asarray([stamp.hour != 0 or stamp.minute != 0 for stamp in data.train.timestamps])
    history = Pipeline([("model", HistGradientBoostingRegressor(**models.FROZEN_HISTORY_PARAMETERS))])
    calendar = Pipeline([("model", RandomForestRegressor(**models.FROZEN_CALENDAR_PARAMETERS))])
    with threadpool_limits(limits=1):
        history.fit(features[mask], data.train.target[mask])
        calendar.fit(data.train.features[mask], data.train.target[mask])
    evidence = json.loads(Path("experiments/history_15min/validation.json").read_text())
    evidence["run_id"] = "a" * 32  # Synthetic fixture; no real-data model or registry is used by these tests.
    evidence["plan"]["train_sha256"] = data.train.sha256
    evidence["plan"]["context_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence["validation_sha256"] = data.validation.sha256
    evidence["fitted_label_rows"] = int(mask.sum())
    workspace = root / "tracking"
    cutoff = data.train.timestamps[-1].isoformat()
    published = models.publish_forecast_pair(history, calendar, features[mask][:3], evidence, workspace,
                                             training_cutoff=cutoff)
    shutil.rmtree(directory)  # Every serving test runs without prepared train/validation/final test CSVs.
    return {"workspace": workspace, "history": history, "calendar": calendar, "example": features[mask][:3],
            "evidence": evidence, "published": published, "cutoff": cutoff}


def observation_payload(stamp="2018-01-03T11:30:00", arrived="2018-01-03T11:45:00"):
    return {"timestamp": stamp, "available_at": arrived, "usage_kwh": 20., "lagging_reactive_kvarh": 2.,
            "leading_reactive_kvarh": 1., "lagging_power_factor_percent": 80., "leading_power_factor_percent": 90.}


def request_body():
    return {"timestamps": ["2018-01-03T12:00:00", "2018-01-03T00:00:00", "2018-01-04T12:00:00",
                           "2018-01-03T12:00:00"], "observations": [observation_payload()]}


def test_saved_pair_serves_real_http_predictions_and_correlated_artifacts(pair):
    workspace = pair["workspace"]
    body = request_body()
    with TestClient(create_app(workspace)) as client:
        ready = client.get("/forecast/ready")
        assert ready.status_code == 200
        response = client.post("/forecast", json=body)
    assert response.status_code == 200, response.text
    output = response.json()
    assert output["model_version"] == pair["published"]["model_version"]
    assert output["evidence_run_id"] == "a" * 32 and output["availability_is_assumed"] is True
    assert output["request_id"] == response.headers["X-Request-ID"]
    rows = output["predictions"]
    assert [row["reason"] for row in rows] == ["history_ready", "midnight_target", "stale_history", "history_ready"]
    assert rows[0] == rows[3] and rows[0]["source_timestamp"] == "2018-01-03T11:30:00"
    assert rows[1]["source_timestamp"] is None and rows[2]["history_age_minutes"] is None
    stamp = datetime.fromisoformat(body["observations"][0]["timestamp"])
    history = (AvailableObservation(EnergyRecord(stamp, 20., ElectricalReadings(2., 1., 80., 90.)),
                                    datetime.fromisoformat(body["observations"][0]["available_at"])),)
    routes = build_forecast_routes(tuple(datetime.fromisoformat(stamp) for stamp in body["timestamps"]), history)
    with threadpool_limits(limits=1):
        for route, actual in zip(routes, rows, strict=True):
            model, values = ((pair["history"], route.history.values) if route.history is not None
                             else (pair["calendar"], route.calendar_values))
            expected = max(float(model.predict(np.asarray([values]))[0]), 0)
            assert actual["usage_kwh"] == pytest.approx(expected, abs=1e-12)
    tracker = store_client(workspace)
    saved = Path(tracker.download_artifacts(output["run_id"], "response.json"))
    assert json.loads(saved.read_text()) == output
    inputs = json.loads(Path(tracker.download_artifacts(output["run_id"], "inputs.json")).read_text())["rows"]
    assert [len(row["features"]) for row in inputs] == [14, 7, 7, 14]
    record = json.loads((workspace / "http_requests.jsonl").read_text().splitlines()[-1])
    assert record["run_id"] == output["run_id"] and record["model_name"] == models.DEFAULT_FORECAST_MODEL
    run = tracker.get_run(output["run_id"])
    assert run.data.tags["http_request_id"] == output["request_id"] and run.info.status == "FINISHED"
    assert tracker.get_experiment(run.info.experiment_id).name == "steel-energy-forecast-prediction"


def test_missing_and_delayed_history_falls_back_without_losing_batch_rows(pair):
    with TestClient(create_app(pair["workspace"])) as client:
        response = client.post("/forecast", json={"timestamps": ["2018-01-03T12:00:00"] * 1000})
        late_body = {"timestamps": ["2018-01-03T12:00:00"],
                     "observations": [observation_payload(arrived="2018-01-03T12:00:00")]}
        late = client.post("/forecast", json=late_body)
    assert response.status_code == 200 and len(response.json()["predictions"]) == 1000
    assert all(row["reason"] == "missing_history" for row in response.json()["predictions"])
    assert late.json()["predictions"][0]["reason"] == "missing_history"


@pytest.mark.parametrize("failure", ["extra", "empty", "timezone", "grid", "numeric-string", "boolean",
                                     "negative", "factor", "before", "duplicate", "nested-extra", "origin"])
def test_invalid_forecast_requests_are_422_before_model_loading(tmp_path, monkeypatch, failure):
    body = request_body()
    row = body["observations"][0]
    if failure == "extra":
        body["model_path"] = "/private/model"
    elif failure == "empty":
        body["timestamps"] = []
    elif failure == "timezone":
        row["available_at"] += "Z"
    elif failure == "grid":
        row["timestamp"] = "2018-01-03T11:31:00"
    elif failure == "numeric-string":
        row["usage_kwh"] = "20"
    elif failure == "boolean":
        row["usage_kwh"] = True
    elif failure == "negative":
        row["usage_kwh"] = -1
    elif failure == "factor":
        row["leading_power_factor_percent"] = 101
    elif failure == "before":
        row["available_at"] = "2018-01-03T11:29:00"
    elif failure == "duplicate":
        body["observations"].append(deepcopy(row))
    elif failure == "nested-extra":
        row["co2"] = 0.01
    else:
        body["timestamps"] = ["0001-01-01T00:00:00"]
    monkeypatch.setattr(api, "predict_forecast", lambda *args, **kwargs: pytest.fail("Invalid request reached model"))
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/forecast", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert not (tmp_path / "mlflow.db").exists()


def test_missing_forecast_pair_and_storage_failures_are_reported_as_503(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/forecast/ready").status_code == 503
        assert client.post("/forecast", json=request_body()).status_code == 503
        monkeypatch.setattr(api, "forecast_status", lambda *args: (_ for _ in ()).throw(OSError("private path")))
        assert client.get("/forecast/ready").status_code == 503


@pytest.mark.parametrize("bad", [np.asarray([np.nan]), np.asarray([1., 2.])])
def test_invalid_model_outputs_fail_without_a_prediction_run(pair, monkeypatch, bad):
    history, calendar, metadata = models.load_forecast_pair(pair["workspace"])
    monkeypatch.setattr(history, "predict", lambda values: bad)
    monkeypatch.setattr(prediction, "load_forecast_pair", lambda *args: (history, calendar, metadata))
    body = {"timestamps": ["2018-01-03T12:00:00"], "observations": [observation_payload()]}
    with TestClient(create_app(pair["workspace"])) as client:
        response = client.post("/forecast", json=body)
    assert response.status_code == 503


def test_frozen_nonnegative_clipping_is_preserved(pair, monkeypatch):
    history, calendar, metadata = models.load_forecast_pair(pair["workspace"])
    monkeypatch.setattr(history, "predict", lambda values: np.asarray([-1.]))
    monkeypatch.setattr(prediction, "load_forecast_pair", lambda *args: (history, calendar, metadata))
    body = {"timestamps": ["2018-01-03T12:00:00"], "observations": [observation_payload()]}
    with TestClient(create_app(pair["workspace"])) as client:
        response = client.post("/forecast", json=body)
    assert response.status_code == 200 and response.json()["predictions"][0]["usage_kwh"] == 0.


def test_publication_cannot_overwrite_a_champion_or_the_calendar_name(pair, tmp_path):
    with pytest.raises(DataValidationError, match="already has a champion"):
        models.publish_forecast_pair(pair["history"], pair["calendar"], pair["example"], pair["evidence"],
                                     pair["workspace"], training_cutoff=pair["cutoff"])
    with pytest.raises(DataValidationError, match="separate registered"):
        models.publish_forecast_pair(pair["history"], pair["calendar"], pair["example"], pair["evidence"],
                                     tmp_path, "steel-energy-calendar", training_cutoff=pair["cutoff"])
    with FileLock(tmp_path / "training.lock"), pytest.raises(DataValidationError, match="workspace lock"):
        models.publish_forecast_pair(pair["history"], pair["calendar"], pair["example"], pair["evidence"],
                                     tmp_path, training_cutoff=pair["cutoff"])


@pytest.mark.parametrize("failure", ["gate", "hash", "configuration", "width", "example"])
def test_publication_rejects_invalid_evidence_or_model_contract(pair, tmp_path, failure):
    evidence = deepcopy(pair["evidence"])
    example = pair["example"]
    history = pair["history"]
    if failure == "gate":
        evidence["checks"]["calendar_mae"] = False
    elif failure == "hash":
        evidence["plan"]["train_sha256"] = "wrong"
    elif failure == "configuration":
        history = deepcopy(history)
        history.set_params(model__learning_rate=0.5)
    elif failure == "width":
        history = pair["calendar"]
    else:
        example = np.ones((1, 7))
    with pytest.raises(DataValidationError):
        models.publish_forecast_pair(history, pair["calendar"], example, evidence, tmp_path,
                                     training_cutoff=pair["cutoff"])


def test_failed_saved_model_round_trip_never_sets_an_alias(pair, tmp_path, monkeypatch):
    monkeypatch.setattr(models.np, "allclose", lambda *args, **kwargs: False)
    with pytest.raises(DataValidationError, match="Saved forecast"):
        models.publish_forecast_pair(pair["history"], pair["calendar"], pair["example"], pair["evidence"],
                                     tmp_path, training_cutoff=pair["cutoff"])
    assert champion_version(store_client(tmp_path), models.DEFAULT_FORECAST_MODEL) is None


@pytest.mark.parametrize("failure", ["checksum", "format", "evidence", "uri", "mixed-role", "unfinished", "cutoff"])
def test_corrupted_pair_fails_before_serving(pair, failure):
    client = store_client(pair["workspace"])
    version = champion_version(client, models.DEFAULT_FORECAST_MODEL)
    location = client.get_run(version.run_id).info.artifact_uri
    path = Path(unquote(urlparse(location).path)) / "forecast_bundle.json"
    original = path.read_bytes()
    old_hash = version.tags["bundle_sha256"]
    manifest = json.loads(original)
    if failure == "format":
        manifest["format_version"] = 99
    elif failure == "evidence":
        manifest["evidence_sha256"] = "0" * 64
    elif failure == "uri":
        manifest["models"]["calendar"] = "models:/some-mutable-alias"
    elif failure == "mixed-role":
        manifest["models"]["calendar"] = manifest["models"]["history"]
    elif failure == "unfinished":
        client.set_terminated(version.run_id, status="FAILED")
    elif failure == "cutoff":
        manifest["train_last_timestamp"] = "not-a-date"
    else:
        manifest["unexpected"] = True
    path.write_text(json.dumps(manifest))
    if failure != "checksum":
        client.set_model_version_tag(version.name, version.version, "bundle_sha256", models._digest(manifest))
    try:
        with pytest.raises(DataValidationError):
            models.load_forecast_pair(pair["workspace"])
    finally:
        path.write_bytes(original)
        client.set_model_version_tag(version.name, version.version, "bundle_sha256", old_hash)
        client.set_terminated(version.run_id, status="FINISHED")


def test_service_bounds_apply_to_non_http_callers(pair):
    with pytest.raises(DataValidationError, match="1-1000"):
        prediction.predict_forecast((), (), pair["workspace"])


def test_pair_cannot_load_an_estimator_from_another_publication(pair, monkeypatch):
    client = store_client(pair["workspace"])
    monkeypatch.setattr(models, "store_client", lambda workspace: client)
    monkeypatch.setattr(client, "get_logged_model", lambda model_id: SimpleNamespace(source_run_id="another-run"))
    with pytest.raises(DataValidationError, match="same publication run"):
        models.load_forecast_pair(pair["workspace"])


def test_forecasts_cannot_predate_training_label_availability(pair):
    with TestClient(create_app(pair["workspace"])) as client:
        response = client.post("/forecast", json={"timestamps": [pair["cutoff"]]})
    assert response.status_code == 422
    assert "training-label availability" in response.json()["error"]["message"]
