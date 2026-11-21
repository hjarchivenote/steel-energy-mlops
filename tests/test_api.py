import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import UUID

from fastapi.testclient import TestClient
from filelock import Timeout
from mlflow.exceptions import MlflowException
import pytest

from steel_energy_mlops import api, cli, prediction
from steel_energy_mlops.api import create_app
from steel_energy_mlops.cli import main
from steel_energy_mlops.tracking import champion_version, store_client
from steel_energy_mlops.training import train_models
from tests.helpers import make_training_data

STAMPS = ["2019-01-07T10:00:00", "2019-01-07T10:15:00"]


def audit_records(workspace: Path):
    return [json.loads(line) for line in (workspace / "http_requests.jsonl").read_text().splitlines()]


def fake_success(timestamps, workspace, model_name, request_id):
    return {"request_id": request_id, "run_id": "fake-run", "model_name": model_name, "model_version": "1",
            "promotion_policy": "legacy-v1", "feature_contract": "calendar-v1-nonnegative-kwh", "unit": "kWh",
            "predictions": [{"timestamp": stamp.isoformat(), "usage_kwh": 12.0} for stamp in timestamps]}


@pytest.fixture(scope="module")
def api_sessions(tmp_path_factory):
    sessions = []
    for name in ("api-a", "api-b"):
        root = tmp_path_factory.mktemp(name)
        directory = make_training_data(root)
        workspace = root / "tracking"
        trained = train_models(directory, workspace, name, families=("ridge",))
        assert trained["promoted"]
        # Serving must work without any prepared train/validation/final test CSVs.
        shutil.rmtree(directory)
        sessions.append({"workspace": workspace, "name": name, "trained": trained})
    return sessions


def test_health_does_not_require_a_model_or_create_a_tracking_database(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        result = client.get("/health", headers={"X-Request-ID": "untrusted-client-value"})
    assert result.status_code == 200 and result.json() == {"status": "ok"}
    request_id = result.headers["X-Request-ID"]
    assert UUID(hex=request_id).hex == request_id
    assert not (tmp_path / "mlflow.db").exists()
    record, = audit_records(tmp_path)
    assert record["request_id"] == request_id and record["status_code"] == 200
    assert record["run_id"] is None and record["model_version"] is None
    assert record["duration_ms"] >= 0 and record["timestamp_utc"].endswith("+00:00")


def test_ready_loads_the_model_without_creating_runs(api_sessions) -> None:
    session = api_sessions[0]
    tracker = store_client(session["workspace"])
    experiments = [item.experiment_id for item in tracker.search_experiments()]
    before = len(tracker.search_runs(experiments))
    with TestClient(create_app(session["workspace"], session["name"])) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready" and body["model_name"] == session["name"]
    assert body["promotion_policy"] == "calendar-reference-v2" and body["unit"] == "kWh"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert len(tracker.search_runs(experiments)) == before


def test_real_http_prediction_matches_mlflow_artifacts_and_audit_record(api_sessions, monkeypatch) -> None:
    session = api_sessions[0]
    loaded = []
    original = prediction.load_model

    def load_once(uri):
        loaded.append(uri)
        return original(uri)

    monkeypatch.setattr(prediction, "load_model", load_once)
    with TestClient(create_app(session["workspace"], session["name"])) as client:
        response = client.post("/predict", json={"timestamps": STAMPS})
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["unit"] == "kWh" and body["model_name"] == session["name"]
    assert body["model_version"] == session["trained"]["candidate_version"]
    assert loaded == [f"models:/{session['name']}/{body['model_version']}"]
    assert [row["timestamp"] for row in body["predictions"]] == STAMPS
    assert all(row["usage_kwh"] >= 0 for row in body["predictions"])
    tracker = store_client(session["workspace"])
    run = tracker.get_run(body["run_id"])
    assert run.info.status == "FINISHED" and run.data.tags["http_request_id"] == body["request_id"]
    for filename, expected in (("response.json", body),
                               ("request.json", {"timestamps": STAMPS, "request_id": body["request_id"]})):
        path = tracker.download_artifacts(body["run_id"], filename)
        assert json.loads(Path(path).read_text()) == expected
    record = next(item for item in audit_records(session["workspace"]) if item["request_id"] == body["request_id"])
    assert record["status_code"] == 200 and record["run_id"] == body["run_id"]
    assert record["model_version"] == body["model_version"] and record["error_code"] is None


def test_maximum_batch_and_canonicalized_timestamps_are_supported(api_sessions) -> None:
    session = api_sessions[0]
    with TestClient(create_app(session["workspace"], session["name"])) as client:
        result = client.post("/predict", json={"timestamps": ["2019-01-07T10:15"] * 1000})
    assert result.status_code == 200 and len(result.json()["predictions"]) == 1000
    assert {row["timestamp"] for row in result.json()["predictions"]} == {"2019-01-07T10:15:00"}


@pytest.mark.parametrize("body", [
    None, {}, [], {"timestamps": []}, {"timestamps": STAMPS * 501}, {"timestamps": STAMPS[0]},
    {"timestamps": [None]}, {"timestamps": [1234567890]}, {"timestamps": [True]},
    {"timestamps": ["2019-01-07"]}, {"timestamps": ["invalid"]},
    {"timestamps": ["2019-02-30T10:00:00"]}, {"timestamps": ["2019-01-07T10:00:00Z"]},
    {"timestamps": ["2019-01-07T10:00:00+09:00"]}, {"timestamps": ["2019-01-07T10:14:00"]},
    {"timestamps": ["2019-01-07T10:15:01"]}, {"timestamps": ["2019-01-07T10:15:00.000001"]},
    {"timestamps": STAMPS, "Usage_kWh": 10}, {"timestamps": STAMPS, "model_name": "../other"},
])
def test_invalid_requests_fail_before_model_access_and_are_logged(tmp_path: Path, body) -> None:
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 422
    error = response.json()
    assert error["request_id"] == response.headers["X-Request-ID"]
    assert error["error"]["code"] == "invalid_request" and error["error"]["issues"]
    assert not (tmp_path / "mlflow.db").exists()
    record, = audit_records(tmp_path)
    assert record["status_code"] == 422 and record["error_code"] == "invalid_request"


def test_malformed_json_and_excessive_validation_issues(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        malformed = client.post("/predict", content=b'{"timestamps":', headers={"Content-Type": "application/json"})
        many = client.post("/predict", json={"timestamps": ["invalid"] * 30})
    assert malformed.status_code == many.status_code == 422
    assert len(many.json()["error"]["issues"]) == 20


@pytest.mark.parametrize("registered", [False, True])
def test_unavailable_champion_returns_503_but_liveness_still_works(tmp_path: Path, registered) -> None:
    if registered:
        store_client(tmp_path).create_registered_model("unavailable-model")
    with TestClient(create_app(tmp_path, "unavailable-model")) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        prediction_result = client.post("/predict", json={"timestamps": STAMPS})
    assert ready.status_code == prediction_result.status_code == 503
    assert ready.json()["error"]["code"] == "model_unavailable"
    assert prediction_result.json()["error"]["code"] == "prediction_unavailable"


def test_legacy_policy_is_disclosed_without_recertifying_the_model(api_sessions) -> None:
    session = api_sessions[0]
    tracker = store_client(session["workspace"])
    version = champion_version(tracker, session["name"])
    original = version.tags["promotion_policy"]
    tracker.delete_model_version_tag(session["name"], version.version, "promotion_policy")
    try:
        with TestClient(create_app(session["workspace"], session["name"])) as client:
            ready = client.get("/ready")
            predicted = client.post("/predict", json={"timestamps": STAMPS})
        assert ready.json()["promotion_policy"] == predicted.json()["promotion_policy"] == "legacy-v1"
    finally:
        tracker.set_model_version_tag(session["name"], version.version, "promotion_policy", original)


def test_incompatible_model_is_not_ready_and_cannot_predict(api_sessions) -> None:
    session = api_sessions[0]
    tracker = store_client(session["workspace"])
    version = champion_version(tracker, session["name"])
    original = version.tags["feature_contract"]
    tracker.set_model_version_tag(session["name"], version.version, "feature_contract", "incompatible")
    try:
        with TestClient(create_app(session["workspace"], session["name"])) as client:
            assert client.get("/ready").status_code == 503
            assert client.post("/predict", json={"timestamps": STAMPS}).status_code == 503
    finally:
        tracker.set_model_version_tag(session["name"], version.version, "feature_contract", original)


@pytest.mark.parametrize("operation,path", [("model_status", "/ready"), ("predict_consumption", "/predict")])
@pytest.mark.parametrize("failure", [OSError("/private/local/path"), MlflowException("store unavailable")])
def test_backend_errors_are_recorded_without_leaking_internal_details(
    tmp_path, monkeypatch, operation, path, failure
) -> None:
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api, operation, fail)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(path) if path == "/ready" else client.post(path, json={"timestamps": STAMPS})
    assert response.status_code == 503 and str(failure) not in response.text
    assert audit_records(tmp_path)[-1]["status_code"] == 503


def test_unloadable_model_is_not_reported_as_ready(api_sessions, monkeypatch) -> None:
    session = api_sessions[0]
    monkeypatch.setattr(prediction, "load_model", lambda uri: None)
    with TestClient(create_app(session["workspace"], session["name"])) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"timestamps": STAMPS}).status_code == 503


@pytest.mark.parametrize("failure", [RuntimeError("unexpected /private/path"), "invalid-response"])
def test_unexpected_failures_and_response_contract_errors_return_recorded_500(tmp_path, monkeypatch, failure) -> None:
    def fail(*args, **kwargs):
        if isinstance(failure, RuntimeError):
            raise failure
        result = fake_success(*args, **kwargs)
        result["predictions"][0]["usage_kwh"] = float("nan")
        return result

    monkeypatch.setattr(api, "predict_consumption", fail)
    with TestClient(create_app(tmp_path), raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"timestamps": STAMPS})
    assert response.status_code == 500 and response.json()["error"]["code"] == "internal_error"
    assert "/private/path" not in response.text
    assert audit_records(tmp_path)[-1]["status_code"] == 500


@pytest.mark.parametrize("failure", [OSError("audit unavailable"), Timeout("audit.lock")])
def test_audit_storage_failure_cannot_return_success(tmp_path, monkeypatch, failure) -> None:
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api, "append_audit", fail)
    monkeypatch.setattr(api, "predict_consumption", fake_success)
    with TestClient(create_app(tmp_path)) as client:
        result = client.post("/predict", json={"timestamps": STAMPS})
    assert result.status_code == 503 and result.json()["error"]["code"] == "audit_unavailable"
    assert result.json()["request_id"] == result.headers["X-Request-ID"]


def test_concurrent_requests_keep_model_stores_and_request_records_separate(api_sessions) -> None:
    left, right = api_sessions
    with TestClient(create_app(left["workspace"], left["name"])) as a:
        with TestClient(create_app(right["workspace"], right["name"])) as b:
            requests = [(a, left, STAMPS), (b, right, STAMPS[::-1]), (a, left, STAMPS[::-1]), (b, right, STAMPS)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(client.post, "/predict", json={"timestamps": stamps})
                           for client, _, stamps in requests]
                responses = [future.result(timeout=30) for future in futures]
    assert all(response.status_code == 200 for response in responses)
    assert len({response.json()["run_id"] for response in responses}) == 4
    assert len({response.headers["X-Request-ID"] for response in responses}) == 4
    for (_, session, stamps), response in zip(requests, responses, strict=True):
        body = response.json()
        assert body["model_name"] == session["name"]
        assert [point["timestamp"] for point in body["predictions"]] == stamps
        tracker = store_client(session["workspace"])
        assert tracker.get_run(body["run_id"]).data.tags["http_request_id"] == body["request_id"]
        assert any(record["request_id"] == body["request_id"] for record in audit_records(session["workspace"]))


def test_liveness_remains_responsive_during_inference(tmp_path: Path, monkeypatch) -> None:
    entered, release = Event(), Event()

    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return fake_success(*args, **kwargs)

    monkeypatch.setattr(api, "predict_consumption", blocking)
    with TestClient(create_app(tmp_path)) as client, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.post, "/predict", json={"timestamps": STAMPS})
        try:
            assert entered.wait(timeout=5)
            assert client.get("/health").status_code == 200
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200


def test_docs_unknown_routes_and_methods_have_consistent_responses(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert {"/health", "/ready", "/predict"} <= set(schema["paths"])
        assert {"200", "422", "500", "503"} <= set(schema["paths"]["/predict"]["post"]["responses"])
        field = schema["components"]["schemas"]["PredictionRequest"]["properties"]["timestamps"]
        assert field["minItems"] == 1 and field["maxItems"] == 1000
        missing = client.get("/missing")
        wrong_method = client.get("/predict")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "http_error"
    assert wrong_method.status_code == 405 and wrong_method.headers["allow"] == "POST"


def test_serve_cli_passes_server_configuration_without_printing_prediction_json(tmp_path, monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    assert main(["serve", "--workspace", str(tmp_path), "--model-name", "api-test", "--port", "8123"]) == 0
    assert calls[0][0].title == "Steel Energy Prediction API"
    assert calls[0][1] == {"host": "127.0.0.1", "port": 8123, "workers": 1}
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("port", ["0", "65536", "text"])
def test_serve_cli_rejects_invalid_ports(port) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--port", port])
    assert exc.value.code == 2


def test_serve_cli_reports_invalid_configuration(capsys) -> None:
    assert main(["serve", "--model-name", "../bad"]) == 2
    assert "API startup failed:" in capsys.readouterr().err
