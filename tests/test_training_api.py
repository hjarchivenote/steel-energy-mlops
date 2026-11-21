import json
import os
import time
from pathlib import Path
from threading import Event
from uuid import uuid4

from fastapi.testclient import TestClient
from filelock import FileLock
import pytest

from steel_energy_mlops import api, training_jobs
from steel_energy_mlops.api import create_app
from steel_energy_mlops.tracking import champion_version, store_client
from steel_energy_mlops.training import train_models
from steel_energy_mlops.training_jobs import TrainingBusyError, TrainingJobs, UnknownJobError
from tests.helpers import make_training_data

PARAMETERS = {"families": ["ridge"], "forest_trees": 2, "minimum_improvement": 0.01, "seed": 42}


def wait_for_job(client, path):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    pytest.fail("Training job did not finish within 45 seconds.")


def result_for(model_name):
    return {"model_name": model_name, "run_id": "test-run", "promoted": False, "test_evaluated": False}


class FakeProcess:
    def __init__(self, returncode=2):
        self.pid = os.getpid()
        self.code = returncode
        self.release = Event()
        self.terminated = False

    def wait(self):
        assert self.release.wait(timeout=10)
        return self.code

    def terminate(self):
        self.terminated = True
        self.release.set()


@pytest.fixture
def fake_training(tmp_path, monkeypatch):
    directory = make_training_data(tmp_path)
    workspace = tmp_path / "tracking"
    process = FakeProcess()
    commands = []

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return process

    monkeypatch.setattr(training_jobs.subprocess, "Popen", popen)
    yield directory, workspace, process, commands
    process.release.set()


def test_real_training_api_promotes_then_rejects_identical_candidate_and_survives_restart(tmp_path) -> None:
    directory = make_training_data(tmp_path)
    (directory / "test.csv").unlink()
    workspace = tmp_path / "tracking"
    with TestClient(create_app(workspace, "http-training", directory)) as client:
        accepted = client.post("/train", json=PARAMETERS)
        assert accepted.status_code == 202
        first = accepted.json()
        assert first["status"] == "running" and first["request_id"] == accepted.headers["X-Request-ID"]
        assert first["training_request_id"] == first["request_id"]
        assert accepted.headers["Location"] == first["status_url"]
        assert client.post("/train", json=PARAMETERS).status_code == 409
        done = wait_for_job(client, first["status_url"])
        assert done["status"] == "succeeded" and done["result"]["promoted"] is True
        assert done["request_id"] != done["training_request_id"]
        assert done["result"]["test_evaluated"] is False
        snapshot = workspace / "training_jobs" / done["job_id"] / "data"
        assert sorted(path.name for path in snapshot.iterdir()) == ["manifest.json", "train.csv", "validation.csv"]
        tracker = store_client(workspace)
        run = tracker.get_run(done["result"]["run_id"])
        assert run.data.tags["http_training_job_id"] == done["job_id"]
        assert run.data.tags["http_training_request_id"] == first["training_request_id"]
        artifact = tracker.download_artifacts(run.info.run_id, "http_training_request.json")
        assert json.loads(Path(artifact).read_text())["parameters"] == PARAMETERS

        second = client.post("/train", json=PARAMETERS).json()
        # Training runs outside the web process: an existing model remains available during training.
        assert client.get("/health").status_code == 200
        prediction = client.post("/predict", json={"timestamps": ["2019-01-07T10:00:00"]})
        assert prediction.status_code == 200
        rejected = wait_for_job(client, second["status_url"])
        assert rejected["status"] == "succeeded" and rejected["result"]["promoted"] is False
        assert rejected["result"]["reason"] == "incumbent_mae_not_improved"
        assert str(champion_version(tracker, "http-training").version) == done["result"]["candidate_version"]
    # Completed records remain queryable after constructing a fresh app/manager.
    with TestClient(create_app(workspace, "http-training", directory)) as restarted:
        restored = restarted.get(first["status_url"]).json()
        assert restored["result"] == done["result"] and restored["status"] == "succeeded"
    records = [json.loads(line) for line in (workspace / "http_requests.jsonl").read_text().splitlines()]
    admission = next(row for row in records if row["request_id"] == first["training_request_id"])
    assert admission["training_job_id"] == first["job_id"] and admission["status_code"] == 202
    assert any(row["run_id"] == done["result"]["run_id"] for row in records)


@pytest.mark.parametrize("body", [
    {"families": []}, {"families": ["unknown"]}, {"families": ["ridge", "ridge"]},
    {"forest_trees": 0}, {"forest_trees": 501}, {"forest_trees": "100"}, {"forest_trees": True},
    {"minimum_improvement": 0.009}, {"minimum_improvement": 1}, {"minimum_improvement": "0.01"},
    {"seed": -1}, {"seed": 2 ** 32}, {"seed": False}, {"data": "../private"},
    {"workspace": "../private"}, {"model_name": "other"}, {"Usage_kWh": 10},
])
def test_invalid_training_requests_are_rejected_before_data_access(tmp_path, body) -> None:
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/train", json=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"
    assert not list((tmp_path / "training_jobs").glob("*/job.json"))


def test_missing_inputs_and_unknown_jobs_are_reported_without_paths(tmp_path) -> None:
    with TestClient(create_app(tmp_path, training_data=tmp_path / "missing")) as client:
        response = client.post("/train", json={})
        assert response.status_code == 503 and "/missing" not in response.text
        for identifier in ("not-a-job", uuid4().hex):
            missing = client.get(f"/train/{identifier}")
            assert missing.status_code == 404 and missing.json()["error"]["code"] == "job_not_found"


def test_process_failure_releases_admission_and_records_failed_status(fake_training) -> None:
    directory, workspace, process, commands = fake_training
    with TestClient(create_app(workspace, "fake-model", directory)) as client:
        accepted = client.post("/train", json=PARAMETERS).json()
        process.release.set()
        failed = wait_for_job(client, accepted["status_url"])
        assert failed["status"] == "failed" and failed["error"]["exit_code"] == 2
    command, options = commands[0]
    assert command[:4] == [training_jobs.sys.executable, "-m", "steel_energy_mlops", "train"]
    assert options["start_new_session"] is True
    assert options["env"]["STEEL_ENERGY_TRAINING_JOB_ID"] == accepted["job_id"]
    with FileLock(workspace / "api-training.lock", timeout=0):
        pass


def test_snapshot_is_fixed_and_duplicate_managers_are_refused(fake_training) -> None:
    directory, workspace, process, _ = fake_training
    a, b = TrainingJobs(workspace, directory, "fake-model"), TrainingJobs(workspace, directory, "fake-model")
    job = a.start(PARAMETERS, uuid4().hex)
    try:
        (directory / "train.csv").write_text("changed after admission")
        snapshot = workspace / "training_jobs" / job["job_id"] / "data" / "train.csv"
        assert snapshot.read_text() != "changed after admission"
        with pytest.raises(TrainingBusyError):
            b.start(PARAMETERS, uuid4().hex)
        assert b.get(job["job_id"])["status"] == "running"
    finally:
        process.release.set()
        a.close()
        b.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        a.start(PARAMETERS, uuid4().hex)


def test_cli_training_lock_is_respected_by_admission(fake_training) -> None:
    directory, workspace, _, commands = fake_training
    with TestClient(create_app(workspace, "fake-model", directory)) as client:
        with FileLock(workspace / "training.lock", timeout=0):
            result = client.post("/train", json=PARAMETERS)
        assert result.status_code == 409 and commands == []


def test_audit_failure_returns_recoverable_job_id_for_an_admitted_job(fake_training, monkeypatch) -> None:
    directory, workspace, process, _ = fake_training
    original = api.append_audit

    def fail(*args, **kwargs):
        raise OSError("audit unavailable")

    monkeypatch.setattr(api, "append_audit", fail)
    with TestClient(create_app(workspace, "fake-model", directory)) as client:
        response = client.post("/train", json=PARAMETERS)
        assert response.status_code == 503
        body = response.json()
        assert body["job_id"] and body["status_url"] == f"/train/{body['job_id']}"
        monkeypatch.setattr(api, "append_audit", original)
        process.release.set()
        assert wait_for_job(client, body["status_url"])["status"] == "failed"


def test_subprocess_start_failure_is_reported_and_releases_lock(fake_training, monkeypatch) -> None:
    directory, workspace, _, _ = fake_training

    def fail(*args, **kwargs):
        raise OSError("private launch failure")

    monkeypatch.setattr(training_jobs.subprocess, "Popen", fail)
    with TestClient(create_app(workspace, "fake-model", directory)) as client:
        result = client.post("/train", json=PARAMETERS)
        assert result.status_code == 503 and "private launch" not in result.text
    with FileLock(workspace / "api-training.lock", timeout=0):
        pass


def test_monitor_start_failure_stops_the_unmonitored_child(fake_training, monkeypatch) -> None:
    directory, workspace, process, _ = fake_training

    def fail(*args, **kwargs):
        raise RuntimeError("monitor unavailable")

    monkeypatch.setattr(training_jobs.Thread, "start", fail)
    manager = TrainingJobs(workspace, directory, "fake-model")
    with pytest.raises(RuntimeError, match="monitor unavailable"):
        manager.start(PARAMETERS, uuid4().hex)
    manager.close()
    assert process.terminated


def seed_record(manager, *, pid=None):
    job_id = uuid4().hex
    (manager.root / job_id).mkdir()
    record = {"job_id": job_id, "training_request_id": uuid4().hex, "model_name": "fake-model", "pid": pid,
              "status": "running", "parameters": PARAMETERS, "created_at": training_jobs._now(),
              "result": None, "error": None, "train_sha256": "a" * 64, "validation_sha256": "b" * 64}
    manager._save(record)
    return record


@pytest.mark.parametrize("payload", [None, [], {}, {"model_name": "fake-model", "run_id": 5},
                                     {**result_for("fake-model"), "promoted": "false"},
                                     {**result_for("fake-model"), "test_evaluated": True}])
def test_recovery_marks_missing_or_invalid_output_as_interrupted(tmp_path, payload) -> None:
    manager = TrainingJobs(tmp_path, tmp_path / "data", "fake-model")
    record = seed_record(manager)
    if payload is not None:
        (manager.root / record["job_id"] / "result.json").write_text(json.dumps(payload))
    recovered = manager.get(record["job_id"])
    assert recovered["status"] == "interrupted" and recovered["error"]["code"] == "training_interrupted"


def test_recovery_recognizes_completed_output_even_with_a_different_configured_model(tmp_path) -> None:
    manager = TrainingJobs(tmp_path, tmp_path / "data", "other-model")
    record = seed_record(manager)
    (manager.root / record["job_id"] / "result.json").write_text(json.dumps(result_for("fake-model")))
    assert manager.get(record["job_id"])["status"] == "succeeded"


def test_orphaned_live_worker_prevents_new_admission(tmp_path) -> None:
    directory = make_training_data(tmp_path)
    manager = TrainingJobs(tmp_path / "tracking", directory, "fake-model")
    seed_record(manager, pid=os.getpid())
    with pytest.raises(TrainingBusyError, match="previous training"):
        manager.start(PARAMETERS, uuid4().hex)


@pytest.mark.parametrize("content", ["invalid json", "[]", "{}"])
def test_corrupt_job_records_are_reported_as_storage_errors(tmp_path, content) -> None:
    manager = TrainingJobs(tmp_path, tmp_path / "data", "fake-model")
    record = seed_record(manager)
    (manager.root / record["job_id"] / "job.json").write_text(content)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/train/{record['job_id']}")
    assert response.status_code == 503 and response.json()["error"]["code"] == "job_storage_unavailable"
    with pytest.raises(UnknownJobError):
        manager.get("../private")


@pytest.mark.parametrize("failure,expected", [(ProcessLookupError(), False), (PermissionError(), True)])
def test_pid_probe_distinguishes_missing_and_inaccessible_processes(monkeypatch, failure, expected) -> None:
    def fail(*args):
        raise failure

    monkeypatch.setattr(training_jobs.os, "kill", fail)
    assert training_jobs._alive(42) is expected


def test_monitor_failure_is_recoverable_and_always_releases_admission(fake_training, monkeypatch) -> None:
    directory, workspace, process, _ = fake_training
    manager = TrainingJobs(workspace, directory, "fake-model")
    original = manager._finish

    def fail(*args):
        raise OSError("temporary status storage failure")

    monkeypatch.setattr(manager, "_finish", fail)
    job = manager.start(PARAMETERS, uuid4().hex)
    process.release.set()
    manager.close()
    monkeypatch.setattr(manager, "_finish", original)
    monkeypatch.setattr(training_jobs, "_alive", lambda pid: False)
    assert manager.get(job["job_id"])["status"] == "interrupted"
    with FileLock(workspace / "api-training.lock", timeout=0):
        pass


def test_training_http_correlation_is_recorded_in_the_parent_process(tmp_path, monkeypatch) -> None:
    directory = make_training_data(tmp_path)
    job_id, request_id = uuid4().hex, uuid4().hex
    monkeypatch.setenv("STEEL_ENERGY_TRAINING_JOB_ID", job_id)
    monkeypatch.setenv("STEEL_ENERGY_TRAINING_REQUEST_ID", request_id)
    result = train_models(directory, tmp_path / "tracking", "tagged-model", families=("ridge",))
    tracker = store_client(tmp_path / "tracking")
    run = tracker.get_run(result["run_id"])
    assert run.data.tags["http_training_job_id"] == job_id
    assert run.data.tags["http_training_request_id"] == request_id
