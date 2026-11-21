from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from filelock import FileLock
import pytest

from steel_energy_mlops import cli, monitoring, monitoring_worker as worker
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.prediction import predict_consumption
from steel_energy_mlops.tracking import store_client
from steel_energy_mlops.training import train_models
from steel_energy_mlops.training_data import load_training_partition
from tests.helpers import make_training_data

MODEL = "automatic-monitoring-test"


@pytest.fixture
def automatic_store(tmp_path):
    directory = make_training_data(tmp_path)
    workspace = tmp_path / "tracking"
    assert train_models(directory, workspace, MODEL, families=("ridge",))["promoted"]
    (directory / "test.csv").unlink()
    (directory / "validation.csv").unlink()
    return directory, workspace, store_client(workspace)


def test_new_predictions_refresh_and_unchanged_inputs_skip_without_reloading_artifacts(automatic_store, monkeypatch):
    directory, workspace, client = automatic_store
    first = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert first["unique_rows"] == 0
    with monkeypatch.context() as patch:
        patch.setattr(monitoring, "_current", lambda *args: pytest.fail("Unchanged inputs must not reload artifacts."))
        skipped = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
        assert skipped == {"status": "skipped_unchanged", "model_name": MODEL, "run_id": first["run_id"]}
    stamps = load_training_partition(directory).timestamps[:2]
    prediction = predict_consumption(stamps, workspace, MODEL)
    updated = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert updated["source_run_ids"] == [prediction["run_id"]] and updated["unique_rows"] == 2
    assert updated["run_id"] != first["run_id"] and updated["status"] == "insufficient_data"
    skipped = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert skipped["status"] == "skipped_unchanged"
    changed_policy = monitoring.monitor_predictions(directory, workspace, MODEL, max_runs=99, only_if_changed=True)
    assert changed_policy["run_id"] != updated["run_id"]
    assert client.get_registered_model(MODEL).aliases["champion"] == 1


def test_manual_and_automatic_monitoring_share_nonblocking_admission(automatic_store):
    directory, workspace, _ = automatic_store
    with FileLock(workspace / "monitoring.lock", timeout=0):
        for automatic in (False, True):
            assert monitoring.monitor_predictions(directory, workspace, MODEL,
                                                  only_if_changed=automatic)["status"] == "skipped_busy"
    assert monitoring.monitor_predictions(directory, workspace, MODEL)["status"] == "insufficient_data"


def test_failed_attempts_back_off_and_manual_repair_can_restore_monitoring(automatic_store):
    directory, workspace, client = automatic_store
    prediction = predict_consumption(load_training_partition(directory).timestamps[:2], workspace, MODEL)
    path = Path(client.download_artifacts(prediction["run_id"], "predictions.csv"))
    original = path.read_bytes()
    path.write_text("broken")
    client.log_artifact(prediction["run_id"], str(path))
    with pytest.raises(DataValidationError, match="schema"):
        monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    waiting = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert waiting["status"] == "retry_wait"
    assert client.get_run(waiting["run_id"]).info.status == "FAILED"
    path.write_bytes(original)
    client.log_artifact(prediction["run_id"], str(path))
    recovered = monitoring.monitor_predictions(directory, workspace, MODEL)
    assert recovered["unique_rows"] == 2
    skipped = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert skipped["run_id"] == recovered["run_id"]


@pytest.mark.parametrize("name,remove", [("train.csv", False), ("manifest.json", False), ("train.csv", True)])
def test_reference_corruption_cannot_be_skipped_as_unchanged_and_restoration_recovers(automatic_store, name, remove):
    directory, workspace, client = automatic_store
    monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    path = directory / name
    original = path.read_bytes()
    if remove:
        path.unlink()
    else:
        path.write_text("broken")
    with pytest.raises((ValueError, OSError)):
        monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    waiting = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert waiting["status"] == "retry_wait" and client.get_run(waiting["run_id"]).info.status == "FAILED"
    path.write_bytes(original)
    restored = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert restored["status"] == "insufficient_data"


def test_deployed_version_change_gets_its_own_snapshot(automatic_store):
    directory, workspace, client = automatic_store
    first = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    current = client.get_model_version(MODEL, "1")
    copied = client.create_model_version(MODEL, current.source, run_id=current.run_id, tags=current.tags)
    client.set_registered_model_alias(MODEL, "champion", copied.version)
    second = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert second["model_version"] == str(copied.version) and second["run_id"] != first["run_id"]


def test_orphaned_running_attempt_is_closed_before_restart_recovers(automatic_store):
    directory, workspace, client = automatic_store
    previous = monitoring.monitor_predictions(directory, workspace, MODEL)
    run = client.get_run(previous["run_id"])
    orphan = client.create_run(run.info.experiment_id, tags={"operation": "request_distribution_monitoring"})
    for name, value in run.data.params.items():
        client.log_param(orphan.info.run_id, name, value)
    recovered = monitoring.monitor_predictions(directory, workspace, MODEL, only_if_changed=True)
    assert client.get_run(orphan.info.run_id).info.status == "KILLED"
    assert recovered["run_id"] != previous["run_id"] and recovered["status"] == "insufficient_data"


@pytest.mark.parametrize("status,age,fingerprint,expected", [
    ("FINISHED", 60, "same", "skipped_unchanged"), ("FINISHED", 86401, "same", None),
    ("FAILED", 60, "same", "retry_wait"), ("FAILED", 301, "same", None), ("KILLED", 60, "same", "retry_wait"),
    ("FINISHED", -60, "same", None), ("FINISHED", 60, "different", None), ("RUNNING", 60, "same", None),
])
def test_refresh_policy_covers_freshness_retries_and_input_changes(status, age, fingerprint, expected):
    now = datetime.now(timezone.utc)
    ended = int((now - timedelta(seconds=age)).timestamp() * 1000)
    previous = SimpleNamespace(info=SimpleNamespace(status=status, start_time=ended, end_time=ended),
                               data=SimpleNamespace(params={"monitor_fingerprint": fingerprint}))
    assert monitoring._refresh_decision(previous, "same", now) == expected
    assert monitoring._refresh_decision(None, "same", now) is None


class StopAfter:
    def __init__(self, polls=1):
        self.polls = polls
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, interval):
        self.polls -= 1
        if self.polls == 0:
            self.set()


@pytest.mark.parametrize("result", [
    {"status": "insufficient_data", "run_id": "test-run"}, {"status": "skipped_unchanged"},
    {"status": "skipped_busy"}, {"status": "retry_wait"},
])
def test_worker_polling_persists_health_and_stops_cleanly(tmp_path, monkeypatch, result):
    writes = []
    original = worker._write_status

    def write(workspace, state):
        original(workspace, state)
        writes.append((state["status"], worker.monitoring_service_healthy(workspace)))

    monkeypatch.setattr(worker, "_write_status", write)
    monkeypatch.setattr(worker, "monitor_predictions", lambda *args, **kwargs: result)
    assert worker.watch_monitoring(tmp_path, tmp_path, MODEL, stop=StopAfter()) == {"status": "stopped"}
    expected = result["status"] if result["status"] != "insufficient_data" else "updated"
    assert writes == [(expected, expected != "retry_wait"), ("stopped", False)]


def test_worker_recovers_from_an_operation_error_on_the_next_poll(tmp_path, monkeypatch):
    calls = []

    def monitor(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("transient artifact failure")
        state = json.loads((tmp_path / "monitoring_service.json").read_text())
        assert state["status"] == "error" and not worker.monitoring_service_healthy(tmp_path)
        return {"status": "skipped_unchanged"}

    monkeypatch.setattr(worker, "monitor_predictions", monitor)
    worker.watch_monitoring(tmp_path, tmp_path, MODEL, interval_seconds=1, stop=StopAfter(2))
    assert len(calls) == 2


def test_worker_installs_and_restores_signal_handlers_for_graceful_shutdown(tmp_path, monkeypatch):
    stop = StopAfter()
    handlers, changes = {}, []

    def register(name, handler):
        changes.append((name, handler))
        handlers[name] = handler
        return "previous-handler"

    def monitor(*args, **kwargs):
        handlers[worker.signal.SIGTERM](worker.signal.SIGTERM, None)
        return {"status": "skipped_unchanged"}

    monkeypatch.setattr(worker, "Event", lambda: stop)
    monkeypatch.setattr(worker.signal, "signal", register)
    monkeypatch.setattr(worker, "monitor_predictions", monitor)
    worker.watch_monitoring(tmp_path, tmp_path, MODEL)
    assert len(changes) == 4 and all(handler == "previous-handler" for _, handler in changes[2:])


@pytest.mark.parametrize("state", [
    None, {}, {"updated_at": "invalid"}, {"updated_at": 0, "status": "updated"},
    {"updated_at": float("nan"), "status": "updated"},
])
def test_missing_corrupt_or_stale_worker_health_is_unhealthy(tmp_path, state):
    if state is not None:
        (tmp_path / "monitoring_service.json").write_text(json.dumps(state))
    assert not worker.monitoring_service_healthy(tmp_path)


@pytest.mark.parametrize("value", [0, True, 3601])
def test_worker_rejects_invalid_poll_intervals(tmp_path, value):
    with pytest.raises(DataValidationError, match="interval"):
        worker.watch_monitoring(tmp_path, tmp_path, MODEL, interval_seconds=value)


def test_monitor_change_detection_requires_a_boolean(tmp_path):
    with pytest.raises(DataValidationError, match="boolean"):
        monitoring.monitor_predictions(tmp_path, tmp_path, MODEL, only_if_changed=1)


def test_cli_can_start_the_worker_and_reports_invalid_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "watch_monitoring", lambda *args: {"status": "stopped"})
    assert cli.main(["watch-monitoring", "--workspace", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "stopped"
    monkeypatch.setattr(cli, "watch_monitoring", worker.watch_monitoring)
    assert cli.main(["watch-monitoring", "--interval-seconds", "0"]) == 2
    assert "Automatic monitoring failed:" in capsys.readouterr().err
