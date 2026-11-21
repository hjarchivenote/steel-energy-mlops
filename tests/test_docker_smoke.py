"""Exercise the shipped image and Compose stack with isolated, generated data and model state."""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import pytest
from mlflow.sklearn import load_model

from steel_energy_mlops.features import calendar_features
from steel_energy_mlops.runtime import export_champion
from steel_energy_mlops.training import train_models
from tests.helpers import make_training_data


def test_compose_build_restore_predict_and_train(tmp_path, pytestconfig, monkeypatch) -> None:
    if not pytestconfig.getoption("--run-docker"):
        pytest.skip("Run with --run-docker to build and check the Compose services.")
    for name in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    directory = make_training_data(tmp_path, record_count=5000)
    (directory / "test.csv").unlink()
    workspace = tmp_path / "tracking"
    assert train_models(directory, workspace, families=("ridge",))["promoted"]
    seeds = tmp_path / "seeds"
    export_champion(workspace, seeds / "champion")
    model = load_model(str(seeds / "champion/model"))
    stamps = [(datetime(2019, 1, 7, 10) + timedelta(minutes=15 * index)).isoformat() for index in range(100)]
    expected = model.predict(np.asarray([calendar_features(datetime.fromisoformat(stamp)) for stamp in stamps]))
    project = "steel-energy-smoke-" + uuid4().hex[:12]
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("COMPOSE_FILE", "COMPOSE_PROJECT_NAME", "COMPOSE_PROFILES")}
    environment.update(STEEL_ENERGY_PREPARED_DATA=str(directory), STEEL_ENERGY_MODEL_SEEDS=str(seeds),
                       STEEL_ENERGY_API_PORT="0", STEEL_ENERGY_MLFLOW_PORT="0", STEEL_ENERGY_DASHBOARD_PORT="0")

    def compose(*arguments, check=True):
        completed = subprocess.run(["docker", "compose", "-p", project, *arguments],
                                   cwd=Path(__file__).resolve().parents[1], env=environment,
                                   text=True, capture_output=True, timeout=600)
        if check:
            assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed

    def request(url, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(url, data=data, headers={"Content-Type": "application/json"}), timeout=20) as response:
            return response.status, json.load(response)

    def wait_for_job(admission):
        job = request(api + admission["status_url"])[1]
        deadline = time.monotonic() + 90
        while job["status"] == "running" and time.monotonic() < deadline:
            time.sleep(1)
            job = request(api + admission["status_url"])[1]
        assert job["status"] == "succeeded", job
        return job

    try:
        compose("up", "--build", "-d", "--wait", "--wait-timeout", "180")
        api = "http://" + compose("port", "api", "8000").stdout.strip()
        ui = "http://" + compose("port", "mlflow", "5000").stdout.strip()
        dashboard = "http://" + compose("port", "dashboard", "8501").stdout.strip()
        with urlopen(dashboard + "/_stcore/health", timeout=20) as response:
            assert response.status == 200
        assert request(api + "/health")[1]["status"] == "ok"
        ready = request(api + "/ready")[1]
        assert ready["model_version"] == "1" and ready["promotion_policy"] == "calendar-reference-v2"
        assert compose("exec", "-T", "api", "id", "-u").stdout.strip() == "10001"
        with urlopen(ui + "/health", timeout=20) as response:
            assert response.status == 200
        prediction = request(api + "/predict", {"timestamps": stamps})[1]
        np.testing.assert_allclose([point["usage_kwh"] for point in prediction["predictions"]], expected,
                                   rtol=0, atol=1e-8)
        artifact = ui + "/get-artifact?" + urlencode({"run_id": prediction["run_id"], "path": "response.json"})
        assert request(artifact)[1] == prediction
        monitored = json.loads(compose("exec", "-T", "api", "steel-energy", "monitor", "--workspace",
                                       "/runtime/mlflow", "--data", "/runtime/data/processed").stdout)
        assert monitored["unique_rows"] == 100 and monitored["html_artifact"]
        assert monitored["status"] in ("distribution_changed", "no_drift_detected")
        report_url = ui + "/get-artifact?" + urlencode({"run_id": monitored["run_id"], "path": "drift_report.html"})
        with urlopen(report_url, timeout=20) as response:
            assert response.status == 200 and b"<html" in response.read().lower()
        dashboard_check = """
import errno
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest
assert os.getuid() == 10001
app = AppTest.from_file('/app/dashboard/app.py').run(timeout=60)
assert not app.exception, app.exception
assert not app.error, app.error
assert app.metric[0].value == '100'
assert len(app.get('iframe')) == 1
try:
    Path('/runtime/dashboard-write-probe').touch()
except OSError as error:
    assert error.errno == errno.EROFS
else:
    raise AssertionError('Dashboard runtime volume must be read-only.')
"""
        compose("exec", "-T", "dashboard", "python", "-c", dashboard_check)
        status, admission = request(api + "/train", {"families": ["ridge"]})
        assert status == 202
        job = wait_for_job(admission)
        assert job["result"]["test_evaluated"] is False and job["result"]["promoted"] is False
        # Reproduce a legacy record left running with a live, unrelated PID after API loss.
        orphan_id = uuid4().hex
        seed_orphan = (
            "import json; from pathlib import Path; "
            f"p=Path('/runtime/mlflow/training_jobs/{job['job_id']}/job.json'); "
            f"d=p.parent.parent/'{orphan_id}'; d.mkdir(); "
            "r=json.loads(p.read_text()); "
            f"r.update(job_id='{orphan_id}',status='running',pid=1,result=None,error=None); "
            "r.pop('process_started_at',None); (d/'job.json').write_text(json.dumps(r)); "
            "(d/'result.json').write_bytes((p.parent/'result.json').read_bytes())"
        )
        compose("exec", "-T", "api", "python", "-c", seed_orphan)
        compose("restart", "api")
        compose("up", "-d", "--wait", "--wait-timeout", "180")
        api = "http://" + compose("port", "api", "8000").stdout.strip()
        recovered = request(api + f"/train/{orphan_id}")[1]
        assert recovered["status"] == "succeeded" and recovered["result"] == job["result"]
        status, retry = request(api + "/train", {"families": ["ridge"]})
        assert status == 202 and retry["job_id"] != job["job_id"]
        assert wait_for_job(retry)["result"]["promoted"] is False
        assert request(api + "/ready")[1]["model_version"] == ready["model_version"]
    finally:
        logs = compose("logs", "--no-color", "--tail", "100", check=False)
        print(logs.stdout, logs.stderr)
        cleanup = compose("down", "--volumes", "--rmi", "all", check=False)
        assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr
