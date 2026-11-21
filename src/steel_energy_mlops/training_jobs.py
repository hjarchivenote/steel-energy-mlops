"""Local subprocess training jobs with immutable inputs, durable status, and bounded admission."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from filelock import FileLock, Timeout

from steel_energy_mlops.training_data import load_training_data

LOGGER = logging.getLogger(__name__)


class TrainingBusyError(RuntimeError):
    """Another job or CLI training operation owns this workspace."""


class UnknownJobError(LookupError):
    """No job with this identifier exists in the configured workspace."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TrainingJobs:
    def __init__(self, workspace: Path, directory: Path, model_name: str):
        self.workspace = workspace.resolve()
        self.directory = directory.resolve()
        self.model_name = model_name
        self.root = self.workspace / "training_jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = Lock()
        self._thread: Thread | None = None
        self._closed = False

    def _read(self, job_id: str) -> dict[str, Any]:
        if re.fullmatch(r"[a-f0-9]{32}", job_id) is None:
            raise UnknownJobError("Unknown training job.")
        try:
            record = json.loads((self.root / job_id / "job.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise UnknownJobError("Unknown training job.") from exc
        required = {"job_id", "status", "pid", "model_name", "parameters", "created_at", "updated_at",
                    "training_request_id", "train_sha256", "validation_sha256", "result", "error"}
        if (not isinstance(record, dict) or record.get("job_id") != job_id or not required <= record.keys()
                or record["status"] not in ("running", "succeeded", "failed", "interrupted")
                or (record["pid"] is not None and type(record["pid"]) is not int)):
            raise ValueError("Invalid training job record.")
        return record

    def _save(self, record: dict[str, Any]) -> None:
        record["updated_at"] = _now()
        path = self.root / record["job_id"] / "job.json"
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(record, allow_nan=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _finish(self, record: dict[str, Any], returncode: int | None) -> dict[str, Any]:
        try:
            if returncode not in (0, None):
                raise ValueError("Training process failed.")
            result = json.loads((self.root / record["job_id"] / "result.json").read_text(encoding="utf-8"))
            if (not isinstance(result, dict) or result.get("model_name") != record["model_name"]
                    or not isinstance(result.get("run_id"), str) or not isinstance(result.get("promoted"), bool)
                    or result.get("test_evaluated") is not False):
                raise ValueError("Invalid training result.")
            record.update(status="succeeded", result=result, error=None)
        except (OSError, ValueError):
            interrupted = returncode is None
            record.update(status="interrupted" if interrupted else "failed", result=None, error={
                "code": "training_interrupted" if interrupted else "training_failed",
                "message": "Training did not complete successfully; inspect the local job log.",
                "exit_code": returncode,
            })
        self._save(record)
        return record

    def _recover(self, record: dict[str, Any]) -> dict[str, Any]:
        if record["status"] == "running" and not _alive(record["pid"]):
            return self._finish(record, None)
        return record

    def get(self, job_id: str) -> dict[str, Any]:
        with self._mutex:
            return self._recover(self._read(job_id))

    def start(self, parameters: dict[str, Any], request_id: str) -> dict[str, Any]:
        with self._mutex:
            if self._closed:
                raise RuntimeError("Training service is shutting down.")
            lease = FileLock(self.workspace / "api-training.lock", timeout=0, thread_local=False)
            try:
                lease.acquire()
            except Timeout as exc:
                raise TrainingBusyError("A training job is already active.") from exc
            process = None
            try:
                for path in self.root.glob("*/job.json"):
                    if self._recover(self._read(path.parent.name))["status"] == "running":
                        raise TrainingBusyError("A previous training worker is still active.")
                # Refuse admission when CLI training is already active. The child also takes this lock.
                with FileLock(self.workspace / "training.lock", timeout=0):
                    job_id = uuid4().hex
                    directory = self.root / job_id
                    snapshot = directory / "data"
                    snapshot.mkdir(parents=True)
                    for name in ("manifest.json", "train.csv", "validation.csv"):
                        shutil.copyfile(self.directory / name, snapshot / name)
                    data = load_training_data(snapshot)
                record = {
                    "job_id": job_id, "training_request_id": request_id, "model_name": self.model_name,
                    "status": "running", "created_at": _now(), "parameters": parameters,
                    "train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
                    "pid": None, "result": None, "error": None,
                }
                self._save(record)
                environment = os.environ.copy()
                for name in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME"):
                    environment.pop(name, None)
                environment.update(MLFLOW_DISABLE_AGENT_HINT="1", STEEL_ENERGY_TRAINING_JOB_ID=job_id,
                                   STEEL_ENERGY_TRAINING_REQUEST_ID=request_id)
                command = [sys.executable, "-m", "steel_energy_mlops", "train", "--data", str(snapshot),
                           "--workspace", str(self.workspace), "--model-name", self.model_name,
                           "--families", *parameters["families"], "--forest-trees", str(parameters["forest_trees"]),
                           "--minimum-improvement", str(parameters["minimum_improvement"]),
                           "--seed", str(parameters["seed"])]
                with (directory / "result.json").open("w") as output, (directory / "training.log").open("w") as log:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=log,
                                               env=environment, start_new_session=True)
                record["pid"] = process.pid
                self._save(record)
                self._thread = Thread(target=self._monitor, args=(process, job_id, lease), daemon=True)
                self._thread.start()
                return record
            except Exception:
                try:
                    if process is not None:
                        process.terminate()
                        process.wait()
                        self._thread = None
                finally:
                    lease.release(force=True)
                raise

    def _monitor(self, process: subprocess.Popen, job_id: str, lease: Any) -> None:
        try:
            code = process.wait()
            with self._mutex:
                self._finish(self._read(job_id), code)
        except Exception:
            LOGGER.exception("Could not persist completion for training job %s", job_id)
        finally:
            lease.release(force=True)

    def close(self) -> None:
        """Graceful API shutdown drains its admitted worker instead of abandoning a promotion job."""
        with self._mutex:
            self._closed = True
            thread = self._thread
        if thread is not None:
            thread.join()
