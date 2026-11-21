import hashlib
import json
import shutil
from datetime import datetime

from filelock import FileLock
from mlflow.exceptions import MlflowException
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from steel_energy_mlops import runtime
from steel_energy_mlops.cli import main
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.prediction import predict_consumption
from steel_energy_mlops.runtime import export_champion, initialize_runtime
from steel_energy_mlops.tracking import champion_version, store_client
from steel_energy_mlops.training import train_models

from tests.helpers import make_training_data


@pytest.fixture(scope="module")
def portable_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("portable-model")
    directory = make_training_data(root)
    workspace = root / "tracking"
    trained = train_models(directory, workspace, "portable-model", families=("ridge",))
    assert trained["promoted"]
    tracker = store_client(workspace)
    version = champion_version(tracker, "portable-model")
    # Transport an existing legacy champion without pretending it passed the newer policy.
    tracker.delete_model_version_tag("portable-model", version.version, "promotion_policy")
    seed = root / "seed"
    manifest = export_champion(workspace, seed, "portable-model")
    (directory / "test.csv").unlink()
    return {"source": directory, "workspace": workspace, "seed": seed, "manifest": manifest}


def initialize(session, root, seed=None):
    return initialize_runtime(session["source"], root / "data", root / "tracking", seed or session["seed"],
                              "portable-model", expected_rows=160)


def test_portable_export_and_restore_preserve_predictions_policy_and_provenance(portable_model, tmp_path) -> None:
    session = portable_model
    assert session["manifest"]["tags"]["promotion_policy"] == "legacy-v1"
    result = initialize(session, tmp_path)
    assert result["restored"]
    assert set(path.name for path in (tmp_path / "data").iterdir()) == set(runtime.RUNTIME_DATA_FILES)
    for name in runtime.RUNTIME_DATA_FILES:
        assert (tmp_path / "data" / name).read_bytes() == (session["source"] / name).read_bytes()
    stamps = (datetime(2019, 1, 7, 10), datetime(2019, 1, 7, 10, 15))
    original = predict_consumption(stamps, session["workspace"], "portable-model")
    restored = predict_consumption(stamps, tmp_path / "tracking", "portable-model")
    assert original["predictions"] == restored["predictions"]
    assert restored["promotion_policy"] == "legacy-v1"
    tracker = store_client(tmp_path / "tracking")
    version = champion_version(tracker, "portable-model")
    assert version.tags["initialization"] == "seed_restore"
    assert version.tags["train_sha256"] == session["manifest"]["tags"]["train_sha256"]
    assert version.tags["restored_from_run"] == session["manifest"]["source_run_id"]
    run = tracker.get_run(result["run_id"])
    assert run.info.status == "FINISHED" and run.data.tags["operation"] == "restore_existing_champion"
    # Restart initialization must keep new runtime state and must not require the seed again.
    repeat = initialize(session, tmp_path, tmp_path / "no-longer-mounted-seed")
    assert repeat["restored"] is False and repeat["model_version"] == result["model_version"]
    assert len(tracker.search_model_versions("name='portable-model'")) == 1


def test_export_and_initialize_cli(portable_model, tmp_path, capsys) -> None:
    session = portable_model
    seed = tmp_path / "seed"
    assert main(["export-model", "--workspace", str(session["workspace"]), "--model-name", "portable-model",
                 "--output", str(seed)]) == 0
    assert json.loads(capsys.readouterr().out)["model_name"] == "portable-model"
    assert main(["init-runtime", "--prepared-data", str(session["source"]), "--data", str(tmp_path / "data"),
                 "--workspace", str(tmp_path / "tracking"), "--model-seed", str(seed),
                 "--model-name", "portable-model", "--expected-rows", "160"]) == 0
    assert json.loads(capsys.readouterr().out)["restored"] is True


def test_export_does_not_overwrite_or_invent_a_champion(portable_model, tmp_path) -> None:
    session = portable_model
    before = (session["seed"] / "manifest.json").read_bytes()
    with pytest.raises(DataValidationError, match="already exists"):
        export_champion(session["workspace"], session["seed"], "portable-model")
    assert (session["seed"] / "manifest.json").read_bytes() == before
    with pytest.raises(DataValidationError, match="No champion"):
        export_champion(tmp_path / "empty", tmp_path / "new-seed", "missing-model")


@pytest.mark.parametrize("failure", ["list", "format", "name", "source-id", "tags", "contract", "tag-type",
                                     "hash", "cutoff", "timezone", "policy", "files", "tampered-file"])
def test_invalid_seeds_fail_before_a_champion_can_be_registered(portable_model, tmp_path, failure) -> None:
    seed = tmp_path / "seed"
    shutil.copytree(portable_model["seed"], seed)
    manifest = json.loads((seed / "manifest.json").read_text())
    if failure == "list":
        manifest = []
    elif failure == "format":
        manifest["format_version"] = 99
    elif failure == "name":
        manifest["model_name"] = "another-model"
    elif failure == "source-id":
        manifest["source_run_id"] = None
    elif failure == "tags":
        manifest["tags"] = None
    elif failure == "contract":
        manifest["tags"]["feature_contract"] = "different"
    elif failure == "tag-type":
        manifest["tags"]["unexpected"] = 3
    elif failure == "hash":
        manifest["tags"]["train_sha256"] = "not-a-hash"
    elif failure == "cutoff":
        manifest["tags"]["train_last_timestamp"] = "not-a-date"
    elif failure == "timezone":
        manifest["tags"]["train_last_timestamp"] = "2018-01-01T00:00:00+00:00"
    elif failure == "policy":
        manifest["tags"]["promotion_policy"] = ""
    elif failure == "files":
        manifest["files_sha256"] = {}
    else:
        (seed / "model" / "unexpected-file").write_text("tampered")
    (seed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError):
        initialize(portable_model, tmp_path, seed)
    assert champion_version(store_client(tmp_path / "tracking"), "portable-model") is None


def test_seed_must_match_runtime_data(portable_model, tmp_path) -> None:
    seed = tmp_path / "seed"
    shutil.copytree(portable_model["seed"], seed)
    manifest = json.loads((seed / "manifest.json").read_text())
    manifest["tags"]["validation_sha256"] = "a" * 64
    (seed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="training provenance"):
        initialize(portable_model, tmp_path, seed)


@pytest.mark.parametrize("failure", ["contains-input", "unknown-file", "output-file", "row-count", "checksum"])
def test_invalid_prepared_data_cannot_initialize_a_runtime(portable_model, tmp_path, failure) -> None:
    source = tmp_path / "input"
    shutil.copytree(portable_model["source"], source)
    original = (portable_model["source"] / "train.csv").read_bytes()
    directory = tmp_path / "data"
    if failure == "contains-input":
        directory = tmp_path
    elif failure == "unknown-file":
        directory.mkdir()
        (directory / "unrelated.txt").write_text("preserve")
    elif failure == "output-file":
        directory.write_text("preserve")
    elif failure == "row-count":
        path = source / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source"]["row_count"] = 5000
        path.write_text(json.dumps(manifest))
    else:
        with (source / "train.csv").open("a") as stream:
            stream.write("modified\n")
    with pytest.raises(DataValidationError):
        initialize_runtime(source, directory, tmp_path / "tracking", portable_model["seed"],
                           "portable-model", expected_rows=160)
    assert champion_version(store_client(tmp_path / "tracking"), "portable-model") is None
    assert (portable_model["source"] / "train.csv").read_bytes() == original
    if failure == "unknown-file":
        assert (directory / "unrelated.txt").read_text() == "preserve"
    elif failure == "output-file":
        assert directory.read_text() == "preserve"


def test_changed_inputs_cannot_overwrite_existing_runtime_data(portable_model, tmp_path) -> None:
    initialize(portable_model, tmp_path)
    before = {name: (tmp_path / "data" / name).read_bytes() for name in runtime.RUNTIME_DATA_FILES}
    source = tmp_path / "changed-input"
    shutil.copytree(portable_model["source"], source)
    train = source / "train.csv"
    lines = train.read_text().splitlines()
    values = lines[1].split(",")
    values[-1] = str(float(values[-1]) + 1)
    lines[1] = ",".join(values)
    train.write_text("\n".join(lines) + "\n")
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["train"]["sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="training provenance"):
        initialize_runtime(source, tmp_path / "data", tmp_path / "tracking", portable_model["seed"],
                           "portable-model", expected_rows=160)
    assert before == {name: (tmp_path / "data" / name).read_bytes() for name in runtime.RUNTIME_DATA_FILES}
    tracker = store_client(tmp_path / "tracking")
    assert len(tracker.search_model_versions("name='portable-model'")) == 1
    assert str(champion_version(tracker, "portable-model").version) == "1"


@pytest.mark.parametrize("outputs", [None, np.asarray([np.nan]), np.asarray([-1.]), np.asarray([1., 2.])])
def test_runtime_rejects_invalid_model_artifacts(portable_model, tmp_path, monkeypatch, outputs) -> None:
    model = Pipeline([("model", DummyRegressor())])
    monkeypatch.setattr(model, "predict", lambda values: outputs)
    monkeypatch.setattr(runtime, "load_model", lambda path: None if outputs is None else model)
    with pytest.raises(DataValidationError):
        initialize(portable_model, tmp_path)
    assert champion_version(store_client(tmp_path / "tracking"), "portable-model") is None


def test_restoration_logging_failure_cannot_publish_a_champion(portable_model, tmp_path, monkeypatch) -> None:
    original = runtime.mlflow.log_dict

    def fail(dictionary, artifact_file, **kwargs):
        if artifact_file == "initialization.json":
            raise MlflowException("initialization record failed")
        return original(dictionary, artifact_file, **kwargs)

    monkeypatch.setattr(runtime.mlflow, "log_dict", fail)
    with pytest.raises(MlflowException, match="initialization record failed"):
        initialize(portable_model, tmp_path)
    tracker = store_client(tmp_path / "tracking")
    assert champion_version(tracker, "portable-model") is None
    assert len(tracker.search_model_versions("name='portable-model'")) == 1


def test_active_training_prevents_runtime_initialization(portable_model, tmp_path) -> None:
    workspace = tmp_path / "tracking"
    workspace.mkdir()
    with FileLock(workspace / "training.lock"):
        with pytest.raises(DataValidationError, match="active training"):
            initialize(portable_model, tmp_path)


def test_cli_reports_export_and_runtime_configuration_failures(tmp_path, capsys) -> None:
    assert main(["export-model", "--model-name", "../bad", "--output", str(tmp_path / "seed")]) == 2
    assert "Model export failed:" in capsys.readouterr().err
    assert main(["init-runtime", "--prepared-data", str(tmp_path / "missing"), "--data", str(tmp_path / "data"),
                 "--workspace", str(tmp_path / "tracking"), "--model-seed", str(tmp_path / "seed")]) == 2
    assert "Runtime initialization failed:" in capsys.readouterr().err
