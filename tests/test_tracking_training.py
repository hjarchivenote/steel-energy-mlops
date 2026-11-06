import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest
from filelock import FileLock
from mlflow.exceptions import MlflowException
from mlflow.sklearn import load_model
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from steel_energy_mlops import prediction, training
from steel_energy_mlops.cli import main
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import calendar_features
from steel_energy_mlops.prediction import predict_consumption
from steel_energy_mlops.tracking import champion_version, open_store, validate_model_name
from steel_energy_mlops.training import train_models
from tests.helpers import make_training_data


@pytest.fixture(scope="module")
def training_session(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("mlflow")
    directory = make_training_data(root)
    workspace = root / "tracking"
    result = train_models(directory, workspace, model_name="test-calendar", forest_trees=2)
    assert result["promoted"] is True
    return {"directory": directory, "workspace": workspace, "result": result, "model_name": "test-calendar"}


def test_real_mlflow_artifacts_and_model_roundtrips(training_session: dict[str, Any]) -> None:
    session = training_session
    result = session["result"]
    client, _ = open_store(session["workspace"], "steel-energy-training")
    run = client.get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["test_evaluated"] == "false"
    assert run.data.metrics["selected_validation_mae"] < run.data.metrics["baseline_median_validation_mae"]
    assert result["selected_family"] == "ridge"
    summary_path = client.download_artifacts(result["run_id"], "training_summary.json")
    assert json.loads(Path(summary_path).read_text())["promotion_eligible"] is True
    features = np.asarray([calendar_features(datetime(2018, 1, 2, 12))], dtype=np.float64)
    for row in result["comparison"]:
        model = load_model(row["model_uri"])
        assert isinstance(model, Pipeline)
        values = model.predict(features)
        assert values.shape == (1,) and np.isfinite(values).all() and (values >= 0).all()
    assert str(champion_version(client, session["model_name"]).version) == result["candidate_version"]


def test_identical_candidate_is_rejected_without_replacing_champion(training_session: dict[str, Any]) -> None:
    session = training_session
    before = session["result"]["candidate_version"]
    result = train_models(session["directory"], session["workspace"], session["model_name"], forest_trees=2)
    client, _ = open_store(session["workspace"], "steel-energy-training")
    assert result["promoted"] is False and result["reason"] == "incumbent_mae_not_improved"
    assert result["candidate_version"] != before
    assert result["previous_champion_version"] == before
    assert str(champion_version(client, session["model_name"]).version) == str(before)


def test_prediction_logs_exact_request_and_response(training_session: dict[str, Any]) -> None:
    session = training_session
    stamps = (datetime(2019, 1, 7, 10), datetime(2019, 1, 7, 10, 15))
    response = predict_consumption(stamps, session["workspace"], session["model_name"])
    client, _ = open_store(session["workspace"], "steel-energy-prediction")
    request_path = client.download_artifacts(response["run_id"], "request.json")
    response_path = client.download_artifacts(response["run_id"], "response.json")
    assert json.loads(Path(request_path).read_text())["timestamps"] == [stamp.isoformat() for stamp in stamps]
    assert json.loads(Path(response_path).read_text()) == response
    assert response["model_version"] == session["result"]["candidate_version"]
    assert all(item["usage_kwh"] >= 0 for item in response["predictions"])


def test_tracking_active_run_is_preserved(training_session: dict[str, Any]) -> None:
    client, experiment_id = open_store(training_session["workspace"], "steel-energy-training")
    with mlflow.start_run(experiment_id=experiment_id) as active:
        with pytest.raises(DataValidationError, match="active MLflow run"):
            open_store(training_session["workspace"], "steel-energy-training")
        assert mlflow.active_run().info.run_id == active.info.run_id
    assert client.get_run(active.info.run_id).info.status == "FINISHED"


def test_mlflow_project_parent_experiment_is_reused(training_session, monkeypatch: pytest.MonkeyPatch) -> None:
    client, experiment_id = open_store(training_session["workspace"], "steel-energy-training")
    parent = client.create_run(experiment_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", parent.info.run_id)
    _, found = open_store(training_session["workspace"], "another-experiment-name")
    assert found == experiment_id
    client.set_terminated(parent.info.run_id)


@pytest.mark.parametrize("name", ["", "../bad", "name/other", "a" * 101])
def test_rejects_invalid_registry_names(name: str) -> None:
    with pytest.raises(DataValidationError, match="Model name"):
        validate_model_name(name)


def test_registry_errors_are_not_treated_as_missing_models(training_session, monkeypatch) -> None:
    client, _ = open_store(training_session["workspace"], "steel-energy-training")

    def fail(*args, **kwargs):
        raise MlflowException("backend unavailable")

    monkeypatch.setattr(client, "get_model_version_by_alias", fail)
    with pytest.raises(MlflowException, match="backend unavailable"):
        champion_version(client, "test-calendar")


def test_concurrent_training_is_refused_before_mutation(training_session) -> None:
    session = training_session
    with FileLock(session["workspace"] / "training.lock"):
        with pytest.raises(DataValidationError, match="workspace lock"):
            train_models(session["directory"], session["workspace"], session["model_name"])


def test_invalid_threshold_is_refused_before_loading_data(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="Minimum improvement"):
        train_models(tmp_path / "missing", minimum_improvement=1)


def test_optional_git_metadata_without_repository(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert training._git_provenance() == {"git_revision": "unavailable", "git_dirty": "unknown"}


def test_optional_git_metadata_without_git_binary(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(training.subprocess, "run", missing)
    assert training._git_provenance() == {"git_revision": "unavailable", "git_dirty": "unknown"}


@pytest.mark.parametrize("failure", ["contract", "cutoff", "missing-cutoff"])
def test_incompatible_incumbents_fail_closed(training_session, failure: str) -> None:
    session = training_session
    client, _ = open_store(session["workspace"], "steel-energy-training")
    version = champion_version(client, session["model_name"])
    original = dict(version.tags)
    key = "feature_contract" if failure == "contract" else "train_last_timestamp"
    if failure == "missing-cutoff":
        client.delete_model_version_tag(session["model_name"], version.version, key)
    else:
        value = "incompatible" if failure == "contract" else "2025-01-01T00:00:00"
        client.set_model_version_tag(session["model_name"], version.version, key, value)
    try:
        with pytest.raises(DataValidationError):
            train_models(session["directory"], session["workspace"], session["model_name"])
        assert champion_version(client, session["model_name"]).version == version.version
    finally:
        client.set_model_version_tag(session["model_name"], version.version, key, original[key])


def test_invalid_champion_artifact_is_rejected(training_session, monkeypatch) -> None:
    session = training_session
    monkeypatch.setattr(training, "load_model", lambda uri: None)
    with pytest.raises(DataValidationError, match="sklearn Pipeline"):
        train_models(session["directory"], session["workspace"], session["model_name"])


def test_tracking_failure_cannot_promote_a_model(training_session, monkeypatch) -> None:
    session = training_session
    client, _ = open_store(session["workspace"], "steel-energy-training")
    before = champion_version(client, session["model_name"]).version

    def fail(*args, **kwargs):
        raise MlflowException("artifact write failed")

    monkeypatch.setattr(training.mlflow, "log_dict", fail)
    with pytest.raises(MlflowException, match="artifact write failed"):
        train_models(session["directory"], session["workspace"], session["model_name"])
    assert str(champion_version(client, session["model_name"]).version) == str(before)
    assert mlflow.active_run() is None


@pytest.mark.parametrize("timestamps", [(), (datetime(2018, 1, 1),) * 1001, (datetime(2018, 1, 1, 0, 1),)])
def test_prediction_rejects_invalid_batch_before_loading(timestamps, tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        predict_consumption(timestamps, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_prediction_requires_a_validated_champion(training_session) -> None:
    with pytest.raises(DataValidationError, match="No validated champion"):
        predict_consumption((datetime(2018, 1, 1),), training_session["workspace"], "missing-model")


def test_prediction_rejects_invalid_artifact(training_session, monkeypatch) -> None:
    monkeypatch.setattr(prediction, "load_model", lambda uri: None)
    with pytest.raises(DataValidationError, match="sklearn Pipeline"):
        predict_consumption((datetime(2018, 1, 1),), training_session["workspace"], "test-calendar")


@pytest.mark.parametrize("outputs", [np.asarray([np.nan]), np.asarray([-1.0]), np.asarray([1.0, 2.0])])
def test_prediction_rejects_invalid_model_outputs(training_session, monkeypatch, outputs) -> None:
    pipeline = Pipeline([("model", DummyRegressor())])
    monkeypatch.setattr(pipeline, "predict", lambda features: outputs)
    monkeypatch.setattr(prediction, "load_model", lambda uri: pipeline)
    with pytest.raises(DataValidationError, match="invalid consumption"):
        predict_consumption((datetime(2018, 1, 1),), training_session["workspace"], "test-calendar")


def test_training_and_prediction_cli(training_session, capsys) -> None:
    session = training_session
    result = main(["train", "--data", str(session["directory"]), "--workspace", str(session["workspace"]),
                   "--model-name", "cli-model", "--families", "ridge", "--forest-trees", "2"])
    assert result == 0
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["promoted"] is True
    result = main(["predict", "--timestamp", "2019-01-07T10:00:00", "--workspace", str(session["workspace"]),
                   "--model-name", "cli-model"])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["model_version"] == outcome["candidate_version"]
