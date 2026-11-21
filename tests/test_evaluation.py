import json
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from steel_energy_mlops import evaluation
from steel_energy_mlops.cli import main
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.evaluation import error_slices, error_summary, evaluate_champion, schedule_cv
from steel_energy_mlops.seasonal import LEVEL_NAMES, fit_schedule_profile
from steel_energy_mlops.tracking import champion_version, open_store
from steel_energy_mlops.training import train_models
from steel_energy_mlops.training_data import Partition
from tests.helpers import make_training_data


def partition(stamps, values) -> Partition:
    return Partition(np.zeros((len(stamps), 7)), np.asarray(values, dtype=np.float64), tuple(stamps), "synthetic")


@pytest.mark.parametrize("statistic,expected", [("mean", [40, 40, 40, 44]), ("median", [20, 20, 20, 40])])
def test_schedule_uses_training_values_and_each_backoff(statistic, expected) -> None:
    stamps = [datetime(2018, 1, day, hour) for day, hour in [(1, 9), (8, 9), (15, 9), (2, 10), (7, 12)]]
    profile = fit_schedule_profile(partition(stamps, [10, 20, 90, 40, 60]), statistic)
    requested = tuple(datetime(2018, 1, day, hour) for day, hour in [(22, 9), (24, 9), (27, 9), (26, 23)])
    prediction, support = profile.predict(requested)
    np.testing.assert_allclose(prediction, expected)
    assert support == dict.fromkeys(LEVEL_NAMES, 1)
    empty, counts = profile.predict(())
    assert empty.shape == (0,) and sum(counts.values()) == 0


@pytest.mark.parametrize("statistic,stamps,targets", [
    ("sum", [datetime(2018, 1, 1)], [1]),
    ("mean", [], [1]),
    ("median", [], []),
    ("mean", [datetime(2018, 1, 1)], [-1]),
])
def test_schedule_rejects_invalid_training_contracts(statistic, stamps, targets) -> None:
    with pytest.raises(DataValidationError):
        fit_schedule_profile(partition(stamps, targets), statistic)


def test_reference_cv_refits_before_each_elapsed_time_window() -> None:
    stamps = [datetime(2018, 1, 1, 9) + timedelta(days=index) for index in range(16)]
    train = partition(stamps, [1] * 4 + [100] * 4 + [1000] * 4 + [10] * 4)
    result = schedule_cv(train)
    for statistic in ("mean", "median"):
        assert len(result[statistic]["fold_mae"]) == 3
        assert result[statistic]["fold_mae"][0] == 99
        assert result[statistic]["cv_mae"] == np.mean(result[statistic]["fold_mae"])
    train.target[8:] = 100000
    changed = schedule_cv(train)
    assert changed["mean"]["fold_mae"][0] == 99
    assert changed["median"]["fold_mae"][0] == 99


def test_error_summary_units_bias_and_tail_quantile() -> None:
    result = error_summary(np.asarray([0., 10., 30.]), np.asarray([5., 5., 20.]))
    assert result == pytest.approx({"row_count": 3, "mae": 20 / 3, "rmse": np.sqrt(50),
                                    "bias_kwh": -10 / 3, "p95_absolute_error_kwh": 9.5,
                                    "maximum_absolute_error_kwh": 10})


def test_error_slices_keep_observations_and_use_train_only_usage_cutoffs() -> None:
    validation = partition([datetime(2018, 1, 1), datetime(2018, 1, 7, 1),
                            datetime(2018, 1, 8, 1), datetime(2018, 2, 6)], [0, 3, 9, 10])
    predictions = {"champion": np.asarray([1., 5., 7., 5.]), "reference": validation.target.copy()}
    frame, slices, cutoffs = error_slices(validation, np.asarray([0., 4., 8., 12.]), predictions)
    assert cutoffs == {"train_q25_kwh": 3, "train_q75_kwh": 9}
    assert frame["champion_absolute_error_kwh"].tolist() == [1, 2, 2, 5]
    for _, group in slices.groupby(["model", "dimension"]):
        assert group.row_count.sum() == 4
        expected = 2.5 if group.iloc[0].model == "champion" else 0
        assert (group.mae * group.row_count).sum() / 4 == expected
    bands = slices[(slices.model == "champion") & (slices.dimension == "observed_usage_band")]
    assert bands.set_index("group").row_count.to_dict() == {"low": 2, "middle": 1, "high": 1}


@pytest.fixture(scope="module")
def diagnostic_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagnostics")
    directory = make_training_data(root)
    workspace = root / "tracking"
    trained = train_models(directory, workspace, "diagnostic-model", families=("ridge",), forest_trees=2)
    assert trained["promoted"]
    # Every diagnostic in this module must work with no final test observations available.
    (directory / "test.csv").unlink()
    return {"directory": directory, "workspace": workspace, "name": "diagnostic-model"}


def test_real_diagnostics_and_cli_log_artifacts_without_mutating_registry(diagnostic_session, capsys) -> None:
    session = diagnostic_session
    client, _ = open_store(session["workspace"], "steel-energy-diagnostics")
    before = champion_version(client, session["name"]).version
    versions_before = [version.version for version in client.search_model_versions(f"name='{session['name']}'")]
    assert main(["evaluate", "--data", str(session["directory"]), "--workspace", str(session["workspace"]),
                 "--model-name", session["name"]]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["test_evaluated"] is False and result["registry_mutated"] is False
    assert result["model_version"] == str(before)
    assert result["reference_selected_by"] == "training_elapsed_time_cv_mae"
    assert len(result["comparison"]) == 3
    assert client.get_run(result["run_id"]).info.status == "FINISHED"
    summary_path = client.download_artifacts(result["run_id"], "validation_summary.json")
    assert json.loads(Path(summary_path).read_text()) == result
    paths = {name: client.download_artifacts(result["run_id"], name) for name in
             ("validation_predictions.csv", "error_slices.csv", "largest_champion_errors.csv")}
    predictions = pd.read_csv(paths["validation_predictions.csv"])
    assert len(predictions) == result["comparison"][0]["row_count"] == 24
    assert predictions.champion_minus_reference_absolute_error_kwh.mean() == pytest.approx(
        result["champion_minus_reference_mae_kwh"])
    largest = pd.read_csv(paths["largest_champion_errors.csv"])
    assert len(largest) == 20 and largest.champion_absolute_error_kwh.is_monotonic_decreasing
    slices = pd.read_csv(paths["error_slices.csv"])
    assert slices.groupby(["model", "dimension"]).row_count.sum().eq(24).all()
    assert champion_version(client, session["name"]).version == before
    assert [version.version for version in client.search_model_versions(f"name='{session['name']}'")] == versions_before


def test_diagnostics_require_a_champion(diagnostic_session) -> None:
    session = diagnostic_session
    with pytest.raises(DataValidationError, match="No validated champion"):
        evaluate_champion(session["directory"], session["workspace"], "missing-model")


@pytest.mark.parametrize("key", ["train_sha256", "validation_sha256", "train_last_timestamp"])
def test_diagnostics_fail_closed_on_incompatible_provenance(diagnostic_session, key) -> None:
    session = diagnostic_session
    client, _ = open_store(session["workspace"], "steel-energy-diagnostics")
    version = champion_version(client, session["name"])
    original = version.tags[key]
    client.set_model_version_tag(session["name"], version.version, key, "incompatible")
    try:
        with pytest.raises(DataValidationError, match="Champion provenance"):
            evaluate_champion(session["directory"], session["workspace"], session["name"])
    finally:
        client.set_model_version_tag(session["name"], version.version, key, original)


@pytest.mark.parametrize("output", [None, np.asarray([-1.] * 24), np.asarray([np.nan] * 24), np.asarray([1.])])
def test_diagnostics_reject_invalid_models_and_predictions(diagnostic_session, monkeypatch, output) -> None:
    model = Pipeline([("model", DummyRegressor())])
    monkeypatch.setattr(model, "predict", lambda values: output)
    monkeypatch.setattr(evaluation, "load_model", lambda uri: None if output is None else model)
    with pytest.raises(DataValidationError):
        evaluate_champion(diagnostic_session["directory"], diagnostic_session["workspace"], diagnostic_session["name"])


def test_diagnostics_logging_failure_preserves_champion_and_ends_run(diagnostic_session, monkeypatch) -> None:
    session = diagnostic_session
    client, _ = open_store(session["workspace"], "steel-energy-diagnostics")
    before = champion_version(client, session["name"]).version

    def fail(*args, **kwargs):
        raise MlflowException("artifact write failed")

    monkeypatch.setattr(evaluation.mlflow, "log_dict", fail)
    with pytest.raises(MlflowException, match="artifact write failed"):
        evaluate_champion(session["directory"], session["workspace"], session["name"])
    assert champion_version(client, session["name"]).version == before
    assert mlflow.active_run() is None


def test_diagnostics_cli_reports_input_failure(tmp_path: Path, capsys) -> None:
    assert main(["evaluate", "--data", str(tmp_path / "missing")]) == 2
    assert "Validation diagnostics failed:" in capsys.readouterr().err
