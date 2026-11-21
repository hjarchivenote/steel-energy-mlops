import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.history_15min import replay_validation as experiment
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.preparation import prepare_dataset
from steel_energy_mlops.tracking import store_client
from steel_energy_mlops.training_data import load_training_data, load_training_partition
from tests.helpers import make_row, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_context_uses_timestamp_availability_and_ignores_future_values(prepared_training, tmp_path):
    source = tmp_path / "training_source.csv"
    train = load_training_partition(prepared_training)
    (prepared_training / "validation.csv").unlink()
    (prepared_training / "test.csv").unlink()
    before = experiment.context_inputs(source, (train,), digest(source))
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    # Raw ordering is irrelevant; numerical values after the cutoff must never be parsed.
    for row in rows[len(train.target) + 1:]:
        row[1:] = ["unavailable"] * 10
    write_csv(source, list(reversed(rows[1:])))
    after = experiment.context_inputs(source, (train,), digest(source))
    assert before == after
    for row in after:
        if row.source_timestamp is not None:
            assert not experiment.is_midnight(row.source_timestamp)
            assert row.source_timestamp + timedelta(minutes=15) <= row.forecast_origin < row.target_timestamp
    # The midnight record and the as-yet-unreported 00:15 record cannot supply this query.
    assert after[1].source_timestamp is None
    assert after[3].source_timestamp == train.timestamps[1]


@pytest.mark.parametrize("failure", ["checksum", "schema", "cells", "duplicate", "missing", "target"])
def test_context_rejects_incompatible_source(prepared_training, tmp_path, failure):
    source = tmp_path / "training_source.csv"
    train = load_training_partition(prepared_training)
    original_digest = digest(source)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if failure == "schema":
        rows[0].reverse()
    elif failure == "cells":
        rows[1].pop()
    elif failure == "duplicate":
        rows.append(rows[1])
    elif failure == "missing":
        rows.pop(1)
    else:
        rows[1][1] = "99999"
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)
    expected = original_digest if failure == "checksum" else digest(source)
    with pytest.raises(DataValidationError):
        experiment.context_inputs(source, (train,), expected)


def test_error_metrics_and_all_acceptance_checks_remain_frozen():
    target = np.array([10., 30., 60., 100.])
    control = np.array([12., 25., 50., 80.])
    candidate = np.array([11., 28., 55., 90.])
    high, mask = target > 40, np.ones(4, dtype=bool)
    control_score = experiment.metrics(target, control, high, mask)
    candidate_score = experiment.metrics(target, candidate, high, mask)
    assert control_score["mae"] == 9.25
    assert control_score["high_usage_bias"] == -15.
    scores = {"calendar": control_score, "persistence": control_score, "candidate": candidate_score}
    assert all(experiment.acceptance_checks(scores).values())
    scores["candidate"] = control_score
    assert not experiment.acceptance_checks(scores)["persistence_mae"]
    assert not experiment.acceptance_checks(scores)["persistence_high_bias"]
    absent = experiment.metrics(target, candidate, np.zeros(4, dtype=bool), mask)
    assert absent["high_usage_mae"] is None and absent["high_usage_bias"] is None
    scores["candidate"] = absent
    assert not experiment.acceptance_checks(scores)["calendar_high_mae"]


def make_gapped_sample(tmp_path):
    start = datetime(2018, 1, 1, 18)
    rows = [make_row(start + timedelta(minutes=15 * (index + (2 if index >= 112 else 0))),
                     usage=10. + (index % 14)**2) for index in range(160)]
    source = write_csv(tmp_path / "source.csv", rows)
    prepared = tmp_path / "prepared"
    prepare_dataset(source, prepared, expected_rows=160)
    data = load_training_data(prepared)
    plan = {"train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
            "context_source_sha256": digest(source),
            "candidate": {"loss": "squared_error", "max_iter": 200, "max_leaf_nodes": 15,
                          "l2_regularization": 1., "early_stopping": False, "random_state": 42}}
    return source, prepared, plan


def test_frozen_replay_logs_evidence_retains_midnight_and_never_opens_test(tmp_path):
    source, prepared, plan = make_gapped_sample(tmp_path)
    (prepared / "test.csv").unlink()
    workspace = tmp_path / "tracking"
    result = experiment.replay(source, prepared, workspace, plan)
    assert result["target_rows"] == 24 and result["midnight_fallback_targets"] == 1
    assert result["registry_mutated"] is False and result["final_test_evaluated"] is False
    assert result["is_replay"] is True
    client = store_client(workspace)
    assert client.search_registered_models() == []
    path = client.download_artifacts(result["run_id"], "validation_predictions_and_availability.json")
    trace = json.loads(Path(path).read_text())["rows"]
    assert len(trace) == 24
    for row in trace:
        assert row["history_timestamp"] <= row["history_available_at"] <= row["origin"] < row["target"]
        if row["midnight_calendar_fallback"]:
            assert row["candidate_kwh"] == row["calendar_kwh"] == row["persistence_kwh"]
    path = client.download_artifacts(result["run_id"], "validation_summary.json")
    assert json.loads(Path(path).read_text()) == result


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_replay_refuses_changed_partition_identity(tmp_path, partition):
    source, prepared, plan = make_gapped_sample(tmp_path)
    plan[partition + "_sha256"] = "0" * 64
    with pytest.raises(DataValidationError, match="checksum"):
        experiment.replay(source, prepared, tmp_path / "tracking", plan)


def test_replay_refuses_labels_not_available_at_first_forecast_origin(prepared_training, tmp_path):
    data = load_training_data(prepared_training)
    source = tmp_path / "training_source.csv"
    plan = {"train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
            "context_source_sha256": digest(source)}
    with pytest.raises(DataValidationError, match="Fitted labels"):
        experiment.replay(source, prepared_training, tmp_path / "tracking", plan)
