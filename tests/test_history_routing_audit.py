import csv
from datetime import datetime, timedelta
import hashlib
import json

import pytest

from experiments.history_15min import check_routing as audit
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.training_data import load_training_data
from tests.helpers import make_row, write_csv


@pytest.fixture
def routing_sample(prepared_training, tmp_path, monkeypatch):
    source = tmp_path / "training_source.csv"
    data = load_training_data(prepared_training)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    first = datetime(2018, 1, 1)
    prefix = [make_row(first - timedelta(minutes=30)), make_row(first - timedelta(minutes=15))]
    write_csv(source, [*prefix, *rows])
    snapshot = {"validation_sha256": data.validation.sha256,
                "plan": {"train_sha256": data.train.sha256,
                         "context_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    frozen = tmp_path / "validation.json"
    frozen.write_text(json.dumps(snapshot))
    monkeypatch.setattr(audit, "HERE", tmp_path)
    return source, prepared_training, frozen


def test_shared_routes_match_all_frozen_inputs_without_opening_final_test(routing_sample):
    source, prepared, _ = routing_sample
    (prepared / "test.csv").unlink()
    result = audit.check_routing(source, prepared)
    assert result["passed"] and result["scope"] == "input_compatibility_only"
    assert result["partitions"]["train"] == {"rows": 112, "reasons": {"midnight_target": 2, "history_ready": 110}}
    assert result["partitions"]["validation"] == {"rows": 24, "reasons": {"history_ready": 24}}
    assert result["identical_history_inputs"] == 134
    assert result["maximum_age_at_origin_minutes"] == 30.
    unchanged = ("model_fitted", "accuracy_scored", "final_test_opened", "registry_mutated")
    assert all(result[key] is False for key in unchanged)


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_routing_audit_refuses_changed_frozen_partition(routing_sample, partition):
    source, prepared, frozen = routing_sample
    snapshot = json.loads(frozen.read_text())
    if partition == "train":
        snapshot["plan"]["train_sha256"] = "0" * 64
    else:
        snapshot["validation_sha256"] = "0" * 64
    frozen.write_text(json.dumps(snapshot))
    with pytest.raises(DataValidationError, match="partitions"):
        audit.check_routing(source, prepared)


def test_input_equivalence_check_detects_a_new_non_midnight_fallback(routing_sample):
    source, prepared, frozen = routing_sample
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))[3:]
    write_csv(source, rows)
    snapshot = json.loads(frozen.read_text())
    snapshot["plan"]["context_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    frozen.write_text(json.dumps(snapshot))
    with pytest.raises(DataValidationError, match="differs from the frozen experiment input"):
        audit.check_routing(source, prepared)
