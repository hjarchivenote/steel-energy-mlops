import csv
import hashlib
import json
from pathlib import Path

import pytest

from steel_energy_mlops.cli import main
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.input_audit import audit_inputs
from steel_energy_mlops.tracking import store_client


def update_source_hash(directory, source):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def test_training_only_audit_records_causal_inputs_without_model_changes(prepared_training, tmp_path) -> None:
    source = tmp_path / "training_source.csv"
    (prepared_training / "validation.csv").unlink()
    (prepared_training / "test.csv").unlink()
    result = audit_inputs(source, prepared_training, 15, tmp_path / "tracking")
    assert result["train_rows"] == 112 and result["scope"] == "training_observations_only"
    assert result["task"]["forecast_horizon_minutes"] == 15
    assert result["task"]["reporting_delay_is_assumed"] is True
    assert result["history"]["missing_rows"] == 2
    assert result["same_row_proxy_diagnostics"]["co2_target_correlation"] is None
    assert all(result[key] is False for key in (
        "model_trained", "validation_evaluated", "test_evaluated", "registry_mutated"))
    client = store_client(tmp_path / "tracking")
    assert client.search_registered_models() == []
    assert client.get_run(result["run_id"]).info.status == "FINISHED"
    summary = Path(client.download_artifacts(result["run_id"], "input_audit.json"))
    assert json.loads(summary.read_text()) == result
    path = client.download_artifacts(result["run_id"], "candidate_train_inputs.csv")
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result["train_rows"]
    for row in rows:
        if row["source_timestamp"]:
            assert row["source_timestamp"] <= row["source_available_at"] <= row["forecast_origin"]
        else:
            assert float(row["history_available"]) == 0.


def test_audit_cli_and_error_messages(prepared_training, tmp_path, capsys) -> None:
    arguments = ["audit-inputs", "--input", str(tmp_path / "training_source.csv"), "--data", str(prepared_training),
                 "--workspace", str(tmp_path / "tracking"), "--reporting-delay-minutes", "0"]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task"]["offline_reporting_delay_minutes"] == 0
    assert main([*arguments[:-1], "-1"]) == 2
    assert "Input audit failed:" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(arguments[:-2])


@pytest.mark.parametrize("delay", [-1, True, 1.5, 10**20])
def test_invalid_or_unrepresentable_reporting_delay(prepared_training, tmp_path, delay) -> None:
    with pytest.raises(DataValidationError, match="delay"):
        audit_inputs(tmp_path / "training_source.csv", prepared_training, delay, tmp_path / "tracking")


def test_audit_checks_raw_source_provenance(prepared_training, tmp_path) -> None:
    source = tmp_path / "training_source.csv"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(DataValidationError, match="provenance"):
        audit_inputs(source, prepared_training, 0, tmp_path / "tracking")


@pytest.mark.parametrize("failure", ["schema", "cells", "duplicate", "missing", "target"])
def test_audit_rejects_source_repointed_to_incompatible_training_rows(prepared_training, tmp_path, failure) -> None:
    source = tmp_path / "training_source.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if failure == "schema":
        rows[0] = rows[0][::-1]
    elif failure == "cells":
        rows[1] = rows[1][:-1]
    elif failure == "duplicate":
        rows.append(rows[1])
    elif failure == "missing":
        rows.pop(1)
    else:
        rows[1][1] = str(float(rows[1][1]) + 1.)
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)
    update_source_hash(prepared_training, source)
    with pytest.raises(DataValidationError):
        audit_inputs(source, prepared_training, 0, tmp_path / "tracking")


def test_singular_power_factors_and_absent_history_are_reported(prepared_training, tmp_path) -> None:
    source = tmp_path / "training_source.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows[1:]:
        row[5] = "100"
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)
    update_source_hash(prepared_training, source)
    result = audit_inputs(source, prepared_training, 10000, tmp_path / "tracking")
    assert result["history"]["available_rows"] == 0 and result["history"]["median_age_minutes_at_origin"] is None
    proxy = result["same_row_proxy_diagnostics"]
    assert proxy["eligible_rows"] == 0 and proxy["median_absolute_reconstruction_error_kwh"] is None


def test_holdout_source_targets_do_not_influence_training_audit(prepared_training, tmp_path) -> None:
    source = tmp_path / "training_source.csv"
    before = audit_inputs(source, prepared_training, 15, tmp_path / "tracking")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows[before["train_rows"] + 1:]:
        row[1:7] = ["999999", "999", "888", "777", "80", "90"]
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)
    update_source_hash(prepared_training, source)
    after = audit_inputs(source, prepared_training, 15, tmp_path / "tracking")
    for key in ("same_row_proxy_diagnostics", "history", "sampling", "train_sha256"):
        assert before[key] == after[key]
