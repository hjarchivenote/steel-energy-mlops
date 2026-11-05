import csv
import hashlib
import json
from pathlib import Path

import pytest

from steel_energy_mlops.data import SOURCE_COLUMNS, TARGET, DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES
from steel_energy_mlops.preparation import OUTPUT_FILES, prepare_dataset
from tests.helpers import write_csv


def test_preparation_produces_disjoint_model_ready_partitions(dataset_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    manifest = prepare_dataset(dataset_path, output, expected_rows=20)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert manifest["source"]["sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert manifest["feature_names"] == list(FEATURE_NAMES)
    assert manifest["target"] == TARGET
    assert set(manifest["excluded_source_predictors"]) == set(SOURCE_COLUMNS[2:])
    timestamps: list[str] = []
    for name, count in (("train", 14), ("validation", 3), ("test", 3)):
        path = output / f"{name}.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        assert reader.fieldnames == ["timestamp", *FEATURE_NAMES, TARGET]
        assert len(rows) == count
        assert manifest["splits"][name]["row_count"] == count
        assert manifest["splits"][name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert float(rows[0][TARGET]) == len(timestamps)
        timestamps.extend(row["timestamp"] for row in rows)
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == 20


def test_preparation_is_byte_reproducible(dataset_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    prepare_dataset(dataset_path, output, expected_rows=20)
    original = {name: (output / name).read_bytes() for name in OUTPUT_FILES}
    prepare_dataset(dataset_path, output, expected_rows=20)
    assert {name: (output / name).read_bytes() for name in OUTPUT_FILES} == original


def test_invalid_data_does_not_modify_previous_outputs(dataset_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    prepare_dataset(dataset_path, output, expected_rows=20)
    original = {name: (output / name).read_bytes() for name in OUTPUT_FILES}
    with pytest.raises(DataValidationError, match="Expected 5000"):
        prepare_dataset(dataset_path, output)
    assert {name: (output / name).read_bytes() for name in OUTPUT_FILES} == original
    absent = tmp_path / "absent"
    with pytest.raises(DataValidationError, match="Expected 5000"):
        prepare_dataset(dataset_path, absent)
    assert not absent.exists()


def test_output_cannot_contain_original_source(dataset_path: Path) -> None:
    original = dataset_path.read_bytes()
    with pytest.raises(DataValidationError, match="must not contain the source"):
        prepare_dataset(dataset_path, dataset_path.parent, expected_rows=20)
    assert dataset_path.read_bytes() == original


def test_rejects_unrelated_files_in_output(dataset_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    unrelated = output / "notes.txt"
    unrelated.write_text("keep this")
    with pytest.raises(DataValidationError, match="only generated"):
        prepare_dataset(dataset_path, output, expected_rows=20)
    assert unrelated.read_text() == "keep this"


def test_rejects_file_as_output_directory(dataset_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.write_text("keep this")
    with pytest.raises(DataValidationError, match="only generated"):
        prepare_dataset(dataset_path, output, expected_rows=20)
    assert output.read_text() == "keep this"


def test_measured_values_and_target_cannot_leak_into_predictors(dataset_path: Path, tmp_path: Path) -> None:
    original = tmp_path / "original"
    changed = tmp_path / "changed"
    prepare_dataset(dataset_path, original, expected_rows=20)
    with dataset_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    for row in rows:
        row[1:7] = ["999", "50", "25", "0.06", "95", "70"]
        row[-1] = "Maximum_Load"
    alternative = write_csv(tmp_path / "alternative.csv", rows)
    prepare_dataset(alternative, changed, expected_rows=20)
    for name in ("train", "validation", "test"):
        with (original / f"{name}.csv").open() as stream:
            left = list(csv.DictReader(stream))
        with (changed / f"{name}.csv").open() as stream:
            right = list(csv.DictReader(stream))
        assert [row[TARGET] for row in left] != [row[TARGET] for row in right]
        assert [{key: row[key] for key in FEATURE_NAMES} for row in left] == [
            {key: row[key] for key in FEATURE_NAMES} for row in right
        ]
