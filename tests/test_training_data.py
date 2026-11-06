import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from steel_energy_mlops.data import TARGET, DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features
from steel_energy_mlops.training_data import load_training_data


def rewrite_partition(directory: Path, name: str, frame: pd.DataFrame) -> None:
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    metadata_path = directory / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["splits"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))


def test_training_does_not_open_final_test_data(prepared_training: Path) -> None:
    (prepared_training / "test.csv").unlink()
    loaded = load_training_data(prepared_training)
    assert loaded.train.features.shape == (112, 7)
    assert loaded.validation.features.shape == (24, 7)
    assert loaded.train.timestamps[-1] < loaded.validation.timestamps[0]


def test_detects_artifact_tampering(prepared_training: Path) -> None:
    path = prepared_training / "train.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DataValidationError, match="checksum"):
        load_training_data(prepared_training)


@pytest.mark.parametrize("key", ["feature_names", "target"])
def test_rejects_incompatible_contract(prepared_training: Path, key: str) -> None:
    path = prepared_training / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = "unexpected"
    path.write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="incompatible"):
        load_training_data(prepared_training)


def test_rejects_incomplete_manifest(prepared_training: Path) -> None:
    path = prepared_training / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["splits"]["train"]["sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="incomplete"):
        load_training_data(prepared_training)


@pytest.mark.parametrize(
    "value", [[], {"feature_names": list(FEATURE_NAMES), "target": TARGET},
              {"feature_names": list(FEATURE_NAMES), "target": TARGET, "source": {"sha256": "bad"}}]
)
def test_rejects_missing_source_provenance(prepared_training: Path, value) -> None:
    (prepared_training / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(DataValidationError):
        load_training_data(prepared_training)


@pytest.mark.parametrize(
    "failure", ["schema", "count", "short", "timezone", "order", "numeric", "negative", "features"]
)
def test_rejects_invalid_prepared_data(prepared_training: Path, failure: str) -> None:
    path = prepared_training / "train.csv"
    frame = pd.read_csv(path)
    if failure == "schema":
        frame = frame.drop(columns=[FEATURE_NAMES[0]])
    elif failure == "count":
        frame = frame.iloc[:-1]
    elif failure == "short":
        frame = frame.iloc[:1]
        manifest_path = prepared_training / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["splits"]["train"]["row_count"] = 1
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "timezone":
        frame.loc[0, "timestamp"] += "+00:00"
    elif failure == "order":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif failure == "numeric":
        frame.loc[0, FEATURE_NAMES[0]] = np.nan
    elif failure == "negative":
        frame.loc[0, TARGET] = -1
    else:
        frame.loc[0, FEATURE_NAMES[0]] = 0.123
    rewrite_partition(prepared_training, "train", frame)
    with pytest.raises(DataValidationError):
        load_training_data(prepared_training)


@pytest.mark.parametrize("field", ["first_timestamp", "last_timestamp"])
def test_rejects_changed_boundaries(prepared_training: Path, field: str) -> None:
    path = prepared_training / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata["splits"]["train"][field] = "2000-01-01T00:00:00"
    path.write_text(json.dumps(metadata))
    with pytest.raises(DataValidationError, match="boundaries"):
        load_training_data(prepared_training)


def test_rejects_validation_overlap(prepared_training: Path) -> None:
    frame = pd.read_csv(prepared_training / "validation.csv")
    stamps = pd.date_range("2017-01-01", periods=len(frame), freq="15min")
    frame["timestamp"] = [stamp.isoformat() for stamp in stamps]
    frame[list(FEATURE_NAMES)] = [calendar_features(stamp.to_pydatetime()) for stamp in stamps]
    rewrite_partition(prepared_training, "validation", frame)
    path = prepared_training / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata["splits"]["validation"]["first_timestamp"] = stamps[0].isoformat()
    metadata["splits"]["validation"]["last_timestamp"] = stamps[-1].isoformat()
    path.write_text(json.dumps(metadata))
    with pytest.raises(DataValidationError, match="precede"):
        load_training_data(prepared_training)


def test_rejects_constant_training_target(prepared_training: Path) -> None:
    frame = pd.read_csv(prepared_training / "train.csv")
    frame[TARGET] = 2.0
    rewrite_partition(prepared_training, "train", frame)
    with pytest.raises(DataValidationError, match="targets must vary"):
        load_training_data(prepared_training)
