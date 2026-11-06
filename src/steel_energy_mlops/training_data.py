"""Load verified training and validation artifacts without opening final test observations."""

import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from steel_energy_mlops.data import TARGET, DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features

FEATURE_CONTRACT = "calendar-v1-nonnegative-kwh"


@dataclass(frozen=True)
class Partition:
    features: NDArray[np.float64]
    target: NDArray[np.float64]
    timestamps: tuple[datetime, ...]
    sha256: str


@dataclass(frozen=True)
class TrainingData:
    train: Partition
    validation: Partition
    manifest: dict[str, Any]


def _load_partition(directory: Path, name: str, metadata: dict[str, Any]) -> Partition:
    raw = (directory / f"{name}.csv").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != metadata["sha256"]:
        raise DataValidationError(f"{name}: checksum differs from the preparation manifest.")
    frame = pd.read_csv(io.BytesIO(raw))
    if list(frame.columns) != ["timestamp", *FEATURE_NAMES, TARGET] or len(frame) != metadata["row_count"]:
        raise DataValidationError(f"{name}: prepared schema or row count differs from the manifest.")
    timestamps = tuple(datetime.fromisoformat(value) for value in frame["timestamp"])
    if len(timestamps) < 2 or any(stamp.tzinfo is not None for stamp in timestamps):
        raise DataValidationError(f"{name}: at least two timezone-naive observations are required.")
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise DataValidationError(f"{name}: timestamps must be strictly increasing.")
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    target = frame[TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(target).all() or np.any(target < 0):
        raise DataValidationError(f"{name}: numeric inputs must be finite and targets non-negative.")
    expected = np.asarray([calendar_features(stamp) for stamp in timestamps], dtype=np.float64)
    if not np.allclose(features, expected, rtol=0, atol=1e-12):
        raise DataValidationError(f"{name}: features do not match the calendar contract.")
    first_matches = timestamps[0].isoformat() == metadata["first_timestamp"]
    last_matches = timestamps[-1].isoformat() == metadata["last_timestamp"]
    if not first_matches or not last_matches:
        raise DataValidationError(f"{name}: date boundaries differ from the manifest.")
    return Partition(features, target, timestamps, digest)


def load_training_data(directory: Path) -> TrainingData:
    """The fixed final test partition is intentionally not loaded or scored here."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DataValidationError("Preparation manifest must be a JSON object.")
    if manifest.get("feature_names") != list(FEATURE_NAMES) or manifest.get("target") != TARGET:
        raise DataValidationError("Preparation manifest has an incompatible feature or target contract.")
    source = manifest.get("source")
    if not isinstance(source, dict) or re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) is None:
        raise DataValidationError("Preparation manifest must identify the source data with its SHA-256 hash.")
    try:
        train = _load_partition(directory, "train", manifest["splits"]["train"])
        validation = _load_partition(directory, "validation", manifest["splits"]["validation"])
    except (KeyError, TypeError) as exc:
        raise DataValidationError("Preparation manifest is incomplete.") from exc
    if train.timestamps[-1] >= validation.timestamps[0]:
        raise DataValidationError("Training observations must precede all validation observations.")
    if np.ptp(train.target) == 0:
        raise DataValidationError("Training targets must vary before comparing learned models.")
    return TrainingData(train, validation, manifest)
