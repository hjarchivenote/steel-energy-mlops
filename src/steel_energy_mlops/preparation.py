"""Reproducible, model-ready partitions with explicit feature and evaluation contracts."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from steel_energy_mlops.data import SOURCE_COLUMNS, TARGET, DataValidationError, load_dataset, split_chronologically
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features

OUTPUT_FILES = ("train.csv", "validation.csv", "test.csv", "manifest.json")


def prepare_dataset(
    source: Path,
    output: Path,
    expected_rows: int = 5000,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, object]:
    """Validate before writing; keep the target separate from timestamp-only predictors."""
    if source.resolve().is_relative_to(output.resolve()):
        raise DataValidationError("Output directory must not contain the source CSV.")
    dataset = load_dataset(source, expected_rows)
    split = split_chronologically(dataset.records, train_fraction, validation_fraction)
    if output.exists() and (not output.is_dir() or any(path.name not in OUTPUT_FILES for path in output.iterdir())):
        raise DataValidationError("Output must be a directory containing only generated preparation files.")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": {"filename": source.name, "sha256": dataset.source_sha256, "row_count": len(dataset.records)},
        "target": TARGET,
        "feature_names": list(FEATURE_NAMES),
        "excluded_source_predictors": list(SOURCE_COLUMNS[2:]),
        "timestamp_policy": "Timezone-naive source timestamps; no timezone conversion.",
        "evaluation_policy": "Chronological split; final test data is reserved for final evaluation, not tuning.",
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
    }
    partitions: dict[str, object] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".steel-energy-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, records in split.partitions():
            path = staging / f"{name}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(("timestamp", *FEATURE_NAMES, TARGET))
                for record in records:
                    writer.writerow(
                        (record.timestamp.isoformat(), *calendar_features(record.timestamp), record.usage_kwh)
                    )
            partitions[name] = {
                "filename": path.name,
                "row_count": len(records),
                "first_timestamp": records[0].timestamp.isoformat(),
                "last_timestamp": records[-1].timestamp.isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest["splits"] = partitions
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.mkdir(exist_ok=True)
        for name in OUTPUT_FILES:
            os.replace(staging / name, output / name)
    return manifest
