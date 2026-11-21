"""Portable champion transfer and idempotent preparation of an isolated container workspace."""

import hashlib
import json
import re
import shutil
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from filelock import FileLock, Timeout
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.sklearn import load_model, log_model
import numpy as np
import skops.io
from sklearn.pipeline import Pipeline

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features
from steel_energy_mlops.tracking import DEFAULT_MODEL, champion_version, open_store, require_compatible_version
from steel_energy_mlops.tracking import store_client, validate_model_name
from steel_energy_mlops.training_data import FEATURE_CONTRACT, TrainingData, load_training_data

PROVENANCE_HASHES = ("train_sha256", "validation_sha256")
RUNTIME_DATA_FILES = ("manifest.json", "train.csv", "validation.csv")


def _hashes(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _seed_manifest(directory: Path, model_name: str) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("format_version") != 1
            or manifest.get("model_name") != model_name
            or any(not isinstance(manifest.get(key), str) or not manifest[key]
                   for key in ("source_version", "source_run_id"))):
        raise DataValidationError("Model seed has an incompatible format or model name.")
    tags = manifest.get("tags")
    if (not isinstance(tags, dict) or tags.get("feature_contract") != FEATURE_CONTRACT
            or any(not isinstance(value, str) for value in tags.values())):
        raise DataValidationError("Model seed has an incompatible feature contract.")
    if any(re.fullmatch(r"[0-9a-f]{64}", str(tags.get(key, ""))) is None for key in PROVENANCE_HASHES):
        raise DataValidationError("Model seed must identify its training and validation data.")
    try:
        cutoff = datetime.fromisoformat(tags.get("train_last_timestamp", ""))
    except ValueError as exc:
        raise DataValidationError("Model seed must preserve a valid training cutoff.") from exc
    if cutoff.tzinfo is not None or not tags.get("promotion_policy"):
        raise DataValidationError("Model seed must preserve its original cutoff and promotion policy.")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or not expected or _hashes(directory / "model") != expected:
        raise DataValidationError("Model seed files differ from the export checksums.")
    return manifest


def export_champion(workspace: Path, output: Path, model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    validate_model_name(model_name)
    if output.exists():
        raise DataValidationError("Model export output already exists; select a new directory.")
    client = store_client(workspace)
    version = champion_version(client, model_name)
    if version is None:
        raise DataValidationError("No champion is available to export.")
    require_compatible_version(version)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "seed"
        staging.mkdir()
        downloaded = download_artifacts(f"models:/{model_name}/{version.version}",
                                        dst_path=str(Path(temporary) / "copy"))
        shutil.copytree(downloaded, staging / "model")
        manifest = {"format_version": 1, "model_name": model_name, "source_version": str(version.version),
                    "source_run_id": version.run_id,
                    "tags": {**version.tags, "promotion_policy": version.tags.get("promotion_policy", "legacy-v1")},
                    "files_sha256": _hashes(staging / "model")}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _seed_manifest(staging, model_name)
        staging.replace(output)
    return manifest


def _require_matching_data(tags: dict[str, str], data: TrainingData) -> None:
    if (tags.get("train_sha256") != data.train.sha256 or tags.get("validation_sha256") != data.validation.sha256
            or tags.get("train_last_timestamp") != data.train.timestamps[-1].isoformat()):
        raise DataValidationError("Runtime data differs from the existing champion's training provenance.")


def _copy_runtime_data(source: Path, directory: Path, tags: dict[str, str], expected_rows: int) -> None:
    # Preserve the original bytes: platform math libraries can differ at the last floating-point digit.
    if source.resolve().is_relative_to(directory.resolve()):
        raise DataValidationError("Runtime output must not contain the prepared input directory.")
    if directory.exists() and (not directory.is_dir() or any(
            path.name not in RUNTIME_DATA_FILES for path in directory.iterdir())):
        raise DataValidationError("Runtime data output must contain only training, validation and their manifest.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".runtime-data-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        for name in RUNTIME_DATA_FILES:
            shutil.copyfile(source / name, staging / name)
        data = load_training_data(staging)
        if data.manifest["source"].get("row_count") != expected_rows:
            raise DataValidationError("Runtime source row count differs from the expected cohort.")
        _require_matching_data(tags, data)
        directory.mkdir(exist_ok=True)
        for name in RUNTIME_DATA_FILES:
            (staging / name).replace(directory / name)


def initialize_runtime(
    source: Path, directory: Path, workspace: Path, seed: Path, model_name: str = DEFAULT_MODEL,
    expected_rows: int = 5000,
) -> dict[str, Any]:
    validate_model_name(model_name)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(workspace / "training.lock", timeout=0):
            client, experiment_id = open_store(workspace, "steel-energy-runtime-import")
            existing = champion_version(client, model_name)
            if existing is not None:
                require_compatible_version(existing)
                _copy_runtime_data(source, directory, existing.tags, expected_rows)
                return {"model_name": model_name, "model_version": str(existing.version), "restored": False}
            manifest = _seed_manifest(seed, model_name)
            _copy_runtime_data(source, directory, manifest["tags"], expected_rows)
            model = load_model(str(seed / "model"))
            if not isinstance(model, Pipeline):
                raise DataValidationError("Runtime champion must contain a fitted sklearn Pipeline.")
            example = np.asarray([calendar_features(datetime(2018, 1, 1))])
            predictions = np.asarray(model.predict(example), dtype=np.float64)
            if predictions.shape != (1,) or not np.isfinite(predictions).all() or np.any(predictions < 0):
                raise DataValidationError("Runtime champion returned invalid consumption predictions.")
            with mlflow.start_run(experiment_id=experiment_id, run_name="restore-existing-champion") as run:
                mlflow.set_tags({"operation": "restore_existing_champion", "source_version": manifest["source_version"],
                                 "source_run_id": manifest["source_run_id"],
                                 "promotion_policy": manifest["tags"]["promotion_policy"]})
                mlflow.log_dict(manifest, "source_model_manifest.json")
                logged = log_model(
                    model, name="model", input_example=example,
                    skops_trusted_types=skops.io.get_untrusted_types(data=skops.io.dumps(model)),
                    pip_requirements=[f"{name}=={package_version(name)}" for name in
                                      ("mlflow", "scikit-learn", "numpy", "scipy", "pandas", "skops")],
                    metadata={"feature_contract": FEATURE_CONTRACT, "feature_names": list(FEATURE_NAMES)},
                )
                restored = mlflow.register_model(logged.model_uri, model_name, tags={
                    **manifest["tags"], "restored_from_version": manifest["source_version"],
                    "restored_from_run": manifest["source_run_id"], "initialization": "seed_restore",
                })
                mlflow.log_dict({"model_name": model_name, "model_version": str(restored.version),
                                 "operation": "restore_existing_champion", "training_performed": False},
                                "initialization.json")
            client.set_registered_model_alias(model_name, "champion", restored.version)
            return {"model_name": model_name, "model_version": str(restored.version),
                    "run_id": run.info.run_id, "restored": True}
    except Timeout as exc:
        raise DataValidationError("Runtime initialization is blocked by an active training job.") from exc
