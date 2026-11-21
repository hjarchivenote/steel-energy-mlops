"""Training-only input audit and a reproducible candidate forecast contract; no model selection."""

import csv
import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from steel_energy_mlops.data import SOURCE_COLUMNS, DataValidationError, EnergyRecord, _parse_record
from steel_energy_mlops.forecast_inputs import AvailableObservation, FORECAST_FEATURE_CONTRACT, FORECAST_FEATURE_NAMES
from steel_energy_mlops.forecast_inputs import build_forecast_inputs
from steel_energy_mlops.tracking import DEFAULT_WORKSPACE, open_store
from steel_energy_mlops.training import _git_provenance
from steel_energy_mlops.training_data import Partition, load_training_partition

DATASET_SOURCE = "https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption"
INPUT_DECISIONS = {
    "date": "Known calendar input; derive the calendar features from the requested target timestamp.",
    "Usage_kWh": "Label at target time; only an earlier observation available by forecast origin may be an input.",
    "Lagging_Current_Reactive.Power_kVarh": "Historical observation only; exclude the target row's measurement.",
    "Leading_Current_Reactive_Power_kVarh": "Historical observation only; exclude the target row's measurement.",
    "Lagging_Current_Power_Factor": "Historical observation only; exclude the target row's measurement.",
    "Leading_Current_Power_Factor": "Historical observation only; exclude the target row's measurement.",
    "CO2(tCO2)": "Exclude: units and derivation are unverified; audit its association with the training target.",
    "NSM": "Calendar information already derived from date; do not add duplicate raw predictors.",
    "WeekStatus": "Calendar information already derived from date; do not add duplicate raw predictors.",
    "Day_of_week": "Calendar information already derived from date; do not add duplicate raw predictors.",
    "Load_Type": "Exclude until availability and derivation before prediction are established.",
}


def _training_source(
    source: Path, train: Partition, manifest: dict[str, Any],
) -> tuple[tuple[EnergyRecord, ...], list[float]]:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["source"]["sha256"]:
        raise DataValidationError("Input audit source differs from the prepared training provenance.")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    if next(reader, None) != list(SOURCE_COLUMNS):
        raise DataValidationError("Input audit source must retain the original 11-column schema.")
    expected = set(train.timestamps)
    selected: dict[datetime, tuple[EnergyRecord, float]] = {}
    for line, values in enumerate(reader, 2):
        if len(values) != len(SOURCE_COLUMNS):
            raise DataValidationError("Input audit source must contain exactly 11 cells per row.")
        stamp = datetime.strptime(values[0], "%d/%m/%Y %H:%M")
        if stamp in expected:
            if stamp in selected:
                raise DataValidationError("Input audit source contains a duplicate training timestamp.")
            row: dict[str, str] = dict(zip(SOURCE_COLUMNS, values, strict=True))
            selected[stamp] = (_parse_record(row, line), float(row["CO2(tCO2)"]))
    if selected.keys() != expected:
        raise DataValidationError("Input audit source must contain every prepared training observation.")
    records = tuple(selected[stamp][0] for stamp in train.timestamps)
    co2 = [selected[stamp][1] for stamp in train.timestamps]
    if not np.allclose([row.usage_kwh for row in records], train.target, rtol=0, atol=1e-9):
        raise DataValidationError("Input audit source targets differ from the prepared training observations.")
    return records, co2


def audit_inputs(
    source: Path, directory: Path, reporting_delay_minutes: int, workspace: Path = DEFAULT_WORKSPACE,
) -> dict[str, Any]:
    if type(reporting_delay_minutes) is not int or reporting_delay_minutes < 0:
        raise DataValidationError("Declare a non-negative integer reporting delay for this offline audit.")
    train = load_training_partition(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records, co2 = _training_source(source, train, manifest)
    try:
        delay = timedelta(minutes=reporting_delay_minutes)
        observations = tuple(AvailableObservation(row, row.timestamp + delay) for row in records)
    except OverflowError as exc:
        raise DataValidationError("Reporting delay exceeds the supported datetime range.") from exc
    inputs = build_forecast_inputs(train.timestamps, observations)
    ages = [row.values[-2] for row in inputs if row.source_timestamp is not None]
    gaps = [(right - left).total_seconds() / 60 for left, right in zip(train.timestamps, train.timestamps[1:])]
    electrical = [row.electrical for row in records]
    # Source parsing supplies these fields; the input builder also rejects missing measurements.
    readings = [row for row in electrical if row is not None]
    reactive = np.asarray([row.lagging_reactive_kvarh for row in readings])
    power_factor = np.asarray([row.lagging_power_factor_percent / 100 for row in readings])
    mask = (power_factor > 0) & (power_factor < 1)
    reconstructed = reactive[mask] * power_factor[mask] / np.sqrt(1 - power_factor[mask] ** 2)
    errors = np.abs(train.target[mask] - reconstructed)
    co2_correlation = float(np.corrcoef(co2, train.target)[0, 1]) if np.ptp(co2) > 0 else None
    summary: dict[str, Any] = {
        "scope": "training_observations_only", "train_rows": len(records), "train_sha256": train.sha256,
        "source_sha256": manifest["source"]["sha256"], "source_reference": DATASET_SOURCE,
        "task": {"id": "forecast-15min-history-v1", "target": "Usage_kWh", "unit": "kWh",
                 "forecast_origin": "target_timestamp minus 15 minutes", "forecast_horizon_minutes": 15,
                 "history_rule": "measurement_timestamp <= available_at <= forecast_origin",
                 "offline_reporting_delay_minutes": reporting_delay_minutes,
                 "reporting_delay_is_assumed": True,
                 "assumption": "The source has no ingestion timestamps; this audit simulates the declared delay.",
                 "evaluation": "Rolling backtests must use identical availability rules and include persistence.",
                 "readiness": "Candidate input design only; accuracy and operational tolerances are not validated."},
        "candidate_feature_contract": FORECAST_FEATURE_CONTRACT, "candidate_features": list(FORECAST_FEATURE_NAMES),
        "field_decisions": INPUT_DECISIONS,
        "sampling": {"median_gap_minutes": float(np.median(gaps)), "maximum_gap_minutes": max(gaps),
                     "exact_15_minute_transition_fraction": sum(gap == 15 for gap in gaps) / len(gaps)},
        "history": {"available_rows": len(ages), "missing_rows": len(inputs) - len(ages),
                    "median_age_minutes_at_origin": float(np.median(ages)) if ages else None,
                    "maximum_age_minutes_at_origin": max(ages) if ages else None,
                    "missing_encoding": "Zero historical values with history_available=0; no backward fill."},
        "same_row_proxy_diagnostics": {
            "eligible_rows": int(mask.sum()), "excluded_singular_power_factor_rows": int((~mask).sum()),
            "formula": "Q * PF / sqrt(1 - PF**2); PF is divided by 100",
            "median_absolute_reconstruction_error_kwh": float(np.median(errors)) if errors.size else None,
            "mean_absolute_reconstruction_error_kwh": float(np.mean(errors)) if errors.size else None,
            "co2_target_correlation": co2_correlation, "co2_unique_values": len(set(co2)),
            "interpretation": "Diagnostic association/near-identity, not proof of the source's computation process. "
                              "It does not establish availability before the forecast origin.",
        },
        "model_trained": False, "validation_evaluated": False, "test_evaluated": False, "registry_mutated": False,
    }
    _, experiment_id = open_store(workspace, "steel-energy-input-audit")
    with mlflow.start_run(experiment_id=experiment_id, run_name="forecast-input-availability-audit") as run:
        summary["run_id"] = run.info.run_id
        mlflow.log_dict(summary, "input_audit.json")
        mlflow.log_dict(manifest, "data/preparation_manifest.json")
        mlflow.log_dict(_git_provenance(), "source/git_provenance.json")
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(("target_timestamp", "forecast_origin", "source_timestamp", "source_available_at",
                         *FORECAST_FEATURE_NAMES))
        for row in inputs:
            writer.writerow((row.target_timestamp.isoformat(), row.forecast_origin.isoformat(),
                             row.source_timestamp.isoformat() if row.source_timestamp is not None else "",
                             row.source_available_at.isoformat() if row.source_available_at is not None else "",
                             *row.values))
        mlflow.log_text(buffer.getvalue(), "candidate_train_inputs.csv")
        for path in sorted(Path(__file__).parent.glob("*.py")):
            mlflow.log_artifact(str(path), artifact_path="source/application")
        mlflow.set_tags({"purpose": "input_contract_audit", "statistical_scope": "train_only",
                         "validation_evaluated": "false", "test_evaluated": "false", "registry_mutated": "false"})
    return summary
