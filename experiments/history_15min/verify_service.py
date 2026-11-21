"""Compare saved-model HTTP outputs with the original frozen validation predictions; never tune."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from steel_energy_mlops.data import SOURCE_COLUMNS, DataValidationError, _parse_record
from steel_energy_mlops.tracking import DEFAULT_MODEL, champion_version, store_client

HERE = Path(__file__).resolve().parent


def verify(url, source, workspace):
    snapshot = json.loads((HERE / "validation.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != snapshot["plan"]["context_source_sha256"]:
        raise DataValidationError("Service verification context differs from the frozen source.")
    client = store_client(workspace)
    incumbent = champion_version(client, DEFAULT_MODEL)
    artifact = client.download_artifacts(snapshot["run_id"], "validation_predictions_and_availability.json")
    reference = json.loads(Path(artifact).read_text())["rows"]
    needed = {row["history_timestamp"]: row["history_available_at"] for row in reference
              if row["history_timestamp"] is not None}
    observations = []
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(SOURCE_COLUMNS):
            raise DataValidationError("Service verification requires the original source schema.")
        for line, values in enumerate(reader, 2):
            stamp = datetime.strptime(values["date"], "%d/%m/%Y %H:%M").isoformat()
            if stamp not in needed:
                continue
            record = _parse_record(values, line)
            observations.append({"timestamp": stamp, "available_at": needed[stamp],
                                 "usage_kwh": record.usage_kwh, **vars(record.electrical)})
    if len(observations) != len(needed):
        raise DataValidationError("Service verification is missing declared historical readings.")
    payload = {"timestamps": [row["target"] for row in reference], "observations": observations}
    request = Request(url.rstrip("/") + "/forecast", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        output = json.load(response)
    actual = output["predictions"]
    values = np.asarray([row["usage_kwh"] for row in actual])
    expected = np.asarray([row["candidate_kwh"] for row in reference])
    if (len(actual) != len(reference) or [row["timestamp"] for row in actual] != payload["timestamps"]
            or not np.allclose(values, expected, rtol=0, atol=1e-8)):
        raise DataValidationError("HTTP predictions differ from the original frozen experiment.")
    for row, old in zip(actual, reference, strict=True):
        fallback = old["midnight_calendar_fallback"]
        if (row["forecast_origin"] != old["origin"]
                or row["method"] != ("calendar_fallback" if fallback else "history_forecast")
                or (not fallback and (row["source_timestamp"] != old["history_timestamp"]
                                      or row["source_available_at"] != old["history_available_at"]))):
            raise DataValidationError("HTTP history selection differs from the original trace.")
    after = champion_version(client, DEFAULT_MODEL)
    if (None if incumbent is None else incumbent.version) != (None if after is None else after.version):
        raise DataValidationError("Calendar champion changed during forecast verification.")
    return {"passed": True, "scope": "saved_model_http_parity", "model_name": output["model_name"],
            "model_version": output["model_version"], "publication_run_id": output["publication_run_id"],
            "prediction_run_id": output["run_id"], "reference_run_id": snapshot["run_id"],
            "rows": len(actual), "history_rows": sum(row["method"] == "history_forecast" for row in actual),
            "calendar_fallback_rows": sum(row["method"] == "calendar_fallback" for row in actual),
            "maximum_absolute_prediction_difference": float(np.max(np.abs(values - expected))),
            "calendar_champion_version": None if after is None else after.version,
            "final_test_evaluated": False, "hyperparameters_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--source", type=Path, default=Path("data/raw/Steel_industry_data.csv"))
    parser.add_argument("--workspace", type=Path, default=Path("artifacts/mlflow"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/history-service/live-proof.json"))
    args = parser.parse_args()
    result = verify(args.url, args.source, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
