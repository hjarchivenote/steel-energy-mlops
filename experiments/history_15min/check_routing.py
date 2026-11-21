"""Check shared routing against frozen experiment inputs, without fitting or scoring a model."""

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.history_15min.context import context_observations
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.forecast_inputs import FORECAST_FEATURE_CONTRACT, build_forecast_inputs
from steel_energy_mlops.forecast_routing import FORECAST_ROUTING_POLICY, MAXIMUM_HISTORY_AGE, MINIMUM_HISTORY_DELAY
from steel_energy_mlops.forecast_routing import build_forecast_routes
from steel_energy_mlops.training_data import load_training_data

HERE = Path(__file__).resolve().parent


def check_routing(source, directory):
    snapshot = json.loads((HERE / "validation.json").read_text())
    plan = snapshot["plan"]
    data = load_training_data(directory)
    if (data.train.sha256 != plan["train_sha256"]
            or data.validation.sha256 != snapshot["validation_sha256"]):
        raise DataValidationError("Input partitions differ from the frozen experiment.")
    partitions = (data.train, data.validation)
    observations = context_observations(source, partitions, plan["context_source_sha256"])
    targets = (*data.train.timestamps, *data.validation.timestamps)
    original = build_forecast_inputs(targets, observations)
    routes = build_forecast_routes(targets, observations)
    for expected, route in zip(original, routes, strict=True):
        if not expected.target_timestamp.hour and not expected.target_timestamp.minute:
            matches = route.reason == "midnight_target" and route.history is None
        else:
            matches = route.history == expected
        if not matches:
            raise DataValidationError("Shared routing differs from the frozen experiment input or fallback.")
    boundary = len(data.train.target)
    training_ages = Counter(str(row.values[-2]) for row in original[:boundary]
                            if row.target_timestamp.hour or row.target_timestamp.minute)
    return {"scope": "input_compatibility_only", "routing_policy": FORECAST_ROUTING_POLICY,
            "feature_contract": FORECAST_FEATURE_CONTRACT, "source_sha256": plan["context_source_sha256"],
            "train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256,
            "minimum_history_delay_minutes": MINIMUM_HISTORY_DELAY.total_seconds() / 60,
            "maximum_age_at_origin_minutes": MAXIMUM_HISTORY_AGE.total_seconds() / 60,
            "training_age_counts_minutes": dict(training_ages),
            "partitions": {name: {"rows": len(partition_routes),
                                  "reasons": dict(Counter(row.reason for row in partition_routes))}
                           for name, partition_routes in (("train", routes[:boundary]),
                                                          ("validation", routes[boundary:]))},
            "identical_history_inputs": sum(row.history is not None for row in routes),
            "passed": True, "model_fitted": False, "accuracy_scored": False,
            "final_test_opened": False, "registry_mutated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/Steel_industry_data.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/history-routing/compatibility.json"))
    args = parser.parse_args()
    result = check_routing(args.source, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
