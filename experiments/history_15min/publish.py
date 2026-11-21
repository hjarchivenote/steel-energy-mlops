"""Fit the frozen pair on training data once and publish saved estimators for the forecast API."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from experiments.history_15min.context import context_observations, is_midnight
from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.forecast_inputs import build_forecast_inputs
from steel_energy_mlops.forecast_models import FROZEN_CALENDAR_PARAMETERS, publish_forecast_pair
from steel_energy_mlops.training_data import load_training_partition

HERE = Path(__file__).resolve().parent


def publish(source, directory, workspace):
    evidence = json.loads((HERE / "validation.json").read_text())
    plan = evidence["plan"]
    manifest = json.loads((directory / "manifest.json").read_text())
    train = load_training_partition(directory)
    if (manifest["source"]["row_count"] != 5000 or train.sha256 != plan["train_sha256"]
            or manifest["splits"]["validation"]["sha256"] != evidence["validation_sha256"]):
        raise DataValidationError("Forecast publication must retain the frozen 5000-target cohort and partitions.")
    observations = context_observations(source, (train,), plan["context_source_sha256"])
    rows = build_forecast_inputs(train.timestamps, observations)
    features = np.asarray([row.values for row in rows])
    mask = np.asarray([not is_midnight(stamp) for stamp in train.timestamps])
    if int(mask.sum()) != evidence["fitted_label_rows"]:
        raise DataValidationError("Forecast fitting labels differ from the frozen experiment.")
    history = Pipeline([("model", HistGradientBoostingRegressor(**plan["candidate"]))])
    calendar = Pipeline([("model", RandomForestRegressor(**FROZEN_CALENDAR_PARAMETERS))])
    with threadpool_limits(limits=1):
        history.fit(features[mask], train.target[mask])
        calendar.fit(train.features[mask], train.target[mask])
    return publish_forecast_pair(history, calendar, features[mask][:3], evidence, workspace,
                                 training_cutoff=train.timestamps[-1].isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/Steel_industry_data.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--workspace", type=Path, default=Path("artifacts/mlflow"))
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.data, args.workspace), indent=2))


if __name__ == "__main__":
    main()
