"""Train-only weekday/hour references with explicit backoff for unseen calendar cells."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.modeling import error_metrics
from steel_energy_mlops.training_data import Partition

PROFILE_STATISTICS = ("mean", "median")
LEVEL_NAMES = ("weekday_hour", "day_type_hour", "hour", "global")


def _keys(stamp: datetime) -> tuple[tuple[int, ...], ...]:
    return ((stamp.weekday(), stamp.hour), (int(stamp.weekday() >= 5), stamp.hour), (stamp.hour,))


@dataclass(frozen=True)
class ScheduleProfile:
    statistic: str
    levels: tuple[dict[tuple[int, ...], float], ...]
    global_value: float

    def predict(self, timestamps: tuple[datetime, ...]) -> tuple[NDArray[np.float64], dict[str, int]]:
        values = []
        support: dict[str, int] = dict.fromkeys(LEVEL_NAMES, 0)
        for stamp in timestamps:
            for name, lookup, key in zip(LEVEL_NAMES, self.levels, _keys(stamp)):
                if key in lookup:
                    values.append(lookup[key])
                    support[name] += 1
                    break
            else:
                values.append(self.global_value)
                support["global"] += 1
        return np.asarray(values, dtype=np.float64), support


def fit_schedule_profile(train: Partition, statistic: str) -> ScheduleProfile:
    """Hourly granularity and backoff order are fixed, never tuned on validation targets."""
    if statistic not in PROFILE_STATISTICS:
        raise DataValidationError("Schedule statistic must be mean or median.")
    error_metrics(train.target, train.target)
    if len(train.timestamps) != train.target.size:
        raise DataValidationError("Schedule timestamps must match the training targets.")
    reducer = np.mean if statistic == "mean" else np.median
    buckets: list[dict[tuple[int, ...], list[float]]] = [{}, {}, {}]
    for stamp, value in zip(train.timestamps, train.target, strict=True):
        for level, key in zip(buckets, _keys(stamp), strict=True):
            level.setdefault(key, []).append(float(value))
    levels = tuple({key: float(reducer(values)) for key, values in level.items()} for level in buckets)
    return ScheduleProfile(statistic, levels, float(reducer(train.target)))
