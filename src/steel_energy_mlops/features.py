"""Predictor construction using information known before the measurement occurs."""

import math
from datetime import datetime

from steel_energy_mlops.data import DataValidationError

FEATURE_NAMES = (
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
)


def calendar_features(timestamp: datetime) -> tuple[float, ...]:
    """Return seven deterministic features; no target or measured load is accepted."""
    if timestamp.tzinfo is not None:
        raise DataValidationError("Use a timezone-naive source timestamp; do not silently convert it to UTC.")
    seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second + timestamp.microsecond / 1_000_000
    daily = math.tau * seconds / 86400
    weekly = math.tau * timestamp.weekday() / 7
    yearly = math.tau * (timestamp.month - 1) / 12
    return (
        math.sin(daily),
        math.cos(daily),
        math.sin(weekly),
        math.cos(weekly),
        math.sin(yearly),
        math.cos(yearly),
        float(timestamp.weekday() >= 5),
    )
