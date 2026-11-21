"""Causal candidate inputs for a fixed 15-minute forecast, separate from the serving calendar model."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from steel_energy_mlops.data import DataValidationError, EnergyRecord
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features

FORECAST_HORIZON = timedelta(minutes=15)
FORECAST_FEATURE_CONTRACT = "history-15min-v1-nonnegative-kwh"
FORECAST_FEATURE_NAMES = (*FEATURE_NAMES, "last_observed_usage_kwh", "last_observed_lagging_reactive_kvarh",
                          "last_observed_leading_reactive_kvarh", "last_observed_lagging_power_factor_percent",
                          "last_observed_leading_power_factor_percent", "history_age_minutes", "history_available")


@dataclass(frozen=True)
class AvailableObservation:
    record: EnergyRecord
    available_at: datetime


@dataclass(frozen=True)
class ForecastInput:
    target_timestamp: datetime
    forecast_origin: datetime
    source_timestamp: datetime | None
    source_available_at: datetime | None
    values: tuple[float, ...]


def _measurement_timestamp(stamp: datetime) -> None:
    if stamp.tzinfo is not None or stamp.minute % 15 or stamp.second or stamp.microsecond:
        raise DataValidationError("Forecast measurements and targets must use the timezone-naive 15-minute grid.")


def _measurements(record: EnergyRecord) -> tuple[float, ...]:
    readings = record.electrical
    if readings is None:
        raise DataValidationError("Historical electrical measurements are required by this candidate contract.")
    measurements = (record.usage_kwh, readings.lagging_reactive_kvarh, readings.leading_reactive_kvarh,
                    readings.lagging_power_factor_percent, readings.leading_power_factor_percent)
    if (any(not math.isfinite(value) or value < 0 for value in measurements)
            or any(value > 100 for value in measurements[3:])):
        raise DataValidationError("Historical readings must be finite, non-negative, with power factors at most 100.")
    return measurements


def _feature_row(target: datetime, origin: datetime, latest: AvailableObservation | None) -> ForecastInput:
    calendar = calendar_features(target)
    if latest is None:
        return ForecastInput(target, origin, None, None, (*calendar, 0., 0., 0., 0., 0., 0., 0.))
    record = latest.record
    measurements = _measurements(record)
    age = (origin - record.timestamp).total_seconds() / 60
    return ForecastInput(target, origin, record.timestamp, latest.available_at, (*calendar, *measurements, age, 1.))


def validate_forecast_observations(observations: tuple[AvailableObservation, ...]) -> None:
    """Reject invalid submitted readings even when their arrival time prevents selection."""
    seen: set[datetime] = set()
    for observation in observations:
        stamp = observation.record.timestamp
        _measurement_timestamp(stamp)
        if observation.available_at.tzinfo is not None or observation.available_at < stamp:
            raise DataValidationError("Availability must be timezone-naive and cannot precede measurement time.")
        if stamp in seen:
            raise DataValidationError("Historical measurement timestamps must be unique.")
        seen.add(stamp)
        _measurements(observation.record)


def build_forecast_inputs(
    targets: tuple[datetime, ...], observations: tuple[AvailableObservation, ...],
) -> tuple[ForecastInput, ...]:
    """Select the latest measurement available at t-15min; retain query order and explicit missingness."""
    for target in targets:
        _measurement_timestamp(target)
    validate_forecast_observations(observations)
    pending = iter(sorted(observations, key=lambda row: row.available_at))
    next_observation = next(pending, None)
    latest: AvailableObservation | None = None
    result: dict[int, ForecastInput] = {}
    for index, target in sorted(enumerate(targets), key=lambda pair: pair[1]):
        try:
            origin = target - FORECAST_HORIZON
        except OverflowError as exc:
            raise DataValidationError("Forecast origin falls outside the supported datetime range.") from exc
        while next_observation is not None and next_observation.available_at <= origin:
            if latest is None or next_observation.record.timestamp > latest.record.timestamp:
                latest = next_observation
            next_observation = next(pending, None)
        result[index] = _feature_row(target, origin, latest)
    return tuple(result[index] for index in range(len(targets)))
