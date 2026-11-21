"""Shared routing conditions for the history candidate; this module performs no model inference."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Literal

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES
from steel_energy_mlops.forecast_inputs import AvailableObservation, ForecastInput, build_forecast_inputs
from steel_energy_mlops.forecast_inputs import validate_forecast_observations

FORECAST_ROUTING_POLICY = "history-15min-routing-v1"
MINIMUM_HISTORY_DELAY = timedelta(minutes=15)
# Non-midnight training inputs have ages of 15 or 30 minutes at forecast origin.
# This is a support boundary, not a delay-tolerance accuracy claim.
MAXIMUM_HISTORY_AGE = timedelta(minutes=30)
RoutingReason = Literal["history_ready", "midnight_target", "missing_history", "stale_history"]


@dataclass(frozen=True)
class ForecastRoute:
    target_timestamp: datetime
    forecast_origin: datetime
    calendar_values: tuple[float, ...]
    history: ForecastInput | None
    reason: RoutingReason

    @property
    def method(self) -> Literal["history_forecast", "calendar_fallback"]:
        return "history_forecast" if self.history is not None else "calendar_fallback"


def build_forecast_routes(
    targets: tuple[datetime, ...], observations: tuple[AvailableObservation, ...],
) -> tuple[ForecastRoute, ...]:
    """Preserve query order; keep unsupported requests out of the history model's input batch."""
    validate_forecast_observations(observations)
    arrival_times = {row.record.timestamp: row.available_at for row in observations}
    try:
        eligible = tuple(
            replace(row, available_at=max(row.available_at, row.record.timestamp + MINIMUM_HISTORY_DELAY))
            for row in observations if row.record.timestamp.hour or row.record.timestamp.minute
        )
    except OverflowError as exc:
        raise DataValidationError("Historical eligibility falls outside the supported datetime range.") from exc
    inputs = build_forecast_inputs(targets, eligible)
    routes = []
    for row in inputs:
        reason: RoutingReason = "history_ready"
        if not row.target_timestamp.hour and not row.target_timestamp.minute:
            reason = "midnight_target"
        elif row.source_timestamp is None:
            reason = "missing_history"
        elif row.forecast_origin - row.source_timestamp > MAXIMUM_HISTORY_AGE:
            reason = "stale_history"
        history = None
        if reason == "history_ready" and row.source_timestamp is not None:
            # Eligibility includes the delay embargo. Keep the caller's actual arrival time in the trace.
            history = replace(row, source_available_at=arrival_times[row.source_timestamp])
        routes.append(ForecastRoute(row.target_timestamp, row.forecast_origin,
                                    row.values[:len(FEATURE_NAMES)], history, reason))
    return tuple(routes)
