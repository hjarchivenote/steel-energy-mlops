from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from steel_energy_mlops.data import DataValidationError, ElectricalReadings, EnergyRecord
from steel_energy_mlops.features import calendar_features
from steel_energy_mlops.forecast_inputs import AvailableObservation, build_forecast_inputs
from steel_energy_mlops.forecast_routing import build_forecast_routes


def reading(stamp, usage=20., arrived=None):
    return AvailableObservation(EnergyRecord(stamp, usage, ElectricalReadings(2., 1., 80., 90.)),
                                arrived if arrived is not None else stamp + timedelta(minutes=15))


def test_normal_forecast_exactly_matches_the_frozen_input_construction():
    target = datetime(2018, 1, 2, 14)
    history = (reading(target - timedelta(minutes=45), 10.), reading(target - timedelta(minutes=30), 20.),
               reading(target - timedelta(minutes=15), 100.), reading(target, 1000.))
    route, = build_forecast_routes((target,), history)
    assert route.method == "history_forecast" and route.reason == "history_ready"
    assert route.history == build_forecast_inputs((target,), history)[0]
    assert route.history.source_timestamp == target - timedelta(minutes=30)
    assert route.calendar_values == calendar_features(target)


def test_early_arrival_never_bypasses_the_frozen_15_minute_delay():
    target = datetime(2018, 1, 2, 14)
    old = reading(target - timedelta(minutes=30), 20., arrived=target - timedelta(minutes=30))
    fresh = reading(target - timedelta(minutes=15), 100., arrived=target - timedelta(minutes=15))
    route, = build_forecast_routes((target,), (old, fresh))
    assert route.history is not None
    assert route.history.source_timestamp == old.record.timestamp
    assert route.history.source_available_at == old.available_at  # Preserve actual arrival, not simulated arrival.
    assert route.history.values[7] == 20.


def test_delayed_and_out_of_order_arrivals_select_the_latest_eligible_measurement():
    target = datetime(2018, 1, 2, 14)
    old = reading(target - timedelta(minutes=45), 10., arrived=target - timedelta(minutes=16))
    late = reading(target - timedelta(minutes=30), 100., arrived=target - timedelta(minutes=14))
    route, = build_forecast_routes((target,), (late, old))
    assert route.history is not None and route.history.source_timestamp == old.record.timestamp
    assert route.history.values[-2] == 30.


@pytest.mark.parametrize("age,expected", [(30, "history_ready"), (45, "stale_history")])
def test_supported_history_age_boundary_is_inclusive(age, expected):
    target = datetime(2018, 1, 2, 14)
    observation = reading(target - timedelta(minutes=15 + age))
    route, = build_forecast_routes((target,), (observation,))
    assert route.reason == expected
    assert (route.history is not None) == (age == 30)
    assert route.calendar_values == calendar_features(target)


@pytest.mark.parametrize("failure", ["empty", "late", "future"])
def test_unavailable_history_routes_to_calendar_with_an_explicit_reason(failure):
    target = datetime(2018, 1, 2, 14)
    observation = reading(target - timedelta(minutes=30), arrived=target)
    observations = () if failure == "empty" else (
        reading(target + timedelta(minutes=15)) if failure == "future" else observation,)
    route, = build_forecast_routes((target,), observations)
    assert route.method == "calendar_fallback" and route.reason == "missing_history"
    assert route.history is None and route.calendar_values == calendar_features(target)


def test_midnight_observations_are_excluded_without_moving_their_dates():
    midnight = datetime(2018, 1, 2)
    observations = (reading(midnight - timedelta(minutes=15), 20.), reading(midnight, 1000.))
    targets = (midnight + timedelta(minutes=30), midnight, midnight + timedelta(minutes=30))
    routes = build_forecast_routes(targets, observations)
    assert [route.target_timestamp for route in routes] == list(targets)
    assert routes[1].reason == "midnight_target" and routes[1].history is None
    assert routes[0] == routes[2]
    assert routes[0].history is not None
    assert routes[0].history.source_timestamp == midnight - timedelta(minutes=15)
    assert routes[0].history.values[7] == 20. and routes[0].history.values[-2] == 30.
    assert build_forecast_routes((), ()) == ()


@pytest.mark.parametrize("location", ["unused_future", "excluded_midnight"])
def test_invalid_readings_cannot_hide_in_unused_or_excluded_history(location):
    target = datetime(2018, 1, 2, 14)
    stamp = target + timedelta(minutes=15) if location == "unused_future" else datetime(2018, 1, 2)
    invalid = reading(stamp, usage=float("nan"))
    with pytest.raises(DataValidationError, match="finite"):
        build_forecast_routes((target,), (reading(target - timedelta(minutes=30)), invalid))


def test_duplicate_excluded_midnight_records_still_fail_request_validation():
    midnight = datetime(2018, 1, 2)
    with pytest.raises(DataValidationError, match="unique"):
        build_forecast_routes((midnight,), (reading(midnight), reading(midnight, usage=100.)))


def test_unrepresentable_origin_and_eligibility_are_validation_errors():
    with pytest.raises(DataValidationError, match="Forecast origin"):
        build_forecast_routes((datetime.min,), ())
    stamp = datetime(9999, 12, 31, 23, 45)
    with pytest.raises(DataValidationError, match="Historical eligibility"):
        build_forecast_routes((), (reading(stamp, arrived=stamp),))


def test_adding_valid_future_measurements_cannot_change_any_route():
    target = datetime(2018, 1, 2, 14)
    history = (reading(target - timedelta(minutes=30)),)
    original = build_forecast_routes((target,), history)
    future = reading(target, usage=1e12)
    assert build_forecast_routes((target,), (*history, future)) == original
    assert build_forecast_routes((target,), (*history, replace(future, record=replace(
        future.record, usage_kwh=1e15)))) == original
