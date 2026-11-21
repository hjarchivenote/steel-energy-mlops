from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from steel_energy_mlops.data import DataValidationError, ElectricalReadings, EnergyRecord
from steel_energy_mlops.features import calendar_features
from steel_energy_mlops.forecast_inputs import (
    AvailableObservation, FORECAST_FEATURE_NAMES, build_forecast_inputs,
)


def observation(hour, minute=0, usage=20., *, available_at=None):
    stamp = datetime(2018, 1, 1, hour, minute)
    record = EnergyRecord(stamp, usage, ElectricalReadings(2., 1., 80., 90.))
    return AvailableObservation(record, available_at or stamp)


def test_target_and_future_measurements_cannot_change_forecast_inputs() -> None:
    target = datetime(2018, 1, 1, 14)
    history = (observation(13, 45, usage=20.), observation(14, usage=50.), observation(14, 15, usage=80.))
    original = build_forecast_inputs((target,), history)[0]
    changed = (history[0], replace(history[1], record=replace(history[1].record, usage_kwh=1e9)),
               replace(history[2], record=replace(history[2].record, usage_kwh=1e10)))
    assert build_forecast_inputs((target,), changed)[0] == original
    assert original.forecast_origin == target - timedelta(minutes=15)
    assert original.source_timestamp == original.source_available_at == original.forecast_origin
    assert original.values[:7] == calendar_features(target)
    assert original.values[7:] == (20., 2., 1., 80., 90., 0., 1.)
    assert len(original.values) == len(FORECAST_FEATURE_NAMES)


def test_delayed_arrival_and_old_late_readings_preserve_query_order() -> None:
    history = (observation(13, usage=10., available_at=datetime(2018, 1, 1, 13, 45)),
               observation(13, 30, usage=30.),
               observation(13, 45, usage=45., available_at=datetime(2018, 1, 1, 14)))
    targets = (datetime(2018, 1, 1, 14, 15), datetime(2018, 1, 1, 14), datetime(2018, 1, 1, 14, 15))
    rows = build_forecast_inputs(targets, history)
    assert [row.target_timestamp for row in rows] == list(targets)
    assert [row.values[7] for row in rows] == [45., 30., 45.]
    assert [row.values[-2] for row in rows] == [15., 15., 15.]
    assert all(row.source_timestamp <= row.source_available_at <= row.forecast_origin for row in rows)


def test_sparse_history_uses_actual_age_instead_of_a_row_based_lag() -> None:
    target = datetime(2018, 1, 1, 14, 30)
    row = build_forecast_inputs((target,), (observation(13),))[0]
    assert row.values[-2:] == (75., 1.)
    assert row.source_timestamp == datetime(2018, 1, 1, 13)


def test_missing_history_is_explicit_and_never_filled_from_the_future() -> None:
    target = datetime(2018, 1, 1, 14)
    row = build_forecast_inputs((target,), (observation(13, 45, available_at=target), observation(14)))[0]
    assert row.source_timestamp is None and row.source_available_at is None
    assert row.values[7:] == (0.,) * 7
    assert build_forecast_inputs((), ()) == ()


def test_origin_crosses_midnight_and_availability_can_have_seconds() -> None:
    target = datetime(2018, 1, 2)
    history = (observation(23, 30, available_at=datetime(2018, 1, 1, 23, 44, 59)),)
    row = build_forecast_inputs((target,), history)[0]
    assert row.forecast_origin == datetime(2018, 1, 1, 23, 45)
    assert row.values[7] == 20. and row.values[-2] == 15.


@pytest.mark.parametrize("location", ["target", "measurement"])
@pytest.mark.parametrize("stamp", [datetime(2018, 1, 1, 14, 1), datetime(2018, 1, 1, 14, tzinfo=timezone.utc)])
def test_invalid_measurement_or_target_time_is_rejected(location, stamp) -> None:
    targets = (stamp,) if location == "target" else (datetime(2018, 1, 1, 15),)
    history = (observation(13),) if location == "target" else (
        AvailableObservation(EnergyRecord(stamp, 10.), stamp),)
    with pytest.raises(DataValidationError, match="15-minute grid"):
        build_forecast_inputs(targets, history)


@pytest.mark.parametrize("available", [datetime(2018, 1, 1, 12),
                                       datetime(2018, 1, 1, 13, tzinfo=timezone.utc)])
def test_availability_cannot_precede_measurement_or_change_timezone(available) -> None:
    with pytest.raises(DataValidationError, match="Availability"):
        build_forecast_inputs((datetime(2018, 1, 1, 15),), (observation(13, available_at=available),))


def test_duplicate_measurement_times_are_ambiguous() -> None:
    with pytest.raises(DataValidationError, match="unique"):
        build_forecast_inputs((datetime(2018, 1, 1, 15),), (observation(13), observation(13, usage=80.)))


def test_candidate_rejects_missing_electrical_history() -> None:
    stamp = datetime(2018, 1, 1, 13)
    with pytest.raises(DataValidationError, match="required"):
        build_forecast_inputs((datetime(2018, 1, 1, 15),), (AvailableObservation(EnergyRecord(stamp, 20.), stamp),))


@pytest.mark.parametrize("failure", ["negative", "nan", "reactive", "factor"])
def test_invalid_historical_values_are_rejected(failure) -> None:
    original = observation(13)
    if failure == "negative":
        record = replace(original.record, usage_kwh=-1.)
    elif failure == "nan":
        record = replace(original.record, usage_kwh=float("nan"))
    else:
        readings = ElectricalReadings(-1., 1., 80., 90.) if failure == "reactive" else (
            ElectricalReadings(2., 1., 101., 90.))
        record = replace(original.record, electrical=readings)
    with pytest.raises(DataValidationError, match="finite"):
        build_forecast_inputs((datetime(2018, 1, 1, 15),), (replace(original, record=record),))
