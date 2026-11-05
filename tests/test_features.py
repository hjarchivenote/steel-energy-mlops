import math
from datetime import datetime, timezone

import pytest

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.features import FEATURE_NAMES, calendar_features


def test_known_monday_midnight_calendar_features() -> None:
    assert calendar_features(datetime(2018, 1, 1)) == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    assert len(FEATURE_NAMES) == 7


@pytest.mark.parametrize(("hour", "sine", "cosine"), [(6, 1.0, 0.0), (12, 0.0, -1.0), (18, -1.0, 0.0)])
def test_clock_positions(hour: int, sine: float, cosine: float) -> None:
    assert calendar_features(datetime(2018, 1, 1, hour))[:2] == pytest.approx((sine, cosine), abs=1e-12)


@pytest.mark.parametrize(("day", "weekend"), [(5, 0.0), (6, 1.0), (7, 1.0), (8, 0.0)])
def test_weekend_comes_from_date(day: int, weekend: float) -> None:
    assert calendar_features(datetime(2018, 1, day))[-1] == weekend


def test_cyclic_time_is_continuous_across_midnight() -> None:
    before = calendar_features(datetime(2018, 1, 1, 23, 59))[:2]
    after = calendar_features(datetime(2018, 1, 2))[:2]
    assert math.dist(before, after) < 0.005


def test_minute_and_second_precision_is_preserved() -> None:
    whole = calendar_features(datetime(2018, 1, 1, 6))
    fractional = calendar_features(datetime(2018, 1, 1, 6, 15, 30, 500000))
    assert whole[:2] != fractional[:2]
    assert whole[2:] == fractional[2:]


@pytest.mark.parametrize("month", range(1, 13))
def test_cyclic_pairs_are_finite_and_unit_length(month: int) -> None:
    features = calendar_features(datetime(2018, month, 15, 12, 30))
    assert all(math.isfinite(value) and -1 <= value <= 1 for value in features)
    for index in (0, 2, 4):
        assert features[index] ** 2 + features[index + 1] ** 2 == pytest.approx(1.0)


def test_features_do_not_silently_change_source_timezone() -> None:
    with pytest.raises(DataValidationError, match="timezone-naive"):
        calendar_features(datetime(2018, 1, 1, tzinfo=timezone.utc))
