from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.modeling import (
    BASELINES, LEARNED_FAMILIES, ErrorMetrics, calendar_folds, error_metrics, fit_candidates, promotion_decision,
)
from steel_energy_mlops.training_data import load_training_data


def test_errors_are_in_kwh() -> None:
    result = error_metrics(np.asarray([0.0, 2.0]), np.asarray([0.0, 4.0]))
    assert result.as_dict() == pytest.approx({"mae": 1.0, "rmse": 2 ** 0.5})


@pytest.mark.parametrize(
    ("target", "predictions"),
    [([], []), ([[1.0]], [[1.0]]), ([1.0], [1.0, 2.0]), ([1.0], [float("nan")]),
     ([float("inf")], [1.0]), ([1.0], [-1.0])],
)
def test_rejects_invalid_errors(target, predictions) -> None:
    with pytest.raises(DataValidationError):
        error_metrics(np.asarray(target, dtype=np.float64), np.asarray(predictions, dtype=np.float64))


def test_cv_uses_expanding_elapsed_time_windows(prepared_training: Path) -> None:
    stamps = load_training_data(prepared_training).train.timestamps
    folds = calendar_folds(stamps)
    assert len(folds) == 3
    for train, validation in folds:
        assert stamps[int(train[-1])] < stamps[int(validation[0])]
        assert not set(train) & set(validation)
    assert set(folds[0][0]) < set(folds[1][0]) < set(folds[2][0])


@pytest.mark.parametrize("failure", ["count", "length", "order", "empty-window"])
def test_rejects_invalid_cv_windows(failure: str) -> None:
    stamps = tuple(datetime(2018, 1, 1) + timedelta(minutes=15 * i) for i in range(20))
    count = 3
    if failure == "count":
        count = 1
    elif failure == "length":
        stamps = stamps[:3]
    elif failure == "order":
        stamps = stamps[::-1]
    else:
        stamps = (*stamps[:4], datetime(2019, 1, 1))
    with pytest.raises(DataValidationError):
        calendar_folds(stamps, count)


def test_models_are_selected_without_validation_or_test_observations(prepared_training: Path) -> None:
    data = load_training_data(prepared_training)
    candidates, windows = fit_candidates(data.train, forest_trees=2)
    assert {candidate.name for candidate in candidates} == set(BASELINES + LEARNED_FAMILIES)
    assert len(windows) == 3
    for candidate in candidates:
        predictions = candidate.model.predict(data.validation.features)
        assert np.all(np.isfinite(predictions)) and np.all(predictions >= 0)
        assert candidate.cv_mae >= 0 and candidate.trials
    ridge = next(candidate for candidate in candidates if candidate.name == "ridge")
    assert ridge.model.named_steps["scale"].n_samples_seen_ == len(data.train.target)


@pytest.mark.parametrize("families", [(), ("unknown",), ("ridge", "ridge")])
def test_rejects_invalid_family_selection(prepared_training: Path, families: tuple[str, ...]) -> None:
    with pytest.raises(DataValidationError, match="unique families"):
        fit_candidates(load_training_data(prepared_training).train, families)


@pytest.mark.parametrize(("trees", "seed"), [(0, 42), (501, 42), (2, -1)])
def test_rejects_invalid_training_limits(prepared_training: Path, trees: int, seed: int) -> None:
    with pytest.raises(DataValidationError, match="Forest trees"):
        fit_candidates(load_training_data(prepared_training).train, forest_trees=trees, seed=seed)


@pytest.mark.parametrize(
    ("candidate", "baseline_mae", "baseline_rmse", "incumbent", "reason", "eligible"),
    [
        (ErrorMetrics(10, 10), 10, 15, None, "baseline_mae_not_improved", False),
        (ErrorMetrics(9.95, 10), 10, 15, None, "baseline_mae_not_improved", False),
        (ErrorMetrics(9, 16), 10, 15, None, "baseline_rmse_worse", False),
        (ErrorMetrics(9, 10), 10, 15, None, "initial_champion", True),
        (ErrorMetrics(9, 10), 10, 15, ErrorMetrics(9, 12), "incumbent_mae_not_improved", False),
        (ErrorMetrics(8.99, 10), 10, 15, ErrorMetrics(9, 12), "incumbent_mae_not_improved", False),
        (ErrorMetrics(8, 11), 10, 15, ErrorMetrics(9, 10), "incumbent_rmse_worse", False),
        (ErrorMetrics(8, 10), 10, 15, ErrorMetrics(9, 10), "improved_champion", True),
        (ErrorMetrics(0, 0), 0, 0, None, "baseline_mae_not_improved", False),
    ],
)
def test_promotion_gate(candidate, baseline_mae, baseline_rmse, incumbent, reason, eligible) -> None:
    decision = promotion_decision(candidate, baseline_mae, baseline_rmse, incumbent)
    assert decision.reason == reason and decision.eligible is eligible


@pytest.mark.parametrize("improvement", [-0.1, 1.0, float("nan")])
def test_rejects_invalid_gate_threshold(improvement: float) -> None:
    with pytest.raises(DataValidationError):
        promotion_decision(ErrorMetrics(1, 1), 2, 2, minimum_improvement=improvement)


@pytest.mark.parametrize("errors", [ErrorMetrics(float("nan"), 1), ErrorMetrics(-1, 1)])
def test_rejects_invalid_gate_metrics(errors: ErrorMetrics) -> None:
    with pytest.raises(DataValidationError):
        promotion_decision(errors, 2, 2)
