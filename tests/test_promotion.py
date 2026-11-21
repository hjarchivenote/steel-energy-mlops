from dataclasses import asdict

import numpy as np
import pytest

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.modeling import ErrorMetrics
from steel_energy_mlops.promotion import REQUIRED_REFERENCES, promotion_decision, tail_promotion_decision


def references() -> dict[str, ErrorMetrics]:
    return dict.fromkeys(REQUIRED_REFERENCES, ErrorMetrics(10, 15))


@pytest.mark.parametrize("candidate,incumbent,reason", [
    ([1, 11, 25, 45], [2, 12, 20, 40], "tail_checks_passed"),
    ([20, 10, 30, 50], [2, 12, 20, 40], "incumbent_p95_worse"),
    ([0, 10, 18, 38], [30, 10, 20, 40], "incumbent_high_usage_mae_worse"),
    ([0, 10, 25, 45], [30, 10, 20, 60], "incumbent_high_usage_bias_worse"),
])
def test_tail_and_high_usage_errors_can_independently_veto_promotion(candidate, incumbent, reason) -> None:
    decision = tail_promotion_decision(np.asarray([0., 10., 20., 30.]), np.asarray([0., 10., 30., 50.]),
                                       np.asarray(candidate, dtype=float), np.asarray(incumbent, dtype=float))
    assert decision.training_q75 == 22.5 and decision.high_usage_rows == 2
    assert decision.reason == reason and decision.eligible == (reason == "tail_checks_passed")


def test_tail_guard_distinguishes_no_incumbent_and_no_high_usage_observations() -> None:
    train = np.asarray([0., 10., 20., 30.])
    target = np.asarray([0., 5., 10.])
    initial = tail_promotion_decision(train, target, target)
    assert initial.eligible and initial.reason == "no_incumbent_for_tail_comparison"
    assert initial.incumbent is None and initial.high_usage_rows == 0
    assert initial.candidate.high_usage_mae is None and initial.candidate.high_usage_bias is None
    compared = tail_promotion_decision(train, target, target, target + 1)
    assert compared.eligible and compared.reason == "tail_checks_passed"


@pytest.mark.parametrize("location", ["train", "validation", "candidate", "incumbent"])
def test_tail_guard_rejects_nonfinite_or_mismatched_inputs(location) -> None:
    values = {name: np.asarray([1., 2., 3., 4.]) for name in ("train", "validation", "candidate", "incumbent")}
    values[location] = np.asarray([float("nan")])
    with pytest.raises(DataValidationError):
        tail_promotion_decision(values["train"], values["validation"], values["candidate"], values["incumbent"])


@pytest.mark.parametrize("incumbent,reason,count", [
    (None, "initial_champion", 4), (ErrorMetrics(10, 15), "improved_champion", 5),
])
def test_candidate_must_clear_all_references_and_incumbent(incumbent, reason, count) -> None:
    decision = promotion_decision(ErrorMetrics(9.9, 15), references(), incumbent)
    assert decision.eligible and decision.reason == reason
    assert len(decision.checks) == count
    assert all(check.mae_passed and check.rmse_passed for check in decision.checks)
    assert all(check.maximum_mae == 9.9 and check.maximum_rmse == 15 for check in decision.checks)
    assert asdict(decision.checks[0])["reference_mae"] == 10


@pytest.mark.parametrize("reference", REQUIRED_REFERENCES)
@pytest.mark.parametrize("metric", ["mae", "rmse"])
def test_each_reference_can_independently_veto_promotion(reference, metric) -> None:
    comparisons = references()
    comparisons[reference] = ErrorMetrics(9.95, 15) if metric == "mae" else ErrorMetrics(10, 14.99)
    decision = promotion_decision(ErrorMetrics(9.9, 15), comparisons, ErrorMetrics(10, 15))
    assert not decision.eligible
    suffix = "mae_not_improved" if metric == "mae" else "rmse_worse"
    assert decision.reason == f"{reference}_{suffix}"
    assert len(decision.checks) == 5
    assert decision.checks[-1].mae_passed and decision.checks[-1].rmse_passed


@pytest.mark.parametrize("candidate,incumbent,reason", [
    (ErrorMetrics(9, 14), ErrorMetrics(9, 15), "incumbent_mae_not_improved"),
    (ErrorMetrics(8.99, 14), ErrorMetrics(9, 15), "incumbent_mae_not_improved"),
    (ErrorMetrics(8, 14), ErrorMetrics(9, 13.99), "incumbent_rmse_worse"),
])
def test_incumbent_veto_is_independent_of_reference_passes(candidate, incumbent, reason) -> None:
    decision = promotion_decision(candidate, references(), incumbent)
    assert not decision.eligible and decision.reason == reason
    assert all(check.mae_passed and check.rmse_passed for check in decision.checks[:-1])


def test_exact_one_percent_passes_and_immediately_smaller_improvement_fails() -> None:
    assert promotion_decision(ErrorMetrics(9.9, 15), references()).eligible
    decision = promotion_decision(ErrorMetrics(float(np.nextafter(9.9, np.inf)), 15), references())
    assert not decision.eligible and decision.reason == "dummy_mean_mae_not_improved"
    assert not promotion_decision(ErrorMetrics(10, 15), references()).eligible


def test_stricter_improvement_can_be_requested_but_floor_cannot_be_lowered() -> None:
    assert promotion_decision(ErrorMetrics(9, 15), references(), minimum_improvement=0.1).eligible
    assert not promotion_decision(ErrorMetrics(9.01, 15), references(), minimum_improvement=0.1).eligible


def test_zero_error_reference_cannot_be_improved_and_all_failures_are_retained() -> None:
    decision = promotion_decision(ErrorMetrics(0, 0), dict.fromkeys(REQUIRED_REFERENCES, ErrorMetrics(0, 0)))
    assert not decision.eligible
    assert all(not check.mae_passed and check.rmse_passed for check in decision.checks)


@pytest.mark.parametrize("improvement", [-0.1, 0, 0.0099, 1, float("nan"), float("inf")])
def test_invalid_improvement_is_rejected(improvement) -> None:
    with pytest.raises(DataValidationError, match="policy floor"):
        promotion_decision(ErrorMetrics(1, 2), references(), minimum_improvement=improvement)


@pytest.mark.parametrize("failure", ["missing", "extra", "empty"])
def test_incomplete_or_unexpected_reference_sets_fail_closed(failure) -> None:
    comparisons = references()
    if failure == "missing":
        del comparisons["weekday_hour_median"]
    elif failure == "extra":
        comparisons["other"] = ErrorMetrics(10, 15)
    else:
        comparisons = {}
    with pytest.raises(DataValidationError, match="exactly"):
        promotion_decision(ErrorMetrics(1, 2), comparisons)


@pytest.mark.parametrize("location", ["candidate", "reference", "incumbent"])
@pytest.mark.parametrize("invalid", [ErrorMetrics(float("nan"), 1), ErrorMetrics(-1, 1),
                                     ErrorMetrics(1, float("inf")), ErrorMetrics(1, -1)])
def test_invalid_metrics_cannot_pass_any_comparison(location, invalid) -> None:
    candidate, incumbent = ErrorMetrics(1, 2), ErrorMetrics(10, 15)
    comparisons = references()
    if location == "candidate":
        candidate = invalid
    elif location == "reference":
        comparisons["weekday_hour_median"] = invalid
    else:
        incumbent = invalid
    with pytest.raises(DataValidationError, match="finite non-negative"):
        promotion_decision(candidate, comparisons, incumbent)
