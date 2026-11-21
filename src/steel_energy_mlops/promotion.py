"""Versioned promotion policy: clear every fixed reference and the incumbent on MAE and RMSE."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.modeling import BASELINES, ErrorMetrics, error_metrics

PROMOTION_POLICY = "calendar-reference-v3"
MINIMUM_IMPROVEMENT = 0.01
REQUIRED_REFERENCES = (*BASELINES, "weekday_hour_mean", "weekday_hour_median")


@dataclass(frozen=True)
class ReferenceCheck:
    reference: str
    reference_mae: float
    reference_rmse: float
    maximum_mae: float
    maximum_rmse: float
    mae_passed: bool
    rmse_passed: bool


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reason: str
    checks: tuple[ReferenceCheck, ...]


@dataclass(frozen=True)
class TailMetrics:
    p95_absolute_error: float
    high_usage_mae: float | None
    high_usage_bias: float | None


@dataclass(frozen=True)
class TailDecision:
    eligible: bool
    reason: str
    training_q75: float
    high_usage_rows: int
    candidate: TailMetrics
    incumbent: TailMetrics | None


def tail_promotion_decision(
    train_target: NDArray[np.float64], validation_target: NDArray[np.float64],
    candidate_prediction: NDArray[np.float64], incumbent_prediction: NDArray[np.float64] | None = None,
) -> TailDecision:
    """Prospective guard against worse tail errors and high-usage bias; the cutoff is training-only."""
    error_metrics(train_target, train_target)
    error_metrics(validation_target, validation_target)
    threshold = float(np.quantile(train_target, 0.75))
    high = validation_target > threshold
    count = int(high.sum())

    def metrics(prediction: NDArray[np.float64]) -> TailMetrics:
        error_metrics(validation_target, prediction)
        error = prediction - validation_target
        return TailMetrics(float(np.quantile(np.abs(error), 0.95)),
                           float(np.mean(np.abs(error[high]))) if count else None,
                           float(np.mean(error[high])) if count else None)

    candidate = metrics(candidate_prediction)
    incumbent = None if incumbent_prediction is None else metrics(incumbent_prediction)
    reason = "tail_checks_passed"
    if incumbent is None:
        reason = "no_incumbent_for_tail_comparison"
    elif candidate.p95_absolute_error > incumbent.p95_absolute_error:
        reason = "incumbent_p95_worse"
    elif candidate.high_usage_mae is not None and incumbent.high_usage_mae is not None:
        if candidate.high_usage_mae > incumbent.high_usage_mae:
            reason = "incumbent_high_usage_mae_worse"
        elif (candidate.high_usage_bias is not None and incumbent.high_usage_bias is not None
              and abs(candidate.high_usage_bias) > abs(incumbent.high_usage_bias)):
            reason = "incumbent_high_usage_bias_worse"
    return TailDecision(not reason.endswith("worse"), reason, threshold, count, candidate, incumbent)


def validate_improvement(minimum_improvement: float) -> None:
    if not MINIMUM_IMPROVEMENT <= minimum_improvement < 1:
        raise DataValidationError("Minimum improvement must be a fraction in [0.01, 1); the policy floor is 1%.")


def promotion_decision(
    candidate: ErrorMetrics,
    references: Mapping[str, ErrorMetrics],
    incumbent: ErrorMetrics | None = None,
    minimum_improvement: float = MINIMUM_IMPROVEMENT,
) -> PromotionDecision:
    validate_improvement(minimum_improvement)
    if set(references) != set(REQUIRED_REFERENCES):
        raise DataValidationError("Promotion requires exactly the two constant and two weekday/hour references.")
    comparisons = [(name, references[name]) for name in REQUIRED_REFERENCES]
    if incumbent is not None:
        comparisons.append(("incumbent", incumbent))
    values = [candidate.mae, candidate.rmse]
    values.extend(value for _, metrics in comparisons for value in (metrics.mae, metrics.rmse))
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise DataValidationError("Promotion requires finite non-negative errors for every comparison.")
    checks = tuple(ReferenceCheck(
        reference=name, reference_mae=metrics.mae, reference_rmse=metrics.rmse,
        maximum_mae=metrics.mae * (1 - minimum_improvement), maximum_rmse=metrics.rmse,
        mae_passed=candidate.mae < metrics.mae and candidate.mae <= metrics.mae * (1 - minimum_improvement),
        rmse_passed=candidate.rmse <= metrics.rmse,
    ) for name, metrics in comparisons)
    # Evaluate and retain every check, even when the first reference already rejects the candidate.
    for check in checks:
        if not check.mae_passed:
            return PromotionDecision(False, f"{check.reference}_mae_not_improved", checks)
        if not check.rmse_passed:
            return PromotionDecision(False, f"{check.reference}_rmse_worse", checks)
    return PromotionDecision(True, "initial_champion" if incumbent is None else "improved_champion", checks)
