"""Train-only time-based model selection and explicit, comparable error metrics."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.training_data import Partition

LEARNED_FAMILIES = ("ridge", "random_forest", "hist_gradient_boosting")
BASELINES = ("dummy_mean", "dummy_median")


@dataclass(frozen=True)
class ErrorMetrics:
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Candidate:
    name: str
    model: Pipeline
    cv_mae: float
    parameters: dict[str, Any]
    trials: list[dict[str, Any]]


def error_metrics(target: NDArray[np.float64], predictions: NDArray[np.float64]) -> ErrorMetrics:
    if target.ndim != 1 or target.shape != predictions.shape or target.size < 1:
        raise DataValidationError("Targets and predictions must be nonempty, matching one-dimensional arrays.")
    if not np.isfinite(target).all() or not np.isfinite(predictions).all() or np.any(predictions < 0):
        raise DataValidationError("Model outputs must be finite, non-negative consumption values.")
    return ErrorMetrics(
        float(mean_absolute_error(target, predictions)), float(root_mean_squared_error(target, predictions))
    )


def calendar_folds(
    timestamps: tuple[datetime, ...], count: int = 3
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Use comparable elapsed-time windows even though the sample has irregular row gaps."""
    if count < 2 or len(timestamps) < count + 2:
        raise DataValidationError("At least two CV folds and enough observations are required.")
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise DataValidationError("Cross-validation timestamps must be strictly increasing.")
    duration = timestamps[-1] - timestamps[0] + timedelta(microseconds=1)
    folds = []
    for number in range(1, count + 1):
        start = timestamps[0] + duration * number / (count + 1)
        end = timestamps[0] + duration * (number + 1) / (count + 1)
        train = np.asarray([i for i, stamp in enumerate(timestamps) if stamp < start], dtype=np.int64)
        validation = np.asarray([i for i, stamp in enumerate(timestamps) if start <= stamp < end], dtype=np.int64)
        if min(train.size, validation.size) < 2:
            raise DataValidationError("An elapsed-time CV window contains fewer than two observations.")
        folds.append((train, validation))
    return folds


def fit_candidates(
    train: Partition, families: tuple[str, ...] = LEARNED_FAMILIES, forest_trees: int = 100, seed: int = 42
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    if not families or len(set(families)) != len(families) or set(families) - set(LEARNED_FAMILIES):
        raise DataValidationError("Choose unique families from ridge, random_forest, hist_gradient_boosting.")
    if forest_trees < 1 or forest_trees > 500 or seed < 0:
        raise DataValidationError("Forest trees must be between 1 and 500; the seed must be non-negative.")
    folds = calendar_folds(train.timestamps)
    # sklearn's inferred annotations omit valid bool/str union options documented by its API.
    boosting_options: dict[str, Any] = {"early_stopping": False}
    search_options: dict[str, Any] = {"error_score": "raise"}
    specifications: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "dummy_mean": (Pipeline([("model", DummyRegressor(strategy="mean"))]), {}),
        "dummy_median": (Pipeline([("model", DummyRegressor(strategy="median"))]), {}),
        "ridge": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", TransformedTargetRegressor(regressor=Ridge(), func=np.sqrt, inverse_func=np.square)),
            ]),
            {"model__regressor__alpha": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline([("model", RandomForestRegressor(n_estimators=forest_trees, random_state=seed, n_jobs=1))]),
            {"model__max_depth": [6, None], "model__min_samples_leaf": [2, 8]},
        ),
        "hist_gradient_boosting": (
            Pipeline([("model", HistGradientBoostingRegressor(
                loss="poisson", max_iter=100, random_state=seed, **boosting_options
            ))]),
            {"model__max_leaf_nodes": [15, 31], "model__l2_regularization": [0.0, 1.0]},
        ),
    }
    candidates = []
    for name in (*BASELINES, *families):
        estimator, grid = specifications[name]
        search = GridSearchCV(
            estimator, grid, cv=folds, scoring="neg_mean_absolute_error", n_jobs=1, **search_options
        )
        with threadpool_limits(limits=1):
            search.fit(train.features, train.target)
        trials = [
            {"parameters": params, "cv_mae": -float(score), "cv_mae_std": float(std)}
            for params, score, std in zip(
                search.cv_results_["params"], search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"], strict=True
            )
        ]
        candidates.append(
            Candidate(name, search.best_estimator_, -float(search.best_score_), search.best_params_, trials)
        )
    windows = [
        {"train_rows": len(left), "validation_rows": len(right),
         "train_last_timestamp": train.timestamps[int(left[-1])].isoformat(),
         "validation_first_timestamp": train.timestamps[int(right[0])].isoformat(),
         "validation_last_timestamp": train.timestamps[int(right[-1])].isoformat()}
        for left, right in folds
    ]
    return candidates, windows


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reason: str


def promotion_decision(
    candidate: ErrorMetrics,
    baseline_mae: float,
    baseline_rmse: float,
    incumbent: ErrorMetrics | None = None,
    minimum_improvement: float = 0.01,
) -> PromotionDecision:
    values = [candidate.mae, candidate.rmse, baseline_mae, baseline_rmse]
    if incumbent is not None:
        values.extend([incumbent.mae, incumbent.rmse])
    if not 0 <= minimum_improvement < 1 or any(not np.isfinite(value) or value < 0 for value in values):
        raise DataValidationError("Promotion requires finite non-negative errors and improvement in [0, 1).")
    if candidate.mae >= baseline_mae or candidate.mae > baseline_mae * (1 - minimum_improvement):
        return PromotionDecision(False, "baseline_mae_not_improved")
    if candidate.rmse > baseline_rmse:
        return PromotionDecision(False, "baseline_rmse_worse")
    if incumbent is None:
        return PromotionDecision(True, "initial_champion")
    if candidate.mae >= incumbent.mae or candidate.mae > incumbent.mae * (1 - minimum_improvement):
        return PromotionDecision(False, "incumbent_mae_not_improved")
    if candidate.rmse > incumbent.rmse:
        return PromotionDecision(False, "incumbent_rmse_worse")
    return PromotionDecision(True, "improved_champion")
