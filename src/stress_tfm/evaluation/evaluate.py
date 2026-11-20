"""Model evaluation: fit, predict, and compute metrics for one run."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stress_tfm.evaluation.metrics import compute_metrics
from stress_tfm.models.base import BaseModel
from stress_tfm.utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Outcome of a single (task, model, seed, n_ctx) evaluation.

    Attributes:
        task: Task name ("binary" or "3class").
        model: Model name.
        seed: Random seed used for the split.
        n_context: Number of context subjects.
        y_pred: Predicted class labels.
        y_proba: Predicted class probabilities.
        metrics: Computed metric name-value pairs.
    """

    task: str
    model: str
    seed: int
    n_context: int
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, float]


def evaluate_model(
    model: BaseModel,
    task: str,
    name: str,
    seed: int,
    n_ctx: int,
    X_ctx: pd.DataFrame,
    y_ctx: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> RunResult | None:
    """Fit, predict, and compute metrics for one model run.

    Args:
        model: Initialized model wrapper.
        task: Task name (for logging).
        name: Model name (for logging).
        seed: Random seed.
        n_ctx: Number of context subjects.
        X_ctx: Context features.
        y_ctx: Context labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        RunResult on success, None if the model raised an exception.
    """
    logger.info("[%s] %s | seed=%d | n_ctx=%d", task, name, seed, n_ctx)
    set_seed(seed)

    try:
        model.fit(X_ctx, y_ctx)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
    except Exception:
        logger.exception("[%s] %s failed | seed=%d | n_ctx=%d", task, name, seed, n_ctx)
        return None

    return RunResult(
        task=task,
        model=name,
        seed=seed,
        n_context=n_ctx,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
    )
