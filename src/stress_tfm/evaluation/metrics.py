"""Centralized evaluation metrics for the stress-TFM benchmark."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calculate evaluation metrics for binary or multiclass classification.

    Returns NaN for roc_auc when it is undefined:
    - Single-class predictions (all mass on one class).
    - y_proba column count doesn't match the task's class count
      (partial-class context, e.g. 2-of-3 classes present).
    """
    n_classes = len(np.unique(y_true))

    if n_classes == 2:
        avg = "binary"
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        except ValueError:
            roc_auc = float("nan")
    else:
        avg = "macro"
        specificity = None
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = float("nan")

    metrics: dict[str, float] = {
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "roc_auc": roc_auc,
    }
    if specificity is not None:
        metrics["specificity"] = specificity

    logger.info(
        f"Metrics computed. F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}"
    )
    return metrics
