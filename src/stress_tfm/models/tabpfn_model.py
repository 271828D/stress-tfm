"""Wrapper for TabPFN-3.5 (Prior Labs / SAP)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stress_tfm.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabPFNModel(BaseModel):
    """TabPFN-3.5: Prior-Fitted Network for tabular data.

    Approximates Bayesian inference over a prior on tabular datasets.
    Predicts in a single forward pass without weight updates.
    """

    name: str = "tabpfn"
    min_context: int = 1

    def __init__(
        self,
        device: str | None = None,
        classes: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TabPFN wrapper."""
        super().__init__(device=device, classes=classes)
        from tabpfn import TabPFNClassifier

        self._model = TabPFNClassifier(**kwargs)
        self._context_classes: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        if self._classes is None and len(y) > 0:
            self._classes = np.array(sorted(np.unique(y)))
        if len(X) == 0:
            raise RuntimeError("TabPFN requires at least 1 labeled context row.")
        self._context_classes = np.array(sorted(np.unique(y)))
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities, aligned to self._classes."""
        if self._classes is None or self._context_classes is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        raw = self._model.predict_proba(X)

        if raw.shape[1] == len(self._classes):
            return raw

        full = np.zeros((raw.shape[0], len(self._classes)))
        for i, c in enumerate(self._context_classes):
            col_idx = list(self._classes).index(int(c))
            full[:, col_idx] = raw[:, i]
        return full
