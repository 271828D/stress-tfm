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
    min_context: int = 1  # requires at least 1 labeled context row

    def __init__(self, device: str | None = None, classes: np.ndarray | None = None, **kwargs: Any) -> None:
        super().__init__(device=device, classes=classes)
        from tabpfn import TabPFNClassifier

        self._model = TabPFNClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        if self._classes is None and len(y) > 0:
            self._classes = np.array(sorted(np.unique(y)))
        if len(X) == 0:
            raise RuntimeError("TabPFN requires at least 1 labeled context row.")
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)