"""Wrapper for TabICLv2 (Inria)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stress_tfm.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabICLModel(BaseModel):
    """TabICLv2: Scalable In-Context Learning for tabular data.

    Two-stage pipeline: column-wise embedding then row-wise ICL attention.
    Scales to 100K+ rows. BSD-3 licensed (commercially usable).
    """

    name: str = "tabicl"
    min_context: int = 1

    def __init__(
        self,
        device: str | None = None,
        classes: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TabICL wrapper."""
        super().__init__(device=device, classes=classes)
        from tabicl import TabICLClassifier

        self._model = TabICLClassifier(**kwargs)
        self._context_classes: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        if self._classes is None and len(y) > 0:
            self._classes = np.array(sorted(np.unique(y)))
        if len(X) == 0:
            raise RuntimeError("TabICL cannot operate on empty context. min_context=1.")
        self._context_classes = np.array(sorted(np.unique(y)))
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        raw = self._model.predict(X).astype(int)
        return raw

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
