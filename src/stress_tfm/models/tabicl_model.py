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
    min_context: int = 1  # VERIFY: test with empty context; set to 0 if it works

    def __init__(self, device: str | None = None, classes: np.ndarray | None = None, **kwargs: Any) -> None:
        super().__init__(device=device, classes=classes)
        from tabicl import TabICLClassifier

        self._model = TabICLClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        if self._classes is None and len(y) > 0:
            self._classes = np.array(sorted(np.unique(y)))
        if len(X) == 0:
            raise RuntimeError("TabICL cannot operate on empty context. min_context=1.")
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)