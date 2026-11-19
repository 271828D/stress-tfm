"""Wrapper for Causilo (Nums AI)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stress_tfm.models.base import BaseModel

logger = logging.getLogger(__name__)


class CausiloModel(BaseModel):
    """Causilo: Latent-Token ICL (4 tokens per row).

    Compresses each row into 4 latent tokens, then performs ICL.
    """

    name: str = "causilo"

    def __init__(self, device: str | None = None, classes: np.ndarray | None = None, **kwargs: Any) -> None:
        super().__init__(device=device, classes=classes)
        from causilo import CausiloClassifier

        self._model = CausiloClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context rows for ICL (no weight updates)."""
        if self._classes is None and len(y) > 0:
            self._classes = np.array(sorted(np.unique(y)))
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)