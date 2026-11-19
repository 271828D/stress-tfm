"""Wrapper for TabDPT (Layer 6 AI)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stress_tfm.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabDPTModel(BaseModel):
    """TabDPT: Denoising Transformer with FAISS retrieval.

    Pretrained on real-world tables (not synthetic). Uses retrieval to
    find relevant context rows, then ICL for prediction.
    """

    name: str = "tabdpt"
    min_context: int = 1  # FAISS index requires at least 1 context row

    def __init__(self, device: str | None = None, classes: np.ndarray | None = None, **kwargs: Any) -> None:
        super().__init__(device=device, classes=classes)
        from tabdpt import TabDPTClassifier

        self._model = TabDPTClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context and build retrieval index (no weight updates)."""
        if self._classes is None and len(y) > 0:
            self._classes = np.array(sorted(np.unique(y)))
        if len(X) == 0:
            raise RuntimeError("TabDPT cannot build a retrieval index on empty context.")
        self._model.fit(X.values.copy(), y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.values.copy()).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X.values.copy())