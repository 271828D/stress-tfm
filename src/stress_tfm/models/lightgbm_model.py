"""LightGBM classifier wrapper for the StressID-TFM benchmark."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from stress_tfm.models.base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM gradient-boosted tree classifier.

    Wraps LGBMClassifier to conform to the uniform BaseModel interface.
    Trained from scratch per split (no pre-training prior).
    """

    name: str = "lightgbm"
    min_context: int = 2  # need at least 1 sample per class

    def __init__(
        self,
        device: str | None = None,
        classes: np.ndarray | None = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs: int | float | str,
    ) -> None:
        super().__init__(device=device, classes=classes)
        lgb_device = "gpu" if device == "cuda" else (device or "cpu")
        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            device=lgb_device,
            verbose=-1,
            **kwargs,
        )
        self._fitted: bool = False

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is not None:
            return self._classes
        if not self._fitted:
            raise RuntimeError("LightGBMModel must be fitted before accessing classes_")
        self._classes = self._model.classes_
        return self._classes

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train the LightGBM classifier."""
        if len(X) == 0:
            raise RuntimeError("LightGBM cannot train on empty data. min_context=2.")
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        with open(path, "wb") as f:
            pickle.dump(self._model, f)