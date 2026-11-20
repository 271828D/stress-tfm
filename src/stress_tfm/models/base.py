"""Abstract base class for all tabular foundation model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Uniform interface for all tabular foundation model wrappers."""

    name: str = "base"
    min_context: int = 1

    def __init__(
        self, device: str | None = None, classes: np.ndarray | None = None
    ) -> None:
        """Initialize the model wrapper.

        Args:
            device: Compute device ("cpu", "cuda"). None = auto-detect.
            classes: Label space as a numpy array.
                     e.g., np.array([0, 1]) for binary, np.array([0, 1, 2]) for 3-class.
                     Required for zero-shot; optional for ICL (derived from y if None).
        """
        self.device = device
        self._classes = classes

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order of predict_proba columns."""
        if self._classes is not None:
            return self._classes
        raise AttributeError(
            "classes_ not available. Provide 'classes' at construction "
            "or call fit() with non-empty y first."
        )

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store context for in-context learning (no weight updates).

        If self._classes is None and y is non-empty, derives classes from y.
        If X is empty (zero-shot), self._classes must have been set at construction.
        """
        if self._classes is None and len(y) > 0:
            self._classes = np.unique(y)

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the test set."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the test set."""
        ...
