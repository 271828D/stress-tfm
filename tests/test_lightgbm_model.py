"""Tests for the LightGBM model wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stress_tfm.models.lightgbm_model import LightGBMModel
from stress_tfm.models.zoo import get_model


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic binary-classification dataset."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.uniform(0, 1, size=n),
        }
    )
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy()
    return X, y


def test_fit_predict_proba(sample_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Happy path: fit, predict, predict_proba all work and shapes match."""
    X, y = sample_data
    model = get_model("lightgbm", device="cpu")
    model.fit(X, y)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == (len(X),)
    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)
    assert set(y_pred) <= {0, 1}


def test_classes_available_after_fit(
    sample_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """classes_ is accessible after fit and matches expected labels."""
    X, y = sample_data
    model = LightGBMModel(device="cpu")
    model.fit(X, y)

    assert np.array_equal(model.classes_, np.array([0, 1]))


def test_classes_from_constructor() -> None:
    """classes_ is available immediately when provided at construction."""
    model = LightGBMModel(device="cpu", classes=np.array([0, 1]))

    assert np.array_equal(model.classes_, np.array([0, 1]))


def test_classes_raises_before_fit(
    sample_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Edge case: accessing classes_ before fit (no constructor classes) raises."""
    model = LightGBMModel(device="cpu")

    with pytest.raises(RuntimeError, match="classes_ not available"):
        _ = model.classes_


def test_min_context_attribute() -> None:
    """LightGBM requires at least 2 context samples (1 per class)."""
    model = LightGBMModel(device="cpu")
    assert model.min_context == 2


def test_fit_empty_raises() -> None:
    """fit() on empty data raises RuntimeError."""
    model = LightGBMModel(device="cpu")
    X_empty = pd.DataFrame({"f1": [], "f2": [], "f3": []})
    y_empty = np.array([], dtype=int)

    with pytest.raises(RuntimeError, match="empty"):
        model.fit(X_empty, y_empty)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_device_cuda_maps_to_gpu(sample_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Runner passes device='cuda'; wrapper must map to LightGBM's 'gpu'."""
    X, y = sample_data
    model = get_model("lightgbm", device="cuda")
    model.fit(X, y)
    assert isinstance(model, LightGBMModel)
    assert model._fitted