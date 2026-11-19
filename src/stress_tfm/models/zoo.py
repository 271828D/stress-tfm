"""Model registry and factory for the stress-tfm benchmark."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from stress_tfm.models.base import BaseModel
from stress_tfm.models.causilo_model import CausiloModel
from stress_tfm.models.kumo_model import KumoModel
from stress_tfm.models.lightgbm_model import LightGBMModel
from stress_tfm.models.tabdpt_model import TabDPTModel
from stress_tfm.models.tabicl_model import TabICLModel
from stress_tfm.models.tabpfn_model import TabPFNModel

logger = logging.getLogger(__name__)

MODELS: dict[str, type[BaseModel]] = {
    "tabpfn": TabPFNModel,
    "tabicl": TabICLModel,
    "tabdpt": TabDPTModel,
    "causilo": CausiloModel,
    "kumo": KumoModel,
    "lightgbm": LightGBMModel,
}


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """Create a model instance by name.

    Args:
        name: Model identifier (e.g. "tabpfn", "causilo").
        **kwargs: Optional parameters passed to the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    name = name.lower()

    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Model '{name}' not recognized. Available: {available}")

    model_class = MODELS[name]

    sig = inspect.signature(model_class.__init__)
    params = sig.parameters
    accepts_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if not accepts_var_keyword:
        valid_keys = set(params.keys()) - {"self"}
        removed = set(kwargs.keys()) - valid_keys
        if removed:
            logger.warning(f"Model '{name}' ignores: {removed}")
        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    logger.info(f"Creating model: {name} with params: {kwargs}")
    return model_class(**kwargs)