"""Data splitting and preprocessing for one (seed, n_ctx) condition."""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from stress_tfm.data.preprocess import preprocess_context_test
from stress_tfm.data.splitting import split_context_test

logger = logging.getLogger(__name__)


class SplitData(NamedTuple):
    """Prepared context/test pair for one (seed, n_ctx) condition.

    Attributes:
        ctx_df: Raw context DataFrame (with all columns).
        test_df: Raw test DataFrame (with all columns).
        X_ctx: Context feature matrix.
        X_test: Test feature matrix.
        y_ctx: Context labels.
        y_test: Test labels.
    """

    ctx_df: pd.DataFrame
    test_df: pd.DataFrame
    X_ctx: pd.DataFrame
    X_test: pd.DataFrame
    y_ctx: np.ndarray
    y_test: np.ndarray


def prepare_split(
    df: pd.DataFrame,
    subject_col: str,
    n_ctx: int,
    seed: int,
    feature_cols: list[str],
    target_col: str,
) -> SplitData | None:
    """Split, preprocess, and validate one (seed, n_ctx) condition.

    Args:
        df: Full master table.
        subject_col: Subject ID column name.
        n_ctx: Number of context subjects.
        seed: Random seed for the split.
        feature_cols: Feature column names.
        target_col: Label column name.

    Returns:
        SplitData on success, None only if the split itself fails.
        Single-class contexts are valid (degenerate ICL conditions).
    """
    ctx_df, test_df, _, _ = split_context_test(
        df,
        subject_col=subject_col,
        n_context=n_ctx,
        random_state=seed,
    )
    X_ctx, X_test, y_ctx, y_test = preprocess_context_test(
        ctx_df,
        test_df,
        feature_cols,
        target_col,
    )

    return SplitData(
        ctx_df=ctx_df,
        test_df=test_df,
        X_ctx=X_ctx,
        X_test=X_test,
        y_ctx=y_ctx,
        y_test=y_test,
    )


def detect_feature_cols(df: pd.DataFrame, subject_col: str) -> list[str]:
    """Auto-detect numeric feature columns, excluding IDs and labels.

    Args:
        df: Master table DataFrame.
        subject_col: Subject ID column name.

    Returns:
        List of feature column names.
    """
    exclude = {subject_col, "label_binary", "label_3class", "file_id", "task_id"}
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]
