"""Experiment runner: thin orchestrator for the data-efficiency benchmark."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from stress_tfm.config import TASK_CONFIG
from stress_tfm.data.loader import load_data
from stress_tfm.evaluation.pipeline import run_grid
from stress_tfm.evaluation.split import detect_feature_cols
from stress_tfm.evaluation.summary import write_summary

logger = logging.getLogger(__name__)


def run_benchmark(
    data_path: str,
    model_names: list[str],
    seeds: list[int],
    n_context_list: list[int],
    tasks: list[str],
    feature_cols: list[str] | None = None,
    subject_col: str = "subject_id",
    output_dir: str = "results/paper1",
    device: str = "cuda",
    resume: bool = True,
) -> pd.DataFrame:
    """Run the data-efficiency benchmark.

    Validates inputs, loads data, delegates to the grid pipeline,
    and writes the summary.

    Args:
        data_path: Path to master_table.parquet.
        model_names: Model names to evaluate.
        seeds: Random seeds for subject-level splits.
        n_context_list: Context subject counts (absolute).
        tasks: Task names. Must be keys of TASK_CONFIG.
        feature_cols: Feature columns (None = auto-detect numeric).
        subject_col: Subject ID column name.
        output_dir: Base output directory.
        device: Compute device ("cpu" or "cuda").
        resume: Skip runs whose predictions.csv already exists.

    Returns:
        Summary DataFrame with one row per successful evaluation.

    Raises:
        ValueError: If a task name is not in TASK_CONFIG.
    """
    _validate_tasks(tasks)

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    if feature_cols is None:
        feature_cols = detect_feature_cols(df, subject_col)

    _log_grid(model_names, tasks, n_context_list, seeds, feature_cols)

    results = run_grid(
        df=df,
        model_names=model_names,
        seeds=seeds,
        n_context_list=n_context_list,
        tasks=tasks,
        feature_cols=feature_cols,
        subject_col=subject_col,
        run_dir=run_dir,
        device=device,
        resume=resume,
    )

    return write_summary(run_dir, results)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_tasks(tasks: list[str]) -> None:
    """Raise if any task name is not a valid key in TASK_CONFIG.

    Args:
        tasks: Task names to validate.

    Raises:
        ValueError: If an unknown task name is found.
    """
    for t in tasks:
        if t not in TASK_CONFIG:
            msg = f"Unknown task '{t}'. Valid: {list(TASK_CONFIG.keys())}"
            raise ValueError(msg)


def _log_grid(
    models: list[str],
    tasks: list[str],
    contexts: list[int],
    seeds: list[int],
    features: list[str],
) -> None:
    """Log the experimental grid at INFO level.

    Args:
        models: Model names.
        tasks: Task names.
        contexts: Context subject counts.
        seeds: Seed values.
        features: Feature column names.
    """
    logger.info("Features (%d): %s...", len(features), features[:5])
    logger.info("Models: %s | Tasks: %s", models, tasks)
    logger.info("Contexts: %s | Seeds: %s", contexts, seeds)
