"""Grid iteration: composes split, evaluate, and save for the full benchmark."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from stress_tfm.config import TASK_CONFIG
from stress_tfm.evaluation.artifacts import save_predictions, save_split_data
from stress_tfm.evaluation.evaluate import RunResult, evaluate_model
from stress_tfm.evaluation.split import SplitData, prepare_split
from stress_tfm.models.base import BaseModel
from stress_tfm.models.zoo import get_model

logger = logging.getLogger(__name__)


def run_grid(
    df: pd.DataFrame,
    model_names: list[str],
    seeds: list[int],
    n_context_list: list[int],
    tasks: list[str],
    feature_cols: list[str],
    subject_col: str,
    run_dir: Path,
    device: str,
    resume: bool,
) -> list[RunResult]:
    """Execute the full experimental grid.

    Iterates over tasks x context sizes x seeds x models.
    For each (task, n_ctx, seed): prepares the split once, then
    evaluates each model and saves artifacts.

    Args:
        df: Full master table.
        model_names: Model names to evaluate.
        seeds: Random seeds.
        n_context_list: Context subject counts.
        tasks: Task names.
        feature_cols: Feature column names.
        subject_col: Subject ID column.
        run_dir: Base output directory.
        device: Compute device.
        resume: Skip runs with existing predictions.

    Returns:
        List of RunResult for all successful evaluations.
    """
    results: list[RunResult] = []
    total = len(tasks) * len(n_context_list) * len(seeds) * len(model_names)

    with tqdm(total=total, desc="Benchmark", unit="run") as pbar:
        for task in tasks:
            target_col: str = TASK_CONFIG[task]["target_col"]
            classes = np.ndarray = np.array(TASK_CONFIG[task]["classes"])
            task_dir: Path = run_dir / task

            for n_ctx in n_context_list:
                for seed in seeds:
                    split_data = _process_condition(
                        df,
                        subject_col,
                        n_ctx,
                        seed,
                        feature_cols,
                        target_col,
                        task_dir,
                    )
                    if split_data is None:
                        pbar.update(len(model_names))
                        continue

                    for name in model_names:
                        if _should_skip(
                            task_dir,
                            name,
                            seed,
                            n_ctx,
                            resume,
                        ):
                            pbar.update(1)
                            continue

                        model: BaseModel = get_model(
                            name,
                            device=device,
                            classes=classes,
                        )
                        if n_ctx < model.min_context:
                            logger.warning(
                                "[%s] Skipped: n_ctx=%d < min_context=%d",
                                name,
                                n_ctx,
                                model.min_context,
                            )
                            pbar.update(1)
                            continue

                        result = evaluate_model(
                            model,
                            task,
                            name,
                            seed,
                            n_ctx,
                            split_data.X_ctx,
                            split_data.y_ctx,
                            split_data.X_test,
                            split_data.y_test,
                        )
                        if result is not None:
                            results.append(result)
                            _persist(
                                task_dir,
                                name,
                                seed,
                                n_ctx,
                                result,
                                split_data,
                                model,
                            )

                        pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _process_condition(
    df: pd.DataFrame,
    subject_col: str,
    n_ctx: int,
    seed: int,
    feature_cols: list[str],
    target_col: str,
    task_dir: Path,
) -> SplitData | None:
    """Prepare the split and save split artifacts for one condition.

    Args:
        df: Full master table.
        subject_col: Subject ID column.
        n_ctx: Context subject count.
        seed: Random seed.
        feature_cols: Feature columns.
        target_col: Label column.
        task_dir: Task output directory.

    Returns:
        SplitData on success, None if context is single-class.
    """
    split_data = prepare_split(
        df,
        subject_col,
        n_ctx,
        seed,
        feature_cols,
        target_col,
    )
    if split_data is None:
        return None

    save_split_data(
        output_dir=task_dir,
        seed=seed,
        n_context=n_ctx,
        ctx_df=split_data.ctx_df,
        test_df=split_data.test_df,
        X_context=split_data.X_ctx,
        X_test=split_data.X_test,
        y_context=split_data.y_ctx,
        y_test=split_data.y_test,
    )
    return split_data


def _should_skip(
    task_dir: Path,
    model_name: str,
    seed: int,
    n_ctx: int,
    resume: bool,
) -> bool:
    """Check if a run should be skipped due to resume.

    Args:
        task_dir: Task output directory.
        model_name: Model name.
        seed: Random seed.
        n_ctx: Context subject count.
        resume: Whether resume mode is active.

    Returns:
        True if the run should be skipped.
    """
    if not resume:
        return False
    pred_path = task_dir / model_name / f"seed_{seed}_ctx_{n_ctx}" / "predictions.csv"
    return pred_path.exists()


def _persist(
    task_dir: Path,
    model_name: str,
    seed: int,
    n_ctx: int,
    result: RunResult,
    split_data: SplitData,
    model: BaseModel,
) -> None:
    """Save predictions and model artifact for one completed run.

    Args:
        task_dir: Task output directory.
        model_name: Model name.
        seed: Random seed.
        n_ctx: Context subject count.
        result: Successful RunResult.
        split_data: SplitData (for file_ids and X_test).
        model: Model wrapper (for optional .save()).
    """
    file_ids = (
        split_data.test_df["file_id"].values
        if "file_id" in split_data.test_df.columns
        else None
    )

    save_predictions(
        output_dir=task_dir,
        model_name=model_name,
        seed=seed,
        n_context=n_ctx,
        y_test=split_data.y_test,
        y_pred=result.y_pred,
        y_proba=result.y_proba,
        file_ids=file_ids,
        X_test=split_data.X_test,
    )

    if hasattr(model, "save"):
        model_path = task_dir / model_name / f"seed_{seed}_ctx_{n_ctx}" / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
