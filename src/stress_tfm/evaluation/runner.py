"""Experiment runner: iterates over tasks × models × context sizes × seeds."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from stress_tfm.data.loader import load_data
from stress_tfm.data.preprocess import preprocess_context_test
from stress_tfm.data.splitting import split_context_test
from stress_tfm.evaluation.artifacts import save_predictions, save_split_data
from stress_tfm.evaluation.metrics import compute_metrics
from stress_tfm.models.zoo import MODELS, get_model
from stress_tfm.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_benchmark(
    data_path: str,
    model_names: list[str] | None = None,
    seeds: list[int] | None = None,
    n_context_list: list[int] | None = None,
    feature_cols: list[str] | None = None,
    tasks: list[str] | None = None,
    subject_col: str = "subject_id",
    output_dir: str = "results/paper1",
    device: str = "cuda",
    resume: bool = True,
) -> pd.DataFrame:
    """Run the full data-efficiency benchmark.

    For each task, model, context size, and seed:
    splits the data, preprocesses, evaluates the model, and saves artifacts.

    Args:
        data_path: Path to master_table.parquet.
        model_names: Subset of models to run (None = all).
        seeds: List of random seeds.
        n_context_list: Number of context subjects per run (absolute counts).
        feature_cols: Feature column names (None = auto-detect numeric).
        tasks: List of task names. Default: ["binary", "3class"].
        subject_col: Subject ID column name.
        output_dir: Base output directory.
        device: Compute device ("cpu" or "cuda").
        resume: If True, skip runs whose predictions.csv already exists.

    Returns:
        DataFrame with one row per (task, model, seed, context_size).
    """
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # ─── Device ─────────────────────────────────────────────────
    if device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            device = "cpu"
        else:
            logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"Device: {device}")

    # ─── Defaults ───────────────────────────────────────────────
    if model_names is None:
        model_names = list(MODELS.keys())
    if seeds is None:
        seeds = list(range(10))
    if n_context_list is None:
        n_context_list = [1, 6, 16, 32, 49, 58]
    if tasks is None:
        tasks = ["binary", "3class"]

    # ─── Load data ──────────────────────────────────────────────
    df = load_data(data_path)

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in (subject_col, "label_binary", "label_3class")
        ]

    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")
    logger.info(f"Models: {model_names}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Context sizes: {n_context_list}")
    logger.info(f"Seeds: {seeds}")

    # ─── Task config ────────────────────────────────────────────
    TASK_CONFIG = {
        "binary": {"target_col": "label_binary", "classes": np.array([0, 1])},
        "3class": {"target_col": "label_3class", "classes": np.array([0, 1, 2])},
    }

    rows: list[dict[str, Any]] = []
    total_trials = len(tasks) * len(model_names) * len(n_context_list) * len(seeds)
    pbar = tqdm(total=total_trials, desc="Benchmark", unit="trial")

    for task in tasks:
        task_cfg = TASK_CONFIG[task]
        target_col = task_cfg["target_col"]
        classes = task_cfg["classes"]
        task_dir = run_dir / task

        for name in model_names:
            # Build model once per (task, model) — classes are task-level
            model = get_model(name, device=device, classes=classes)
            min_ctx = getattr(model, "min_context", 1)

            for n_ctx in n_context_list:
                # Skip if model can't handle this context size
                if n_ctx < min_ctx:
                    logger.warning(
                        f"[{name}] Skipped: n_ctx={n_ctx} < min_context={min_ctx}"
                    )
                    pbar.update(1)
                    continue

                for seed in seeds:
                    # ─── Resume check ───────────────────────────
                    pred_path = (
                        task_dir / name / f"seed_{seed}_ctx_{n_ctx}" / "predictions.csv"
                    )
                    if resume and pred_path.exists():
                        pbar.update(1)
                        continue

                    # ─── Split ──────────────────────────────────
                    ctx_df, test_df, ctx_subjects, test_subjects = split_context_test(
                        df,
                        subject_col=subject_col,
                        n_context=n_ctx,
                        random_state=seed,
                    )

                    # ─── Preprocess ─────────────────────────────
                    X_ctx, X_test, y_ctx, y_test = preprocess_context_test(
                        ctx_df, test_df, feature_cols, target_col
                    )

                    # ─── NEW: Skip if context has < 2 unique classes ──
                    if len(np.unique(y_ctx)) < 2:
                        logger.warning(
                            f"[{task}] seed={seed} n_ctx={n_ctx}: "
                            f"context has only {len(np.unique(y_ctx))} class(es). Skipping all models."
                        )
                        pbar.update(len(model_names))
                        continue

                    # ─── Save split data (once per seed+ctx, not per model) ──
                    if name == model_names[0]:
                        save_split_data(
                            output_dir=task_dir,
                            seed=seed,
                            n_context=n_ctx,
                            ctx_df=ctx_df,
                            test_df=test_df,
                            X_context=X_ctx,
                            X_test=X_test,
                            y_context=y_ctx,
                            y_test=y_test,
                        )

                    # ─── Evaluate ───────────────────────────────
                    logger.info(
                        f"[{task}] {name} | seed={seed} | n_ctx={n_ctx} | "
                        f"X_ctx={X_ctx.shape} X_test={X_test.shape}"
                    )
                    set_seed(seed)

                    try:
                        model.fit(X_ctx, y_ctx)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)
                        metrics = compute_metrics(y_test, y_pred, y_proba)
                    except Exception:
                        logger.exception(
                            f"[{task}] {name} failed | seed={seed} | n_ctx={n_ctx}"
                        )
                        pbar.update(1)
                        continue

                    rows.append(
                        {
                            "task": task,
                            "model": name,
                            "seed": seed,
                            "n_context_subjects": n_ctx,
                            **metrics,
                        }
                    )

                    # ─── Save predictions ───────────────────────
                    try:
                        file_ids = (
                            test_df["file_id"].values
                            if "file_id" in test_df.columns
                            else None
                        )
                        save_predictions(
                            output_dir=task_dir,
                            model_name=name,
                            seed=seed,
                            n_context=n_ctx,
                            y_test=y_test,
                            y_pred=y_pred,
                            y_proba=y_proba,
                            file_ids=file_ids,
                            X_test=X_test,
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to save predictions for [{task}] {name}"
                        )

                    # ─── Save model artifact (LightGBM only) ────
                    if hasattr(model, "save"):
                        model_path = (
                            task_dir / name / f"seed_{seed}_ctx_{n_ctx}" / "model.pkl"
                        )
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        model.save(model_path)

                    pbar.update(1)

    pbar.close()

    # ─── Summary ────────────────────────────────────────────────
    results = pd.DataFrame(rows)
    out_path = run_dir / "benchmark_summary.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Summary saved to {out_path}: {len(results)} rows")

    return results
