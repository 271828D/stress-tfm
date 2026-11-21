"""Flatten run results into a summary DataFrame and persist to CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from stress_tfm.evaluation.evaluate import RunResult

logger = logging.getLogger(__name__)


def write_summary(run_dir: Path, results: list[RunResult]) -> pd.DataFrame:
    """Flatten RunResult list into a DataFrame and save to CSV.

    Args:
        run_dir: Output directory where the CSV is written.
        results: List of RunResult objects from the benchmark.

    Returns:
        Summary DataFrame (empty if no results).
    """
    if not results:
        logger.warning("No successful runs to summarize.")
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "task": r.task,
                "model": r.model,
                "seed": r.seed,
                "n_context_subjects": r.n_context,
                "n_classes_in_ctx": r.n_classes_in_ctx,
                **r.metrics,
            }
            for r in results
        ]
    )

    # Effective-N report per (task, n_ctx)
    for (task, n_ctx), group in df.groupby(["task", "n_context_subjects"]):
        total = len(group)
        full = (group["n_classes_in_ctx"] >= group["n_classes_in_ctx"].max()).sum()
        partial = (group["n_classes_in_ctx"] >= 2).sum() - full
        single = (group["n_classes_in_ctx"] == 1).sum()
        logger.info(
            "Effective N [%s, n_ctx=%d]: %d full-coverage, %d partial, %d single-class (of %d)",
            task,
            n_ctx,
            full,
            partial,
            single,
            total,
        )

    out_path = run_dir / "benchmark_summary.csv"

    # Merge with existing results (resume-friendly)
    key_cols = ["task", "model", "seed", "n_context_subjects"]
    if out_path.exists():
        existing = pd.read_csv(out_path)
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=key_cols, keep="last")
            .sort_values(key_cols)
            .reset_index(drop=True)
        )

    df.to_csv(out_path, index=False)
    logger.info("Summary saved to %s: %d rows", out_path, len(df))
    return df
