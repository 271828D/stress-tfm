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
                **r.metrics,
            }
            for r in results
        ]
    )

    out_path = run_dir / "benchmark_summary.csv"
    df.to_csv(out_path, index=False)
    logger.info("Summary saved to %s: %d rows", out_path, len(df))
    return df
