"""Consolidate per-run predictions.csv files into per-task parquet files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PRED_COLS = ["file_id", "y_true", "y_pred"]


def _parse_path(path: Path, paper_dir: Path) -> dict[str, object]:
    """Extract task, model, seed, n_context from a predictions.csv path.

    Args:
        path: Full path to predictions.csv.
        paper_dir: The paper output directory (e.g., results/paper1).

    Returns:
        Dict with task, model, seed, n_context.
    """
    rel = path.relative_to(paper_dir)
    task = rel.parts[0]
    model = rel.parts[1]
    match = re.match(r"seed_(\d+)_ctx_(\d+)", rel.parts[2])
    if match is None:
        msg = f"Unexpected directory name: {rel.parts[2]}"
        raise ValueError(msg)
    return {
        "task": task,
        "model": model,
        "seed": int(match.group(1)),
        "n_context": int(match.group(2)),
    }


def consolidate(paper_dir: Path) -> list[Path]:
    """Read all predictions.csv and write per-task parquet files.

    Args:
        paper_dir: Paper output directory (e.g., results/paper1).

    Returns:
        List of written parquet file paths.
    """
    summary_path = paper_dir / "benchmark_summary.csv"
    summary: pd.DataFrame | None = None
    if summary_path.exists():
        summary = pd.read_csv(summary_path)

    frames: list[pd.DataFrame] = []

    for csv_path in sorted(paper_dir.glob("*/*/*/predictions.csv")):
        meta = _parse_path(csv_path, paper_dir)
        df = pd.read_csv(csv_path)

        # Keep only prediction columns + y_proba_*
        proba_cols = [c for c in df.columns if c.startswith("y_proba_")]
        keep = PRED_COLS + proba_cols
        df = df[keep].copy()

        df["task"] = meta["task"]
        df["model"] = meta["model"]
        df["seed"] = meta["seed"]
        df["n_context"] = meta["n_context"]
        frames.append(df)

    if not frames:
        print(f"No predictions found under {paper_dir}")
        return []

    all_preds = pd.concat(frames, ignore_index=True)

    # Join n_classes_in_ctx from summary
    if summary is not None:
        all_preds = all_preds.merge(
            summary[
                ["task", "model", "seed", "n_context_subjects", "n_classes_in_ctx"]
            ],
            left_on=["task", "model", "seed", "n_context"],
            right_on=["task", "model", "seed", "n_context_subjects"],
            how="left",
        ).drop(columns=["n_context_subjects"])

    # Reorder columns
    proba_cols = [c for c in all_preds.columns if c.startswith("y_proba_")]
    col_order = (
        ["task", "model", "seed", "n_context", "n_classes_in_ctx"]
        + PRED_COLS
        + proba_cols
    )
    all_preds = all_preds[[c for c in col_order if c in all_preds.columns]]

    # Write per-task parquet
    written: list[Path] = []
    for task, group in all_preds.groupby("task"):
        out_path = paper_dir / f"predictions_{task}.parquet"
        group.to_parquet(out_path, index=False)
        written.append(out_path)
        print(f"  {out_path}: {len(group):,} rows")

    return written


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Consolidate predictions into parquet")
    parser.add_argument(
        "--paper", type=int, required=True, help="Paper number (1 or 2)"
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    paper_dir = (
        Path(args.output_dir) if args.output_dir else Path(f"results/paper{args.paper}")
    )

    if not paper_dir.exists():
        raise SystemExit(f"Directory not found: {paper_dir}")

    print(f"Consolidating predictions from {paper_dir}/")
    written = consolidate(paper_dir)
    if written:
        print(f"Done. {len(written)} parquet file(s) written.")


if __name__ == "__main__":
    main()
