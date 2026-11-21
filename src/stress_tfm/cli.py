"""Command-line interface for the StressID-TFM benchmark."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict

from stress_tfm.config import (
    PAPER1_CONTEXT_SIZES,
    PAPER1_OUTPUT,
    PAPER1_SEEDS,
    PAPER2_CONTEXT_SIZES,
    PAPER2_OUTPUT,
    PAPER2_SEEDS,
    TASKS,
)
from stress_tfm.models.zoo import MODELS

logger = logging.getLogger(__name__)


class PaperDefaults(TypedDict):
    """Default grid parameters for a paper."""

    n_context: list[int]
    seeds: list[int]
    output: str


PAPER_DEFAULTS: dict[int, PaperDefaults] = {
    1: {
        "n_context": PAPER1_CONTEXT_SIZES,
        "seeds": PAPER1_SEEDS,
        "output": str(PAPER1_OUTPUT),
    },
    2: {
        "n_context": PAPER2_CONTEXT_SIZES,
        "seeds": PAPER2_SEEDS,
        "output": str(PAPER2_OUTPUT),
    },
}


def main() -> None:
    """Entry point for the stress-tfm CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="stress-tfm",
        description="Data-efficiency benchmark for tabular foundation models on StressID",
    )
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run the benchmark")
    run.add_argument("--data", default="data/master_table.parquet")
    run.add_argument("--paper", type=int, choices=[1, 2], default=1)
    run.add_argument("--models", default="all")
    run.add_argument("--seeds", type=int, nargs="+", default=None)
    run.add_argument("--n-context", type=int, nargs="+", default=None)
    run.add_argument("--tasks", default="all")
    run.add_argument("--feature-cols", default=None)
    run.add_argument(
        "--feature-set",
        default=None,
        help="Feature set name from FEATURE_SETS (e.g., 'ecg', 'all'). "
        "Overrides --feature-cols if both are given.",
    )
    run.add_argument("--subject-col", default="subject_id")
    run.add_argument("--output", default=None)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    if args.cmd == "run":
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        defaults = PAPER_DEFAULTS[args.paper]
        n_context: list[int] = (
            args.n_context if args.n_context is not None else defaults["n_context"]
        )
        seeds: list[int] = args.seeds if args.seeds is not None else defaults["seeds"]
        output: str = args.output if args.output is not None else defaults["output"]
        resume: bool = not args.no_resume

        model_names: list[str] = (
            [m.strip() for m in args.models.split(",")]
            if args.models != "all"
            else list(MODELS.keys())
        )
        tasks: list[str] = (
            [t.strip() for t in args.tasks.split(",")]
            if args.tasks != "all"
            else list(TASKS)
        )
        feature_cols: list[str] | None = (
            [c.strip() for c in args.feature_cols.split(",")]
            if args.feature_cols
            else None
        )
        if args.feature_set:
            from stress_tfm.config import FEATURE_SETS

            if args.feature_set not in FEATURE_SETS:
                valid = ", ".join(FEATURE_SETS.keys())
                parser.error(
                    f"Unknown feature set '{args.feature_set}'. Valid: {valid}"
                )
            feature_cols = FEATURE_SETS[args.feature_set]
            # Nest output under feature_set subdirectory
            output = str(Path(output) / args.feature_set)

        from stress_tfm.evaluation.runner import run_benchmark

        run_benchmark(
            data_path=args.data,
            model_names=model_names,
            seeds=seeds,
            n_context_list=n_context,
            tasks=tasks,
            feature_cols=feature_cols,
            subject_col=args.subject_col,
            output_dir=output,
            device=args.device,
            resume=resume,
        )


if __name__ == "__main__":
    main()
