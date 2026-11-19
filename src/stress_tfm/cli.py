"""Command-line interface for the StressID-TFM benchmark."""

from __future__ import annotations

import argparse
import logging
import os
import sys


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

    # --- run subcommand ---
    run = sub.add_parser("run", help="Run the benchmark")
    run.add_argument(
        "--data",
        type=str,
        default="data/master_table.parquet",
        help="Path to master table (.parquet)",
    )
    run.add_argument(
        "--paper",
        type=int,
        choices=[1, 2],
        default=1,
        help="Paper number (sets default grid: 1=6 ctx sizes/10 seeds, 2=12 ctx sizes/20 seeds)",
    )
    run.add_argument(
        "--models",
        type=str,
        default="all",
        help='Comma-separated model names, or "all" (default: all)',
    )
    run.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (default: 10 for paper 1, 20 for paper 2)",
    )
    run.add_argument(
        "--n-context",
        type=int,
        nargs="+",
        default=None,
        help="Context subject counts (default: paper-specific grid)",
    )
    run.add_argument(
        "--tasks",
        type=str,
        default="all",
        help='Comma-separated task names ("binary", "3class"), or "all" (default: all)',
    )
    run.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated feature columns (default: auto-detect numeric)",
    )
    run.add_argument(
        "--subject-col",
        type=str,
        default="subject_id",
        help="Subject ID column name (default: subject_id)",
    )
    run.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/paper{N})",
    )
    run.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Compute device (default: cuda)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already-completed runs (default: True)",
    )
    run.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Re-run all conditions (overrides --resume)",
    )

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    if args.cmd == "run":
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # ─── Paper-specific defaults ────────────────────────────
        PAPER_DEFAULTS = {
            1: {
                "n_context": [1, 6, 16, 32, 49, 58],
                "seeds": list(range(10)),
                "output": "results/paper1",
            },
            2: {
                "n_context": [1, 6, 13, 16, 20, 26, 32, 39, 46, 49, 52, 58],
                "seeds": list(range(20)),
                "output": "results/paper2",
            },
        }
        defaults = PAPER_DEFAULTS[args.paper]

        # ─── Resolve overrides ──────────────────────────────────
        n_context = args.n_context if args.n_context is not None else defaults["n_context"]
        seeds = args.seeds if args.seeds is not None else defaults["seeds"]
        output = args.output if args.output is not None else defaults["output"]
        resume = not args.no_resume

        model_names = (
            [m.strip() for m in args.models.split(",")]
            if args.models != "all"
            else None
        )
        tasks = (
            [t.strip() for t in args.tasks.split(",")]
            if args.tasks != "all"
            else None
        )
        feature_cols = (
            [c.strip() for c in args.feature_cols.split(",")]
            if args.feature_cols
            else None
        )

        from stress_tfm.evaluation.runner import run_benchmark

        run_benchmark(
            data_path=args.data,
            model_names=model_names,
            seeds=seeds,
            n_context_list=n_context,
            feature_cols=feature_cols,
            tasks=tasks,
            subject_col=args.subject_col,
            output_dir=output,
            device=args.device,
            resume=resume,
        )


if __name__ == "__main__":
    main()