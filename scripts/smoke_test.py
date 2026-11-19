"""Smoke test: 1 run per model to verify the full pipeline and measure wall-clock time."""

import argparse
import time
from pathlib import Path

import numpy as np

from stress_tfm.data.loader import load_data
from stress_tfm.data.preprocess import preprocess_context_test
from stress_tfm.data.splitting import split_context_test
from stress_tfm.evaluation.artifacts import save_predictions
from stress_tfm.evaluation.metrics import compute_metrics
from stress_tfm.models.zoo import MODELS, get_model
from stress_tfm.utils.seed import set_seed


def main() -> None:
    """Run the smoke test with CLI arguments."""
    parser = argparse.ArgumentParser(description="Smoke test: 1 run per selected model")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help='Comma-separated model names, or "all" (default: all)',
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "3class"],
        help="Task to test (default: binary)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--n-context",
        type=int,
        default=2,
        help="Number of context subjects (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/master_table.parquet",
        help="Path to master table",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/smoke_test",
        help="Output directory",
    )
    args = parser.parse_args()

    # -- Resolve models
    model_names = (
        [m.strip() for m in args.models.split(",")]
        if args.models != "all"
        else list(MODELS.keys())
    )

    # -- Task config
    TASK_CONFIG = {
        "binary": {"target_col": "label_binary", "classes": np.array([0, 1])},
        "3class": {"target_col": "label_3class", "classes": np.array([0, 1, 2])},
    }
    task_cfg = TASK_CONFIG[args.task]
    target_col = task_cfg["target_col"]
    classes = task_cfg["classes"]

    # -- Load
    df = load_data(args.data)
    feature_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c
        not in (
            target_col,
            "label_binary",
            "label_3class",
            "subject_id",
            "file_id",
            "task_id",
        )
    ]
    print(f"Loaded: {df.shape} | Features: {len(feature_cols)}")
    print(
        f"Task: {args.task} | Device: {args.device} | k={args.n_context} | seed={args.seed}"
    )
    print(f"Models: {model_names}")
    print()

    # -- Split (once, shared across models)
    ctx_df, test_df, ctx_subjects, test_subjects = split_context_test(
        df, subject_col="subject_id", n_context=args.n_context, random_state=args.seed
    )
    X_ctx, X_test, y_ctx, y_test = preprocess_context_test(
        ctx_df, test_df, feature_cols, target_col
    )
    print(f"Context: {X_ctx.shape} | Test: {X_test.shape}")
    print(f"Context subjects: {ctx_subjects}")
    print()

    # -- Run each model
    results = []

    for name in model_names:
        t0 = time.time()
        try:
            model = get_model(name, device=args.device, classes=classes)
            set_seed(args.seed)
            model.fit(X_ctx, y_ctx)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            metrics = compute_metrics(y_test, y_pred, y_proba)
            elapsed = time.time() - t0

            save_predictions(
                output_dir=Path(args.output) / args.task,
                model_name=name,
                seed=args.seed,
                n_context=args.n_context,
                y_test=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                file_ids=test_df["file_id"].values,
                X_test=X_test,
            )

            results.append(
                {
                    "model": name,
                    "status": "OK",
                    "time_s": round(elapsed, 2),
                    "f1": round(metrics["f1"], 4),
                    "auc": round(metrics["roc_auc"], 4),
                }
            )
            print(
                f"  OK   {name:12s} | {elapsed:6.2f}s | F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
            )

        except Exception as e:
            elapsed = time.time() - t0
            results.append(
                {
                    "model": name,
                    "status": f"FAIL: {e}",
                    "time_s": round(elapsed, 2),
                    "f1": None,
                    "auc": None,
                }
            )
            print(f"  FAIL {name:12s} | {elapsed:6.2f}s | ERROR: {e}")

    # -- Summary
    print("\n" + "=" * 60)
    for r in results:
        print(f"  {r['model']:12s} | {r['status']:40s} | {r['time_s']}s")
    print("=" * 60)

    ok = [r for r in results if r["status"] == "OK"]
    if ok:
        avg = sum(r["time_s"] for r in ok) / len(ok)
        print(f"\n{len(ok)}/{len(model_names)} passed | Avg: {avg:.1f}s/run")


if __name__ == "__main__":
    main()
