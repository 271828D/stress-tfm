"""Smoke test: 1 run per model to verify the full pipeline and measure wall-clock time."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from stress_tfm.data.loader import load_data
from stress_tfm.data.preprocess import preprocess_context_test
from stress_tfm.data.splitting import split_context_test
from stress_tfm.evaluation.artifacts import save_predictions
from stress_tfm.evaluation.metrics import compute_metrics
from stress_tfm.models.zoo import MODELS, get_model
from stress_tfm.utils.seed import set_seed

# ─── Config ─────────────────────────────────────────────────────
DATA_PATH = "data/master_table.parquet"
N_CONTEXT = 6
SEED = 0
TASK = "binary"
TARGET_COL = "label_binary"
CLASSES = np.array([0, 1])
SUBJECT_COL = "subject_id"
DEVICE = "cuda"
OUTPUT_DIR = Path("results/smoke_test")

# ─── Load ───────────────────────────────────────────────────────
df = load_data(DATA_PATH)
feature_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in (TARGET_COL, "label_3class", SUBJECT_COL, "file_id", "task_id")
]
print(f"Loaded: {df.shape} | Features: {len(feature_cols)}")

# ─── Split (once, shared across models) ─────────────────────────
ctx_df, test_df, ctx_subjects, test_subjects = split_context_test(
    df, subject_col=SUBJECT_COL, n_context=N_CONTEXT, random_state=SEED
)
X_ctx, X_test, y_ctx, y_test = preprocess_context_test(
    ctx_df, test_df, feature_cols, TARGET_COL
)
print(f"Context: {X_ctx.shape} | Test: {X_test.shape}")
print(f"Context subjects: {ctx_subjects}")
print()

# ─── Run each model ─────────────────────────────────────────────
results = []

for name in MODELS:
    t0 = time.time()
    try:
        model = get_model(name, device=DEVICE, classes=CLASSES)
        set_seed(SEED)
        model.fit(X_ctx, y_ctx)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        elapsed = time.time() - t0

        # Save
        task_dir = OUTPUT_DIR / TASK / name / f"seed_{SEED}_ctx_{N_CONTEXT}"
        task_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(
            output_dir=OUTPUT_DIR / TASK,
            model_name=name,
            seed=SEED,
            n_context=N_CONTEXT,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            file_ids=test_df["file_id"].values,
            X_test=X_test,
        )

        results.append({
            "model": name,
            "status": "OK",
            "time_s": round(elapsed, 2),
            "f1": round(metrics["f1"], 4),
            "auc": round(metrics["roc_auc"], 4),
        })
        print(f"  ✓ {name:12s} | {elapsed:6.2f}s | F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")

    except Exception as e:
        elapsed = time.time() - t0
        results.append({
            "model": name,
            "status": f"FAIL: {e}",
            "time_s": round(elapsed, 2),
            "f1": None,
            "auc": None,
        })
        print(f"  ✗ {name:12s} | {elapsed:6.2f}s | ERROR: {e}")

# ─── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
summary = pd.DataFrame(results)
print(summary.to_string(index=False))
print("=" * 60)

total_time = summary["time_s"].sum()
n_ok = (summary["status"] == "OK").sum()
print(f"\n{ n_ok}/6 models passed | Total: {total_time:.1f}s")

# Estimate full Paper 1 runtime
per_run_avg = summary[summary["status"] == "OK"]["time_s"].mean()
n_runs = 720  # 6 models × 6 ctx × 10 seeds × 2 tasks
est_total_hours = (per_run_avg * n_runs) / 3600
print(f"Avg per run: {per_run_avg:.1f}s")
print(f"Estimated Paper 1 (720 runs): {est_total_hours:.1f} hours")