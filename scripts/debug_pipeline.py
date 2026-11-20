"""Debug the data pipeline without running any model."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from stress_tfm.data.loader import load_data
from stress_tfm.evaluation.artifacts import save_predictions, save_split_data
from stress_tfm.evaluation.split import detect_feature_cols, prepare_split

# --- Load ---
df = load_data("data/master_table.parquet")
feature_cols = detect_feature_cols(df, "subject_id")
print(f"Loaded: {df.shape} | Features: {len(feature_cols)}")

# --- Split + preprocess (same code path as pipeline.py) ---
split_data = prepare_split(
    df,
    subject_col="subject_id",
    n_ctx=1,
    seed=0,
    feature_cols=feature_cols,
    target_col="label_binary",
)
assert split_data is not None, "prepare_split should not return None"

print(f"Context: {split_data.X_ctx.shape} | Test: {split_data.X_test.shape}")
print(f"Context subjects: {split_data.ctx_df['subject_id'].unique().tolist()}")
print(f"y_ctx: {split_data.y_ctx[:5]}... | y_test: {split_data.y_test[:5]}...")

# --- Save split data ---
save_split_data(
    output_dir="results/debug/",
    seed=0,
    n_context=1,
    ctx_df=split_data.ctx_df,
    test_df=split_data.test_df,
    X_context=split_data.X_ctx,
    X_test=split_data.X_test,
    y_context=split_data.y_ctx,
    y_test=split_data.y_test,
)

# --- Mock predictions ---
n = len(split_data.y_test)
y_pred = np.random.choice([0, 1], size=n)
y_proba = np.column_stack([1 - np.random.rand(n), np.random.rand(n)])

save_predictions(
    output_dir="results/debug/binary",
    model_name="mock",
    seed=0,
    n_context=1,
    y_test=split_data.y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    file_ids=split_data.test_df["file_id"].values,
    X_test=split_data.X_test,
)

print("\nDone. Check results/debug/")
