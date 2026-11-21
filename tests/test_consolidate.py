"""Tests for the consolidate_predictions script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "consolidate_predictions",
    Path(__file__).parent.parent / "scripts" / "consolidate_predictions.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_parse_path = _mod._parse_path


class TestParsePath:
    """Tests for _parse_path regex extraction."""

    def test_binary_tabpfn(self) -> None:
        paper_dir = Path("results/paper1")
        path = paper_dir / "binary" / "tabpfn" / "seed_42_ctx_6" / "predictions.csv"
        result = _parse_path(path, paper_dir)
        assert result["task"] == "binary"
        assert result["model"] == "tabpfn"
        assert result["seed"] == 42
        assert result["n_context"] == 6

    def test_3class_lightgbm(self) -> None:
        paper_dir = Path("results/paper1")
        path = paper_dir / "3class" / "lightgbm" / "seed_0_ctx_58" / "predictions.csv"
        result = _parse_path(path, paper_dir)
        assert result["task"] == "3class"
        assert result["model"] == "lightgbm"
        assert result["seed"] == 0
        assert result["n_context"] == 58

    def test_invalid_dir_name_raises(self) -> None:
        paper_dir = Path("results/paper1")
        path = paper_dir / "binary" / "tabpfn" / "bad_name" / "predictions.csv"
        with pytest.raises(ValueError, match="Unexpected directory name"):
            _parse_path(path, paper_dir)


class TestColumnSelection:
    """Tests that only prediction columns are kept."""

    def test_feature_columns_dropped(self) -> None:
        df = pd.DataFrame(
            {
                "file_id": ["a", "b"],
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "y_proba_c0": [0.7, 0.3],
                "y_proba_c1": [0.3, 0.7],
                "meanhr": [60.0, 70.0],
                "sdhr": [10.0, 12.0],
            }
        )
        proba_cols = [c for c in df.columns if c.startswith("y_proba_")]
        keep = ["file_id", "y_true", "y_pred"] + proba_cols
        result = df[keep]
        assert list(result.columns) == [
            "file_id",
            "y_true",
            "y_pred",
            "y_proba_c0",
            "y_proba_c1",
        ]
        assert "meanhr" not in result.columns

    def test_3class_proba_cols(self) -> None:
        df = pd.DataFrame(
            {
                "file_id": ["a"],
                "y_true": [0],
                "y_pred": [0],
                "y_proba_c0": [0.5],
                "y_proba_c1": [0.3],
                "y_proba_c2": [0.2],
            }
        )
        proba_cols = [c for c in df.columns if c.startswith("y_proba_")]
        assert proba_cols == ["y_proba_c0", "y_proba_c1", "y_proba_c2"]
