# ============================================================
# Path: tests/test_train_logreg.py
# Filename: test_train_logreg.py
# Author: Your Team
# Date: December 9, 2025
# Purpose: Unit tests for train_logreg function
# ============================================================

import os
import pandas as pd
import joblib
import pytest

from src.model_training.train_logreg import train_logreg

@pytest.fixture
def sample_features(tmp_path):
    """Create a small sample dataset for testing."""
    df = pd.DataFrame({
        "PTS": [100, 110, 95, 120],
        "REB": [40, 42, 38, 45],
        "AST": [20, 18, 22, 25],
        "FG_PCT": [0.45, 0.50, 0.42, 0.55],
        "FT_PCT": [0.70, 0.75, 0.68, 0.80],
        "PLUS_MINUS": [5, -3, 10, -8],
        "TOV": [12, 10, 14, 9],
        "win": [1, 0, 1, 0]
    })
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path

def test_train_logreg_creates_model_and_metrics(sample_features, tmp_path):
    """Ensure train_logreg trains, saves model, and returns metrics."""
    out_dir = tmp_path / "models"
    result = train_logreg(str(sample_features), out_dir=str(out_dir))

    # Check result keys
    assert "metrics" in result
    assert "model_path" in result

    # Check metrics structure
    metrics = result["metrics"]
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["f1_score"], float)

    # Check model file exists
    model_path = result["model_path"]
    assert os.path.exists(model_path)

    # Check model can be loaded
    model = joblib.load(model_path)
    assert hasattr(model, "predict")
