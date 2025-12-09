# ============================================================
# Path: tests/test_training.py
# Purpose: Unit tests for src/model_training/training.py
# Project: nba_analysis
# ============================================================

import pytest
import pandas as pd
from src.model_training.training import train_logreg

def test_train_logreg_valid(monkeypatch, tmp_path):
    # Create synthetic dataset
    df = pd.DataFrame({
        "feat1": [0, 1, 0, 1, 0, 1],
        "feat2": [1, 0, 1, 0, 1, 0],
        "win":   [0, 1, 0, 1, 0, 1],
    })
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path)

    # Mock mlflow functions
    monkeypatch.setattr("mlflow.start_run", lambda *a, **kw: __import__("contextlib").nullcontext())
    monkeypatch.setattr("mlflow.log_param", lambda *a, **kw: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *a, **kw: None)
    monkeypatch.setattr("mlflow.sklearn.log_model", lambda *a, **kw: None)

    model, metrics = train_logreg(str(features_path), out_dir=str(tmp_path))
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics


def test_train_logreg_missing_target(tmp_path):
    df = pd.DataFrame({"feat1": [0, 1], "feat2": [1, 0]})
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path)

    with pytest.raises(ValueError):
        train_logreg(str(features_path), out_dir=str(tmp_path))


def test_train_logreg_empty_data(tmp_path):
    df = pd.DataFrame(columns=["feat1", "feat2", "win"])
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path)

    with pytest.raises(ValueError):
        train_logreg(str(features_path), out_dir=str(tmp_path))
