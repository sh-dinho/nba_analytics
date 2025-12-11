# ============================================================
# File: tests/test_predictor.py
# Purpose: Unit tests for src/prediction_engine/predictor.py
# Project: nba_analysis
# ============================================================

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.prediction_engine.predictor import NBAPredictor, main as predictor_cli


@pytest.fixture
def trained_model(tmp_path):
    """
    Train a tiny LogisticRegression on synthetic data and save it.
    Ensures feature_names_in_ is available after fit.
    """
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(50),
            "feat_b": np.random.randn(50),
            "feat_c": np.random.randn(50),
        }
    )
    y = (X["feat_a"] * 0.5 + X["feat_b"] * -0.3 + np.random.randn(50) * 0.1 > 0).astype(
        int
    )

    model = LogisticRegression()
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    return model_path, X.columns.tolist()


@pytest.fixture
def features_df():
    """
    Build a features DataFrame with extra non-numeric and identifier columns.
    Ensures the predictor's numeric coercion and dropping logic works.
    """
    df = pd.DataFrame(
        {
            "feat_a": [0.1, 0.2, -0.3],
            "feat_b": [1.0, -0.5, 0.0],
            "feat_c": [2.0, 2.5, -1.5],
            "TEAM_NAME": ["LAL", "BOS", "NYK"],
            "OPPONENT_TEAM_NAME": ["BOS", "LAL", "PHI"],
            "GAME_ID": ["g1", "g2", "g3"],
            "unique_id": ["u1", "u2", "u3"],
            "prediction_date": ["2025-12-10"] * 3,
        }
    )
    return df


def test_predict_proba_numeric_coercion_and_alignment(trained_model, features_df):
    model_path, expected_cols = trained_model
    predictor = NBAPredictor(model_path=str(model_path), log_dir=str(model_path.parent))

    # Ensure predictor aligns to the trained model's expected features
    proba = predictor.predict_proba(features_df)
    assert isinstance(proba, pd.Series)
    assert proba.name == "win_proba"
    assert len(proba) == len(features_df)

    # Verify no non-numeric columns are passed to the model (implicitly via no error)
    # And mean is a finite number
    assert np.isfinite(proba.mean())


def test_predict_label_threshold_behavior(trained_model, features_df):
    model_path, _ = trained_model
    predictor = NBAPredictor(model_path=str(model_path), log_dir=str(model_path.parent))

    labels_low = predictor.predict_label(features_df, threshold=0.1)
    labels_high = predictor.predict_label(features_df, threshold=0.9)

    assert isinstance(labels_low, pd.Series)
    assert labels_low.name == "win_pred"
    assert labels_low.dtype in (np.int64, np.int32, "int64", "int32")
    # Lower threshold should yield more 1s than high threshold
    assert labels_low.sum() >= labels_high.sum()


def test_missing_expected_features_are_filled(trained_model):
    model_path, expected_cols = trained_model
    predictor = NBAPredictor(model_path=str(model_path), log_dir=str(model_path.parent))

    # Provide only a subset; predictor should reindex and fill missing with 0
    partial = pd.DataFrame({"feat_a": [0.1, -0.2, 0.3]})
    proba = predictor.predict_proba(partial)
    assert len(proba) == 3
    # Sanity: predictions succeed even with missing features
    assert np.isfinite(proba).all()


def test_cli_smoke_proba(tmp_path, trained_model, features_df, monkeypatch):
    # Save features to CSV
    feats_path = tmp_path / "feats.csv"
    features_df.to_csv(feats_path, index=False)

    model_path, _ = trained_model
    out_path = tmp_path / "preds.csv"

    # Mock MLflow to avoid external interactions
    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import src.prediction_engine.predictor as predictor_mod

    monkeypatch.setattr(predictor_mod.mlflow, "start_run", lambda **kwargs: DummyRun())
    monkeypatch.setattr(predictor_mod.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        predictor_mod.mlflow, "log_metric", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        predictor_mod.mlflow, "log_artifact", lambda *args, **kwargs: None
    )

    # Prepare argv
    argv = [
        "predictor",
        "--model",
        str(model_path),
        "--features",
        str(feats_path),
        "--mode",
        "proba",
        "--output",
        str(out_path),
        "--mlflow",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Run CLI
    predictor_cli()

    # Output file should exist and contain predictions
    assert out_path.exists()
    df_out = pd.read_csv(out_path)
    # csv contains one column of predictions; name may be 'win_proba'
    assert len(df_out) == len(features_df)
    assert df_out.shape[1] >= 1
