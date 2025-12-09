# ============================================================
# Path: tests/test_integration_pipeline_mlflow.py
# Filename: test_integration_pipeline_mlflow.py
# Author: Your Team
# Date: December 2025
# Purpose: End-to-end integration test for MLflow pipeline
# ============================================================

import pytest
import mlflow
import pandas as pd
from src.model_training.train_logreg import train_logreg
from features.game_features import (
    fetch_season_games,
    generate_features_for_games,
)
from src.prediction_engine.predictor import NBAPredictor

@pytest.mark.integration
def test_full_pipeline(tmp_path):
    # -----------------------------
    # Step 1: Fetch real NBA data
    # -----------------------------
    game_ids = fetch_season_games(2023, limit=3)
    features = generate_features_for_games(game_ids)

    # Validate schema consistency
    expected_columns = ["PTS", "REB", "AST", "FG_PCT", "FT_PCT", "PLUS_MINUS", "TOV", "win"]
    for col in expected_columns:
        assert col in features.columns

    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    # -----------------------------
    # Step 2: Train and log model
    # -----------------------------
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    result = train_logreg(str(features_path), out_dir=str(model_dir))

    # Ensure result dict contains metrics
    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    assert "f1_score" in result["metrics"]
    assert (model_dir / "nba_logreg.pkl").exists()

    # -----------------------------
    # Step 3: Load model via NBAPredictor
    # -----------------------------
    predictor = NBAPredictor(model_path=str(model_dir / "nba_logreg.pkl"))

    # -----------------------------
    # Step 4: Make predictions
    # -----------------------------
    sample = features.drop(columns=["win"]).iloc[:2]
    preds = predictor.predict(sample)
    probas = predictor.predict_proba(sample)

    # Validate predictions
    assert isinstance(preds, pd.Series)
    assert all(label in [0, 1] for label in preds)

    # Validate probabilities are flat list of floats
    assert isinstance(probas, list)
    assert all(isinstance(p, float) for p in probas)
    assert all(0 <= p <= 1 for p in probas)

    # -----------------------------
    # Step 5: Verify MLflow logging
    # -----------------------------
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])
    assert len(runs) > 0
