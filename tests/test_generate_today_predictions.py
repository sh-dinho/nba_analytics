# ============================================================
# File: tests/test_generate_today_predictions.py
# Purpose: Unit test for generate_today_predictions
# ============================================================

import os
import pandas as pd
import pytest
import joblib

from scripts.generate_today_predictions import generate_today_predictions
from core.config import PREDICTIONS_FILE, MODEL_FILE_PKL


def test_generate_today_predictions(tmp_path):
    # --- Step 1: Create synthetic features file ---
    features_file = tmp_path / "features.csv"
    df = pd.DataFrame([
        {
            "game_id": "GSW_vs_MIA",
            "home_team": "GSW",
            "away_team": "MIA",
            "home_avg_pts": 110, "home_avg_ast": 25, "home_avg_reb": 45, "home_avg_games_played": 20,
            "away_avg_pts": 105, "away_avg_ast": 23, "away_avg_reb": 42, "away_avg_games_played": 20,
            "decimal_odds": 1.9
        }
    ])
    df.to_csv(features_file, index=False)

    # --- Step 2: Create a dummy model artifact ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[[
        "home_avg_pts", "home_avg_ast", "home_avg_reb", "home_avg_games_played",
        "away_avg_pts", "away_avg_ast", "away_avg_reb", "away_avg_games_played"
    ]]
    y = [1]  # dummy label

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)

    artifact = {"model": pipeline, "features": list(X.columns)}
    joblib.dump(artifact, MODEL_FILE_PKL)

    # --- Step 3: Run prediction ---
    preds = generate_today_predictions(str(features_file), threshold=0.5)

    # --- Step 4: Assertions ---
    assert "pred_home_win_prob" in preds.columns
    assert "predicted_home_win" in preds.columns
    assert os.path.exists(PREDICTIONS_FILE)

    # Check values are floats between 0 and 1
    assert preds["pred_home_win_prob"].between(0, 1).all()
    # Check predicted labels are 0 or 1
    assert set(preds["predicted_home_win"].unique()).issubset({0, 1})