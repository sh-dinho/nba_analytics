"""
Prediction Pipeline
-------------------
Loads the latest trained model, builds features for a given date's games,
generates predictions, and saves them.
"""

import pandas as pd
import joblib
from datetime import date
from loguru import logger
from pathlib import Path

# Import exactly what is defined in your paths.py
from src.config.paths import (
    MODEL_REGISTRY_DIR,
    SCHEDULE_SNAPSHOT,
    LONG_SNAPSHOT,  # This appears to be your features/historical data
    PREDICTIONS_DIR,
)


def _get_production_model():
    """Finds the most recent model in the registry."""
    models = list(MODEL_REGISTRY_DIR.glob("*.joblib"))
    if not models:
        raise FileNotFoundError(f"No models found in {MODEL_REGISTRY_DIR}")

    # Sort by modification time to get the latest trained model
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    return latest_model


def _build_prediction_features(matchups_df, features_df):
    """Joins matchups with team features using a Home/Away prefix strategy."""
    # Ensure team IDs/Names are strings for the merge
    matchups_df["home_team"] = matchups_df["home_team"].astype(str)
    matchups_df["away_team"] = matchups_df["away_team"].astype(str)

    # We join features twice: once for Home stats, once for Away stats
    home_feats = features_df.add_prefix("home_")
    away_feats = features_df.add_prefix("away_")

    # Merge Home Team data
    pred_df = matchups_df.merge(
        home_feats, left_on="home_team", right_on="home_team", how="left"
    )
    # Merge Away Team data
    pred_df = pred_df.merge(
        away_feats, left_on="away_team", right_on="away_team", how="left"
    )

    # Drop non-numeric metadata that isn't a model feature
    cols_to_drop = ["game_id", "date", "home_team", "away_team", "status", "season"]
    X = pred_df.drop(
        columns=[c for c in cols_to_drop if c in pred_df.columns], errors="ignore"
    )

    return X, pred_df


def run_prediction_for_date(target_date: date):
    """Orchestrates the full prediction flow for a specific date."""
    logger.info(f"Generating predictions for: {target_date}")

    # 1. Load Schedule and Filter for target_date
    if not SCHEDULE_SNAPSHOT.exists():
        logger.error(f"Missing schedule snapshot: {SCHEDULE_SNAPSHOT}")
        return

    df_schedule = pd.read_parquet(SCHEDULE_SNAPSHOT)

    # Handle date conversion to ensure matching
    df_schedule["date"] = pd.to_datetime(df_schedule["date"]).dt.date
    todays_games = df_schedule[df_schedule["date"] == target_date].copy()

    if todays_games.empty:
        logger.warning(f"No games scheduled for {target_date}.")
        return

    # 2. Load Features (using LONG_SNAPSHOT as the source of team stats)
    if not LONG_SNAPSHOT.exists():
        logger.error("Feature snapshot (LONG_SNAPSHOT) not found.")
        return

    features_df = pd.read_parquet(LONG_SNAPSHOT)
    # If LONG_SNAPSHOT has multiple rows per team, get the most recent one
    features_df = features_df.sort_values("date").groupby("team_id").tail(1)

    # 3. Prepare Feature Matrix
    X, combined_df = _build_prediction_features(todays_games, features_df)

    # 4. Load Model and Predict
    model_path = _get_production_model()
    logger.info(f"Predicting with model: {model_path.name}")
    model_data = joblib.load(model_path)

    # Handle if model is wrapped in a dict or is a raw pipeline
    model = model_data["model"] if isinstance(model_data, dict) else model_data

    probs = model.predict_proba(X)[:, 1]

    # 5. Format Results
    results = todays_games[["game_id", "home_team", "away_team"]].copy()
    results["home_win_probability"] = probs
    results["predicted_winner"] = results.apply(
        lambda x: x["home_team"] if x["home_win_probability"] > 0.5 else x["away_team"],
        axis=1,
    )

    # 6. Save
    out_path = PREDICTIONS_DIR / f"preds_{target_date}.csv"
    results.to_csv(out_path, index=False)
    logger.success(f"Predictions saved to {out_path}")

    return results


if __name__ == "__main__":
    run_prediction_for_date(date.today())
