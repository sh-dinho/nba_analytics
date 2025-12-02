# ============================================================
# File: prediction.py
# Path: <project_root>/prediction.py
#
# Generates predictions for today's NBA games using the trained
# model and feature set. Integrates fully with config.py paths
# and settings (DATA_DIR, MODELS_DIR, RESULTS_DIR, DEFAULT_THRESHOLD).
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib

from config import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    DEFAULT_THRESHOLD,
)


def generate_today_predictions(
    features_path=None,
    model_path=None,
    threshold=None
):
    # --- Resolve default paths from config.py ---
    features_path = features_path or os.path.join(DATA_DIR, "training_features.csv")
    model_path = model_path or os.path.join(MODELS_DIR, "game_predictor.pkl")
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold

    # --- Load features ---
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"❌ Features file not found: {features_path}")

    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Identify future/unplayed games
    future_mask = df["home_win"].isna()
    df_future = df.loc[future_mask].copy()

    if df_future.empty:
        print("ℹ️ No games found to predict.")
        return pd.DataFrame()

    # --- Load model ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = joblib.load(model_path)

    # --- Prepare features ---
    feature_cols = [c for c in df_future.columns if c.startswith(("home_", "away_"))]
    X = df_future[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Predict probabilities ---
    df_out = df_future[["game_id", "date", "home_team", "away_team"]].copy()
    df_out["home_win_prob"] = model.predict_proba(X)[:, 1]

    # --- Odds & Expected Value ---
    if {"decimal_odds_home", "decimal_odds_away"}.issubset(df_future.columns):

        df_out["decimal_odds_home"] = df_future["decimal_odds_home"]
        df_out["decimal_odds_away"] = df_future["decimal_odds_away"]

        # EV for home
        df_out["ev_home"] = (
            df_out["home_win_prob"] * (df_out["decimal_odds_home"] - 1)
            - (1 - df_out["home_win_prob"])
        )

        # EV for away
        df_out["ev_away"] = (
            (1 - df_out["home_win_prob"]) * (df_out["decimal_odds_away"] - 1)
            - df_out["home_win_prob"]
        )

        # Determine pick
        df_out["pick"] = np.where(
            (df_out["ev_home"] > df_out["ev_away"]) &
            (df_out["home_win_prob"] >= threshold),
            df_out["home_team"],
            np.where(
                (df_out["ev_away"] > df_out["ev_home"]) &
                ((1 - df_out["home_win_prob"]) >= threshold),
                df_out["away_team"],
                "No Bet"
            )
        )
    else:
        df_out["pick"] = "No Odds Available"

    return df_out


if __name__ == "__main__":
    predictions = generate_today_predictions()

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save output
    predictions_file = os.path.join(RESULTS_DIR, "predictions.csv")
    picks_file = os.path.join(RESULTS_DIR, "picks.csv")

    predictions.to_csv(predictions_file, index=False)
    print(f"✅ Predictions saved to {predictions_file}")

    # Only save picks with an actual bet
    if "pick" in predictions.columns:
        picks = predictions[predictions["pick"] != "No Bet"].copy()
    else:
        picks = predictions.copy()

    picks.to_csv(picks_file, index=False)
    print(f"✅ Picks saved to {picks_file}")
