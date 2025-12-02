# File: scripts/generate_today_predictions.py

import os
import pandas as pd
import numpy as np
import joblib
import logging
import json
from datetime import datetime
import argparse

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("predictions")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


def generate_today_predictions(
    threshold=0.6,
    features_file="data/training_features.csv",
    model_file="models/game_predictor.pkl"
):
    """
    Generate today's NBA predictions using a trained model.

    Parameters:
        threshold: probability threshold for strong picks
        features_file: path to features CSV
        model_file: path to trained model file

    Returns:
        DataFrame of predictions and picks
    """
    # Load features
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    features = pd.read_csv(features_file)

    if "date" in features.columns:
        features["date"] = pd.to_datetime(features["date"], errors="coerce")

    # Validate required column
    if "home_win" not in features.columns:
        raise ValueError("Missing 'home_win' column in features file.")

    # Filter for future games (no outcomes yet)
    today_mask = features["home_win"].isna()
    df = features.loc[today_mask].copy()
    if df.empty:
        logger.warning("No future games found in features file.")
        return pd.DataFrame()

    # Load trained model
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = joblib.load(model_file)

    feature_cols = [c for c in df.columns if c.startswith("home_") or c.startswith("away_")]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict home win probabilities
    proba = model.predict_proba(X)[:, 1]
    df_out = df[["game_id", "date", "home_team", "away_team"]].copy()
    df_out["home_win_prob"] = proba

    # Add odds if available
    if {"decimal_odds_home", "decimal_odds_away"}.issubset(df.columns):
        df_out["decimal_odds_home"] = df["decimal_odds_home"]
        df_out["decimal_odds_away"] = df["decimal_odds_away"]

        # Expected value for betting on home/away
        df_out["ev_home"] = df_out["home_win_prob"] * (df_out["decimal_odds_home"] - 1) - (1 - df_out["home_win_prob"])
        df_out["ev_away"] = (1 - df_out["home_win_prob"]) * (df_out["decimal_odds_away"] - 1) - df_out["home_win_prob"]

        # Pick side with higher EV (only if above threshold)
        df_out["pick"] = np.where(
            (df_out["ev_home"] > df_out["ev_away"]) & (df_out["home_win_prob"] >= threshold),
            df_out["home_team"],
            np.where(
                (df_out["ev_away"] > df_out["ev_home"]) & ((1 - df_out["home_win_prob"]) >= threshold),
                df_out["away_team"],
                "No Bet"
            )
        )

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate today's NBA predictions")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--features", type=str, default="data/training_features.csv", help="Path to features CSV")
    parser.add_argument("--model", type=str, default="models/game_predictor.pkl", help="Path to trained model file")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    try:
        preds = generate_today_predictions(
            threshold=args.threshold,
            features_file=args.features,
            model_file=args.model
        )
    except Exception as e:
        logger.error(f"❌ Prediction generation failed: {e}")
        exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if preds.empty:
        logger.info("No predictions generated.")
    else:
        # Save predictions
        preds_file = os.path.join(args.outdir, "predictions.csv")
        preds.to_csv(preds_file, index=False)
        logger.info(f"✅ Predictions saved to {preds_file}")

        # Save picks (only rows with a recommended bet)
        picks = preds[preds["pick"] != "No Bet"].copy() if "pick" in preds.columns else preds
        picks_file = os.path.join(args.outdir, "picks.csv")
        picks.to_csv(picks_file, index=False)
        logger.info(f"✅ Picks saved to {picks_file}")

        # Save metadata
        meta = {
            "generated_at": datetime.now().isoformat(),
            "rows": len(preds),
            "threshold": args.threshold,
            "features_file": args.features,
            "model_file": args.model,
            "outdir": args.outdir
        }
        meta_file = os.path.join(args.outdir, "predictions_meta.json")
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"🧾 Metadata saved to {meta_file}")