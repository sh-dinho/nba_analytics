# File: scripts/generate_today_predictions.py

import os
import pandas as pd
import numpy as np
import joblib
import logging
import json
from datetime import datetime
import argparse
from nba_analytics_core.notifications import send_telegram_message

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("predictions")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def moneyline_to_decimal(ml):
    """Convert American moneyline to decimal odds."""
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        return ml / 100 + 1
    else:
        return 100 / abs(ml) + 1


def generate_today_predictions(
    threshold=0.6,
    features_file="data/training_features.csv",
    model_file="models/game_predictor.pkl",
    notify=False,
    outdir="results"
):
    """
    Generate today's NBA predictions using a trained model.
    Adds EV calculations, strong/weak classification, and top EV notifications.

    Parameters:
        threshold: probability threshold for strong picks
        features_file: path to features CSV
        model_file: path to trained model file
        notify: send Telegram notification for top EV pick
        outdir: output directory
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

    # Default pick based on probability threshold
    df_out["pick"] = np.where(df_out["home_win_prob"] >= threshold, df_out["home_team"], df_out["away_team"])
    df_out["confidence"] = np.where(df_out["home_win_prob"] >= threshold, "Strong", "Weak")

    # Handle odds if available
    odds_cols_moneyline = {"home_moneyline", "away_moneyline"}
    odds_cols_decimal = {"decimal_odds_home", "decimal_odds_away"}

    if odds_cols_moneyline.issubset(df.columns) or odds_cols_decimal.issubset(df.columns):
        # Convert moneyline to decimal odds if needed
        if odds_cols_moneyline.issubset(df.columns):
            df_out["decimal_odds_home"] = df["home_moneyline"].apply(moneyline_to_decimal)
            df_out["decimal_odds_away"] = df["away_moneyline"].apply(moneyline_to_decimal)
        else:
            df_out["decimal_odds_home"] = df["decimal_odds_home"]
            df_out["decimal_odds_away"] = df["decimal_odds_away"]

        # Expected value calculations
        df_out["ev_home"] = df_out["home_win_prob"] * (df_out["decimal_odds_home"] - 1) - (1 - df_out["home_win_prob"])
        df_out["ev_away"] = (1 - df_out["home_win_prob"]) * (df_out["decimal_odds_away"] - 1) - df_out["home_win_prob"]

        # Pick based on EV
        df_out["pick_ev"] = np.where(df_out["ev_home"] >= df_out["ev_away"], df_out["home_team"], df_out["away_team"])
        # Final pick: only strong picks override EV if probability below threshold
        df_out["pick"] = np.where(df_out["confidence"] == "Strong", df_out["pick"], df_out["pick_ev"])

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Save main predictions
    preds_file = os.path.join(outdir, "predictions.csv")
    df_out.to_csv(preds_file, index=False)
    ts_file = os.path.join(outdir, f"predictions_{_timestamp()}.csv")
    df_out.to_csv(ts_file, index=False)

    # Save picks only (exclude weak/no bet if EV available)
    picks_file = os.path.join(outdir, "picks.csv")
    picks_df = df_out[df_out["pick"] != "No Bet"].copy()
    picks_df.to_csv(picks_file, index=False)
    ts_picks_file = os.path.join(outdir, f"picks_{_timestamp()}.csv")
    picks_df.to_csv(ts_picks_file, index=False)

    logger.info(f"✅ Predictions saved to {preds_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_file}")
    logger.info(f"✅ Picks saved to {picks_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_picks_file}")

    # Telegram notification for top EV pick
    if notify and "ev_home" in df_out.columns:
        top_game = df_out.loc[df_out[["ev_home", "ev_away"]].max(axis=1).idxmax()]
        msg = (
            f"🏀 Today's Top EV Pick\n"
            f"{top_game['home_team']} vs {top_game['away_team']}\n"
            f"Pick: {top_game['pick']} | EV Home: {top_game['ev_home']:.2f}, EV Away: {top_game['ev_away']:.2f}"
        )
        try:
            send_telegram_message(msg)
            logger.info("✅ Telegram notification sent for top EV pick")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram message: {e}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(df_out),
        "columns": df_out.columns.tolist(),
        "features_file": features_file,
        "model_file": model_file,
        "outdir": outdir,
        "threshold": threshold
    }
    meta_file = os.path.join(outdir, "predictions_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate today's NBA predictions")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--features", type=str, default="data/training_features.csv", help="Path to features CSV")
    parser.add_argument("--model", type=str, default="models/game_predictor.pkl", help="Path to trained model file")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification for top EV pick")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    try:
        generate_today_predictions(
            threshold=args.threshold,
            features_file=args.features,
            model_file=args.model,
            notify=args.notify,
            outdir=args.outdir
        )
    except Exception as e:
        logger.error(f"❌ Prediction generation failed: {e}")
        exit(1)
