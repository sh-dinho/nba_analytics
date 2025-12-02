import os
import pandas as pd
import numpy as np
import joblib

def generate_today_predictions(threshold=0.6):
    # Load features
    features = pd.read_csv("data/training_features.csv")
    features["date"] = pd.to_datetime(features["date"])

    # Filter for future games (no outcomes yet)
    today_mask = features["home_win"].isna()
    df = features.loc[today_mask].copy()

    # Load trained model
    model = joblib.load("models/game_predictor.pkl")
    feature_cols = [c for c in df.columns if c.startswith("home_") or c.startswith("away_")]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict home win probabilities
    proba = model.predict_proba(X)[:, 1]
    df_out = df[["game_id", "date", "home_team", "away_team"]].copy()
    df_out["home_win_prob"] = proba

    # Add odds if available
    if "decimal_odds_home" in df.columns and "decimal_odds_away" in df.columns:
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
    preds = generate_today_predictions()
    os.makedirs("results", exist_ok=True)

    # Save predictions
    preds.to_csv("results/predictions.csv", index=False)
    print("✅ Predictions saved to results/predictions.csv")

    # Save picks (only rows with a recommended bet)
    picks = preds[preds["pick"] != "No Bet"].copy() if "pick" in preds.columns else preds
    picks.to_csv("results/picks.csv", index=False)
    print("✅ Picks saved to results/picks.csv")