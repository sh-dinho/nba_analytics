# Prediction pipeline placeholder
def generate_predictions():
    pass
import os
import pandas as pd
import joblib
from datetime import datetime
import logging

from nba_analytics_core.data import fetch_today_games, build_team_stats, build_matchup_features
from nba_analytics_core.odds import fetch_odds

MODEL_PATH = "models/classification_model.pkl"

def generate_predictions(threshold=0.6, strategy="kelly", max_fraction=0.05, cli=False):
    """
    Generate predictions for today's games by combining model probabilities with live odds.
    Returns a DataFrame with columns: home_team, away_team, pred_home_win_prob, ev, decimal_odds.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first: python scripts/train_model.py")

    model = joblib.load(MODEL_PATH)
    games = fetch_today_games()
    if not games:
        logging.info("No games found for today.")
        return pd.DataFrame(columns=["home_team", "away_team", "pred_home_win_prob", "ev", "decimal_odds"])

    # Build simple team stats from recent history (fallback to equal stats if none)
    team_stats = build_team_stats(games)

    rows = []
    for g in games:
        home = g["home_team"]
        away = g["away_team"]

        features = build_matchup_features(home, away, team_stats)
        prob = float(model.predict_proba([features])[0][1])  # prob of home win

        odds = fetch_odds(home_team=home, away_team=away)
        if not odds:
            continue

        home_odds = odds["home_odds"]
        ev = (prob * (home_odds - 1)) - (1 - prob)

        rows.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "pred_home_win_prob": round(prob, 4),
            "decimal_odds": home_odds,
            "ev": round(ev, 4)
        })

    df = pd.DataFrame(rows).sort_values(by="ev", ascending=False)

    if cli and not df.empty:
        print("\n=== BEST PICKS ===")
        for _, r in df[df["pred_home_win_prob"] >= threshold].iterrows():
            print(f'{r["home_team"]} vs {r["away_team"]} | Prob={r["pred_home_win_prob"]:.2f} | EV={r["ev"]:.3f}')

    return df