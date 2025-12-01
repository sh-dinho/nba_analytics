# Path: app/predict_pipeline.py
import os
import joblib
import pandas as pd
from datetime import datetime
from nba_analytics_core.data import fetch_today_games, build_team_stats, build_matchup_features

MODEL_PATH = "models/classification_model.pkl"
OU_MODEL_PATH = "models/ou_model.pkl"

def generate_today_predictions():
    """
    Predict home win probabilities for today's games.
    Returns DataFrame: date, home_team, away_team, home_win_prob.
    """
    if not os.path.exists(MODEL_PATH):
        return pd.DataFrame()

    model = joblib.load(MODEL_PATH)
    games = fetch_today_games()
    if not games:
        return pd.DataFrame()

    # Build minimal team stats context (from today's pairs as placeholder)
    team_stats = build_team_stats(games)

    rows = []
    for g in games:
        feats = build_matchup_features(g["home_team"], g["away_team"], team_stats)
        prob = float(model.predict_proba([feats])[0][1])
        rows.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_win_prob": round(prob, 4),
        })

    return pd.DataFrame(rows).sort_values(by="home_win_prob", ascending=False)

def generate_today_predictions_with_totals(line=220.0):
    """
    Predict Over/Under probabilities for today's games vs a given line.
    Requires OU model; falls back if not present.
    """
    base = generate_today_predictions()
    if base.empty or not os.path.exists(OU_MODEL_PATH):
        return base

    ou_model = joblib.load(OU_MODEL_PATH)
    games = fetch_today_games()
    team_stats = build_team_stats(games)

    probs_over = []
    for _, row in base.iterrows():
        feats = build_matchup_features(row["home_team"], row["away_team"], team_stats)
        prob_over = float(ou_model.predict_proba([feats])[0][1])
        probs_over.append(prob_over)

    base["prob_over"] = probs_over
    base["prob_under"] = 1 - base["prob_over"]
    base["line"] = line
    return base