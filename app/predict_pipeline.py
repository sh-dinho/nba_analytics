# File: app/predict_pipeline.py
import joblib
import pandas as pd
import os

MODEL_PATH = "models/game_predictor.pkl"

def predict_game(game):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Run train_model.py first.")

    model = joblib.load(MODEL_PATH)
    # Encode teams consistently
    teams = [game["home_team"], game["away_team"]]
    team_codes = pd.Series(teams).astype("category").cat.codes
    X = pd.DataFrame([team_codes], columns=["home_team_id","away_team_id"])
    prob_home = model.predict_proba(X)[0][1]
    return prob_home

def generate_today_predictions():
    # Placeholder: generate dummy predictions
    games = [
        {"date": "2025-12-01", "home_team": "LAL", "away_team": "BOS"},
        {"date": "2025-12-01", "home_team": "NYK", "away_team": "MIA"},
    ]
    rows = []
    for g in games:
        prob_home = predict_game(g)
        rows.append({
            "date": g["date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_win_prob": prob_home
        })
    return pd.DataFrame(rows)

def generate_today_predictions_with_totals():
    # Placeholder: add dummy totals
    df = generate_today_predictions()
    df["total_points_pred"] = 220  # dummy constant
    return df