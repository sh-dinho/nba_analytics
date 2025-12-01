# File: app/predict_pipeline.py
import pandas as pd
import joblib

# Load pipeline (includes imputer + model)
model = joblib.load("models/game_predictor.pkl")

def predict_game(game_features: pd.DataFrame) -> float:
    """
    Predict probability of home team winning.
    game_features: single-row DataFrame with feature columns
    """
    prob_home = model.predict_proba(game_features)[0][1]
    return prob_home

def generate_today_predictions():
    # Example: load today’s games features
    df_games = pd.read_csv("data/today_games.csv")  # adjust path

    results = []
    for _, row in df_games.iterrows():
        X = row.drop(labels=["date", "home_team", "away_team"]).to_frame().T
        prob_home = predict_game(X)
        results.append({
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_prob": prob_home,
            "away_win_prob": 1 - prob_home
        })

    return pd.DataFrame(results)