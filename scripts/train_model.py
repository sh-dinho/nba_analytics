# File: scripts/train_model.py
import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Example: replace with your actual data loader
def fetch_historical_games(seasons):
    # Placeholder: generate dummy data
    data = []
    for season in seasons:
        data.append({"season": season, "home_team": "LAL", "away_team": "BOS", "home_win": 1})
        data.append({"season": season, "home_team": "NYK", "away_team": "MIA", "home_win": 0})
    return pd.DataFrame(data)

def train_model(seasons, train_ou=False):
    print(f"Training model on seasons: {seasons} (train_ou={train_ou})")
    df = fetch_historical_games(seasons)

    # Simple features: home_team vs away_team encoded
    df["home_team_id"] = df["home_team"].astype("category").cat.codes
    df["away_team_id"] = df["away_team"].astype("category").cat.codes
    X = df[["home_team_id", "away_team_id"]]
    y = df["home_win"]

    model = LogisticRegression()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/game_predictor.pkl")
    print("✅ Model trained and saved to models/game_predictor.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True, help="List of seasons to train on")
    parser.add_argument("--train_ou", action="store_true", help="Train over/under model as well")
    args = parser.parse_args()

    train_model(args.seasons, args.train_ou)