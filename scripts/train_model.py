# scripts/train_model.py
import logging
import joblib
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def current_season() -> str:
    """Return current NBA season string, e.g. '2025-26'."""
    year = datetime.now().year
    # NBA season starts in October, so adjust if before July
    if datetime.now().month < 7:
        return f"{year-1}-{str(year)[-2:]}"
    else:
        return f"{year}-{str(year+1)[-2:]}"

def fetch_historical_games(seasons) -> pd.DataFrame:
    """Fetch NBA games across multiple seasons."""
    all_games = []
    for season in seasons:
        logging.info(f"Fetching NBA games for season {season}...")
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = gamefinder.get_data_frames()[0]
        games["SEASON"] = season
        all_games.append(games)
        logging.info(f"✔ Retrieved {len(games)} games for {season}")
    return pd.concat(all_games, ignore_index=True)

def build_features(games: pd.DataFrame):
    """Build features and labels for training."""
    features = games[["PTS", "REB", "AST", "FG_PCT", "PLUS_MINUS"]].copy()
    labels = (games["PLUS_MINUS"] > 0).astype(int)  # 1 = win, 0 = loss
    return features, labels

def train_and_save(seasons, model_path="models/classification_model.pkl"):
    games = fetch_historical_games(seasons)
    X, y = build_features(games)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    logging.info(f"Model accuracy on test set: {acc:.3f}")

    joblib.dump(model, model_path)
    logging.info(f"✔ Model saved to {model_path}")

if __name__ == "__main__":
    # Train on last 5 seasons including current
    current = current_season()
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25", current]
    train_and_save(seasons)