# ============================================================
# File: src/daily_runner/daily_runner.py
# Purpose: Daily NBA prediction runner
# Version: 1.0 (MLflow + NBAPredictor + enhanced features)
# ============================================================

import os
import logging
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder

from src.prediction_engine.predictor import NBAPredictor
from src.features.game_features import generate_features_for_games
from src.utils.mapping import map_team_ids

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "data/cache/games_history.csv"
MODEL_PATH = "models/nba_logreg.pkl"
LOG_FILE = "data/logs/daily_runner.log"

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# Setup logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def fetch_season_games(season: str) -> pd.DataFrame:
    """Fetch all games for a season using nba_api."""
    logging.info(f"Fetching games for season {season}")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]
        df = df[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'TEAM_ID', 'OPPONENT_TEAM_ID', 'PTS']]  # keep useful columns
        return df
    except Exception as e:
        logging.error(f"Error fetching season {season}: {e}")
        return pd.DataFrame()


def update_historical_data(existing_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch historical games and merge with existing cache."""
    all_new_games = []
    for season in SEASONS:
        df = fetch_season_games(season)
        if not df.empty:
            all_new_games.append(df)
    if all_new_games:
        combined_df = pd.concat(all_new_games, ignore_index=True)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True).drop_duplicates(
            subset=['GAME_DATE', 'TEAM_NAME', 'MATCHUP']
        )
    else:
        combined_df = existing_df

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    combined_df.to_csv(DATA_FILE, index=False)
    logging.info(f"Updated historical games. Total games: {len(combined_df)}")
    return combined_df


def fetch_today_games() -> pd.DataFrame:
    """Fetch today’s NBA games from nba_api."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        current_season = SEASONS[-1]
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=current_season)
        df = gamefinder.get_data_frames()[0]
    except Exception as e:
        logging.error(f"Error fetching today's games: {e}")
        return pd.DataFrame()

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    today_games = df[df['GAME_DATE'] == pd.to_datetime(today_str)]
    if today_games.empty:
        logging.info("No NBA games today")
        return pd.DataFrame()

    # Add enhanced columns
    today_games['prediction_date'] = today_str
    today_games['unique_id'] = today_games['GAME_ID'].astype(str) + "_" + today_games['TEAM_ID'].astype(str) + "_" + today_str

    # Placeholder columns for advanced stats
    today_games['PointSpread'] = 0
    today_games['OverUnder'] = 0
    today_games['Players20PlusPts'] = 0

    return today_games


def generate_predictions(today_games: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Generate predictions using NBAPredictor and return enhanced DataFrame."""
    if today_games.empty:
        return today_games

    features = generate_features_for_games(today_games.to_dict(orient="records"))
    features = map_team_ids(features, "TEAM_ID")

    predictor = NBAPredictor(model_path=model_path)
    features["win_proba"] = predictor.predict_proba(features)
    features["win_pred"] = predictor.predict_label(features)

    return features


def print_summary(predictions: pd.DataFrame):
    """Print human-readable summary of today’s games and predicted winners."""
    if predictions.empty:
        print("No NBA games today.")
        return
    print("\nToday's NBA Predictions:")
    for _, row in predictions.iterrows():
        winner = row["TEAM_NAME"] if row["win_pred"] else row["OPPONENT_TEAM_ID"]
        print(
            f"{row['TEAM_NAME']} vs Opponent {row['OPPONENT_TEAM_ID']} | "
            f"Win probability: {row['win_proba']:.2f} | Predicted winner: {winner}"
        )


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    logging.info("Starting daily NBA runner...")

    # Load historical data
    try:
        existing_df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logging.info("No historical data found, starting fresh.")
        existing_df = pd.DataFrame(columns=['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'TEAM_ID', 'OPPONENT_TEAM_ID', 'PTS'])

    # Update historical games
    updated_games_df = update_historical_data(existing_df)

    # Fetch today's games
    today_games = fetch_today_games()

    # Generate predictions
    predictions = generate_predictions(today_games, MODEL_PATH)

    # Print summary
    print_summary(predictions)

    # Save today's predictions
    if not predictions.empty:
        os.makedirs("data/csv", exist_ok=True)
        predictions.to_csv(f"data/csv/daily_predictions_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        logging.info("Saved today's predictions CSV.")

    logging.info("Daily NBA runner finished.")
