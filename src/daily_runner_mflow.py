# ============================================================
# File: src/daily_runner/daily_runner_mflow.py
# Purpose: Daily NBA prediction runner with MLflow logging
# Version: 1.1 (MLflow + NBAPredictor + enhanced features)
# ============================================================

import os
import logging
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
import mlflow

from src.prediction_engine.predictor import Predictor
from src.features.feature_engineering import generate_features_for_games
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
    logging.info(f"Fetching games for season {season}")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]
        df = df[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'TEAM_ID', 'OPPONENT_TEAM_ID', 'PTS']]
        return df
    except Exception as e:
        logging.error(f"Error fetching season {season}: {e}")
        return pd.DataFrame()


def update_historical_data(existing_df: pd.DataFrame) -> pd.DataFrame:
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

    today_games['prediction_date'] = today_str
    today_games['unique_id'] = today_games['GAME_ID'].astype(str) + "_" + today_games['TEAM_ID'].astype(str) + "_" + today_str
    today_games['PointSpread'] = 0
    today_games['OverUnder'] = 0
    today_games['Players20PlusPts'] = 0
    return today_games


def generate_predictions(today_games: pd.DataFrame, model_path: str) -> pd.DataFrame:
    if today_games.empty:
        return today_games

    features = generate_features_for_games(today_games.to_dict(orient="records"))
    features = map_team_ids(features, "TEAM_ID")

    predictor = Predictor(model_path=model_path)
    features["win_proba"] = predictor.predict_proba(features)
    features["win_pred"] = predictor.predict_label(features)

    return features


def log_to_mlflow(predictions: pd.DataFrame):
    if predictions.empty:
        return
    with mlflow.start_run(run_name=f"daily_predictions_{datetime.now().strftime('%Y%m%d')}"):
        # Log model used
        mlflow.log_param("model_path", MODEL_PATH)
        # Save today's predictions as CSV
        csv_path = f"data/csv/daily_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        predictions.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="daily_predictions")
        # Log summary metrics
        avg_proba = predictions["win_proba"].mean()
        mlflow.log_metric("avg_win_proba", avg_proba)
        logging.info(f"Logged predictions to MLflow. Average win probability: {avg_proba:.2f}")


def print_summary(predictions: pd.DataFrame):
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
    logging.info("Starting MLflow NBA daily runner...")

    try:
        existing_df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logging.info("No historical data found, starting fresh.")
        existing_df = pd.DataFrame(columns=['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'TEAM_ID', 'OPPONENT_TEAM_ID', 'PTS'])

    updated_games_df = update_historical_data(existing_df)
    today_games = fetch_today_games()
    predictions = generate_predictions(today_games, MODEL_PATH)
    print_summary(predictions)
    log_to_mlflow(predictions)

    logging.info("MLflow NBA daily runner finished.")
