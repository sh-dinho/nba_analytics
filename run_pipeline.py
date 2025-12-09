# ============================================================
# Path: run_pipeline.py
# Purpose: NBA analytics pipeline with caching, incremental updates,
#          progress logging, unique ID deduplication, and organized folders
# ============================================================

import pandas as pd
from datetime import date
import logging
import os

from src.prediction_engine.game_features import fetch_season_games, generate_features_for_games
from src.model_training.train_logreg import train_logreg
from src.prediction_engine.predictor import NBAPredictor

# -----------------------------
# Logging configuration
# -----------------------------
def configure_logging(log_file: str = "pipeline.log"):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured. Writing to %s", log_file)
    return log_file

# -----------------------------
# Ensure data directories exist
# -----------------------------
def ensure_dirs():
    os.makedirs("data/csv", exist_ok=True)
    os.makedirs("data/parquet", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("data/history", exist_ok=True)

# -----------------------------
# Helper: Add unique identifier
# -----------------------------
def add_unique_id(df):
    if "GAME_ID" in df.columns and "TEAM_ID" in df.columns:
        if "prediction_date" not in df.columns:
            df["prediction_date"] = pd.to_datetime(date.today()).strftime("%Y-%m-%d")
        df["unique_id"] = (
            df["GAME_ID"].astype(str) + "_" +
            df["TEAM_ID"].astype(str) + "_" +
            df["prediction_date"].astype(str)
        )
    return df

# -----------------------------
# Helper: Load cached features
# -----------------------------
def load_cached_features(cache_file="data/cache/features_full.parquet"):
    if os.path.exists(cache_file):
        logging.info("Loading cached features from %s", cache_file)
        return pd.read_parquet(cache_file)
    else:
        logging.info("No cache found, starting fresh")
        return None

# -----------------------------
# Helper: Save features to cache
# -----------------------------
def save_features_cache(df, cache_file="data/cache/features_full.parquet"):
    df = add_unique_id(df)
    df = df.drop_duplicates(subset=["unique_id"])
    df.to_parquet(cache_file, index=False)
    logging.info("Saved features cache to %s (rows=%d)", cache_file, len(df))

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    log_file = configure_logging("pipeline.log")
    ensure_dirs()
    logging.info("Pipeline started")

    cache_file = "data/cache/features_full.parquet"
    features_full = load_cached_features(cache_file)

    # If no cache, fetch all seasons (2022–2024 + 2025 so far)
    if features_full is None:
        logging.info("Fetching full historical dataset (2022–2024 + 2025 so far)")
        all_features = []
        for season in [2022, 2023, 2024, 2025]:
            game_ids = fetch_season_games(season, limit=None)
            logging.info("Season %d: %d games to fetch", season, len(game_ids))
            season_features = []
            for i, gid in enumerate(game_ids, start=1):
                season_features.append(generate_features_for_games([gid]))
                if i % 100 == 0 or i == len(game_ids):
                    logging.info("Season %d progress: %d/%d games fetched", season, i, len(game_ids))
            season_df = pd.concat(season_features, ignore_index=True)
            season_df = add_unique_id(season_df)
            all_features.append(season_df)
        features_full = pd.concat(all_features, ignore_index=True)
        save_features_cache(features_full, cache_file)
    else:
        # Incremental update: fetch new 2025 games not in cache
        logging.info("Checking for new 2025 games to append")
        game_ids_2025 = fetch_season_games(2025, limit=None)
        features_2025 = generate_features_for_games(game_ids_2025)
        features_2025 = add_unique_id(features_2025)

        features_full = pd.concat([features_full, features_2025], ignore_index=True)
        features_full = features_full.drop_duplicates(subset=["unique_id"])
        save_features_cache(features_full, cache_file)

    logging.info("Training dataset size: %d rows", len(features_full))

    # Train model
    result = train_logreg(cache_file, out_dir="models")
    print("Training metrics:", result["metrics"])
    logging.info("Training metrics: %s", result["metrics"])

    # Predict today’s games
    today = date.today().strftime("%Y-%m-%d")
    logging.info(f"Fetching games for {today}")
    game_ids_today = fetch_season_games(2025, limit=5)
    features_today = generate_features_for_games(game_ids_today)

    predictor = NBAPredictor(model_path="models/nba_logreg.pkl")
    X_today = features_today.drop(columns=["win"])
    labels_today = predictor.predict(X_today)
    probas_today = predictor.predict_proba(X_today)

    features_today["pred_label"] = labels_today
    features_today["pred_proba"] = probas_today
    features_today["prediction_date"] = today
    features_today = add_unique_id(features_today)
    features_today = features_today.drop_duplicates(subset=["unique_id"])

    print("Today's games with predictions:")
    print(features_today)

    # Save outputs
    features_today.to_csv("data/csv/predictions_today.csv", index=False)
    features_today.to_parquet("data/parquet/predictions_today.parquet", index=False)
    features_today.to_csv(f"data/csv/predictions_{today}.csv", index=False)
    features_today.to_parquet(f"data/parquet/predictions_{today}.parquet", index=False)

    # Append to history with deduplication
    history_file = "data/history/predictions_history.parquet"
    if os.path.exists(history_file):
        history_df = pd.read_parquet(history_file)
        history_df = pd.concat([history_df, features_today], ignore_index=True)
        history_df = history_df.drop_duplicates(subset=["unique_id"])
    else:
        history_df = features_today
    history_df.to_parquet(history_file, index=False)
    logging.info("Appended today's predictions to %s (rows=%d)", history_file, len(history_df))

    logging.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()
