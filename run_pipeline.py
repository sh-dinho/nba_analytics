"""
File: run_pipeline.py
Path: project root / src
Purpose: Daily NBA pipeline to fetch historical data, prepare features,
         train ML model, and predict outcomes.
"""

import logging
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.features.features import prepare_features

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------
# Constants
# ----------------------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
MODEL_PATH = DATA_DIR / "models" / "nba_model.pkl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----------------------
# Load Historical Schedule
# ----------------------
def load_master_schedule() -> pd.DataFrame:
    master_file = CACHE_DIR / "master_schedule.parquet"
    if master_file.exists():
        df = pd.read_parquet(master_file)
        logger.info(f"Loaded master schedule ({df.shape[0]} rows)")
        return df
    logger.warning("No master schedule found. Returning empty DataFrame.")
    return pd.DataFrame()


# ----------------------
# Train Model
# ----------------------
def train_model(features: pd.DataFrame, target: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    logger.info("Model training complete.")
    return model


# ----------------------
# Save Model
# ----------------------
def save_model(model: RandomForestClassifier, path: Path):
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")


# ----------------------
# Predict Games
# ----------------------
def predict_games(
    model: RandomForestClassifier, features: pd.DataFrame
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    predictions = model.predict(features)
    result = features.copy()
    result["predicted_home_win"] = predictions
    logger.info(f"Predicted {len(predictions)} games.")
    return result


# ----------------------
# Main Pipeline
# ----------------------
def main():
    logger.info("===== NBA DAILY PIPELINE START =====")

    # Load historical data
    historical_schedule = load_master_schedule()
    if historical_schedule.empty:
        logger.error("No historical schedule available. Exiting.")
        return

    # Prepare features for training
    features_df = prepare_features(historical_schedule)
    if features_df.empty:
        logger.error("No features prepared. Exiting.")
        return

    # Split features and target
    target = features_df["target"]
    x = features_df.drop(columns=["target"])

    # Train the model
    model = train_model(x, target)

    # Save the model
    save_model(model, MODEL_PATH)

    # Predict upcoming games (if any)
    # Here we assume upcoming_games are in a CSV or similar
    upcoming_file = CACHE_DIR / "upcoming_games.parquet"
    if upcoming_file.exists():
        upcoming_games = pd.read_parquet(upcoming_file)
        upcoming_features = prepare_features(historical_schedule, upcoming_games)
        predictions = predict_games(model, upcoming_features)
        logger.info(predictions.head())
    else:
        logger.info("No upcoming games file found. Skipping predictions.")

    logger.info("===== NBA DAILY PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()
