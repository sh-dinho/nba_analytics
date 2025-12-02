# File: scripts/build_features.py
# FIX: Use config.py paths. Removed synthetic label fallback for training.

import os
import logging
import pandas as pd
from core.config import TRAINING_FEATURES_FILE, PLAYER_STATS_FILE, GAME_RESULTS_FILE 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_features():
    """Build training features, requires actual game results."""
    os.makedirs(os.path.dirname(TRAINING_FEATURES_FILE), exist_ok=True) 

    if not os.path.exists(PLAYER_STATS_FILE):
        raise FileNotFoundError(f"{PLAYER_STATS_FILE} not found. Run fetch_player_stats.py first.")

    df = pd.read_csv(PLAYER_STATS_FILE)

    # Add engineered features
    df["PTS_per_AST"] = df["PTS"] / df["AST"].replace(0, 1)
    df["REB_rate"] = df["REB"] / df["GAMES_PLAYED"].replace(0, 1)

    # Add ACTUAL labels - CRITICAL FIX: Must have real labels for training
    if os.path.exists(GAME_RESULTS_FILE):
        results = pd.read_csv(GAME_RESULTS_FILE)
        df = df.merge(results, on=["PLAYER_NAME"], how="left")
        
        if 'home_win' not in df.columns or df['home_win'].isnull().all():
             logger.warning("⚠️ Labels not successfully merged. Training data may be incomplete.")
    else:
        # CRITICAL FIX: Raise error instead of adding synthetic labels for training
        raise FileNotFoundError(f"Missing ACTUAL game results file: {GAME_RESULTS_FILE}. Cannot create training data.")

    df.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Features saved to {TRAINING_FEATURES_FILE} ({len(df)} rows)")

def build_features_for_new_games(new_games_file: str) -> pd.DataFrame:
    # ... (function body remains the same but uses central config paths if updated) ...
    # This is where features for NEW games are built.
    if not os.path.exists(new_games_file):
        raise FileNotFoundError(f"{new_games_file} not found. Run fetch_new_games.py first.")
    
    df = pd.read_csv(new_games_file)
    
    df["PTS_per_AST"] = df["PTS"] / df["AST"].replace(0, 1)
    df["REB_rate"] = df["REB"] / df["GAMES_PLAYED"].replace(0, 1)
    
    return df