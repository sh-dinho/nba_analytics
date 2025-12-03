# ============================================================
# File: scripts/build_features.py
# Purpose: Build training features from player stats and game results
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from core.config import TRAINING_FEATURES_FILE, PLAYER_STATS_FILE, GAME_RESULTS_FILE
from core.log_config import setup_logger
from core.exceptions import DataError
from core.utils import ensure_columns

logger = setup_logger("build_features")


def build_features() -> pd.DataFrame:
    """Build training features, requires actual game results."""
    TRAINING_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not PLAYER_STATS_FILE.exists():
        raise FileNotFoundError(f"{PLAYER_STATS_FILE} not found. Run fetch_player_stats.py first.")

    df = pd.read_csv(PLAYER_STATS_FILE)

    # Validate input schema
    required_input_cols = {"PLAYER_NAME", "PTS", "AST", "REB", "GAMES_PLAYED"}
    missing = required_input_cols - set(df.columns)
    if missing:
        raise DataError(f"Missing columns in player stats: {missing}")

    # Add engineered features
    df["PTS_per_AST"] = np.where(df["AST"] > 0, df["PTS"] / df["AST"], 0)
    df["REB_rate"] = np.where(df["GAMES_PLAYED"] > 0, df["REB"] / df["GAMES_PLAYED"], 0)

    # Add ACTUAL labels
    if GAME_RESULTS_FILE.exists():
        results = pd.read_csv(GAME_RESULTS_FILE)
        df = df.merge(results, on=["PLAYER_NAME"], how="left")

        if "home_win" not in df.columns or df["home_win"].isnull().all():
            logger.warning("⚠️ Labels not successfully merged. Training data may be incomplete.")
    else:
        raise FileNotFoundError(f"Missing ACTUAL game results file: {GAME_RESULTS_FILE}")

    # Validate required columns
    ensure_columns(df, {"PLAYER_NAME", "PTS", "AST", "REB", "home_win"}, "training features")

    df.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Features saved to {TRAINING_FEATURES_FILE} ({len(df)} rows)")
    return df


def build_features_for_new_games(new_games_file) -> pd.DataFrame:
    """Build features for new games (no labels)."""
    new_games_file = Path(new_games_file)
    if not new_games_file.exists():
        raise FileNotFoundError(f"{new_games_file} not found. Run fetch_new_games.py first.")

    df = pd.read_csv(new_games_file)

    df["PTS_per_AST"] = np.where(df["AST"] > 0, df["PTS"] / df["AST"], 0)
    df["REB_rate"] = np.where(df["GAMES_PLAYED"] > 0, df["REB"] / df["GAMES_PLAYED"], 0)

    ensure_columns(df, {"PLAYER_NAME", "PTS", "AST", "REB"}, "new game features")

    logger.info(f"✅ Features built for new games ({len(df)} rows)")
    return df


# Alias for pipeline compatibility
main = build_features