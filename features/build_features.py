# ============================================================
# File: generate_features.py
# Purpose: Safely generate team & player feature CSVs for modeling
# ============================================================

import pandas as pd
from pathlib import Path

from nba_core.paths import (
    DATA_DIR,
    TRAINING_FEATURES_FILE,
    PLAYER_FEATURES_FILE,
    HISTORICAL_GAMES_FILE,
    PLAYER_GAMES_FILE,
    ensure_dirs,
)
from nba_core.config import USE_ROLLING_AVG, ROLLING_WINDOW
from nba_core.log_config import init_global_logger
from nba_core.exceptions import DataError

# Import new feature builder
from features.feature_builder import build_features

logger = init_global_logger()
ensure_dirs(strict=False)

# ---------------- Helper Functions ----------------
def safe_read_csv(file_path: Path):
    """Safely read CSV, return empty DataFrame if missing or error occurs."""
    if not file_path.exists():
        logger.warning(f"⚠️ File not found: {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"⚠️ File is empty: {file_path}. Returning empty DataFrame.")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to read {file_path}: {e}")
        return pd.DataFrame()


# ---------------- Team Features ----------------
def generate_team_features():
    """
    Generate team features using advanced feature builder with rolling averages.
    Falls back to placeholder if no data is available.
    """
    try:
        df_features = build_features(out_file=TRAINING_FEATURES_FILE)
    except DataError as e:
        logger.warning(f"{e}. Generating placeholder team features.")
        df_features = pd.DataFrame({
            "home_team": [],
            "away_team": [],
            "home_win": [],
        })
        TRAINING_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"📂 Team features saved → {TRAINING_FEATURES_FILE}")
    return df_features


# ---------------- Player Features ----------------
def generate_player_features():
    """
    Generate player-level features by aggregating player games.
    """
    df = safe_read_csv(PLAYER_GAMES_FILE)
    if df.empty:
        logger.warning("No player games found. Generating placeholder player features.")
        df = pd.DataFrame({
            "player_id": [],
            "player_name": [],
        })

    # Aggregate stats per player
    player_features = df.groupby("player_id").agg({
        "player_name": "first",
        "PTS": "mean" if "PTS" in df.columns else "first",
        "AST": "mean" if "AST" in df.columns else "first",
        "REB": "mean" if "REB" in df.columns else "first",
    }).reset_index()

    # Fallback dummy numeric features if empty
    if player_features.shape[0] == 0:
        player_features = pd.DataFrame({
            "player_feat1": [],
            "player_feat2": [],
            "player_name": []
        })

    PLAYER_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    player_features.to_csv(PLAYER_FEATURES_FILE, index=False)
    logger.info(f"📂 Player features saved → {PLAYER_FEATURES_FILE}")
    return player_features


# ---------------- Entrypoint ----------------
def main():
    generate_team_features()
    generate_player_features()
    logger.info("✅ Feature generation complete.")


if __name__ == "__main__":
    main()
