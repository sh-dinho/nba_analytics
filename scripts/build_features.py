# ============================================================
# File: scripts/build_features.py
# Purpose: Build features for new games (aggregate player stats to team level)
# ============================================================

import os
import pandas as pd
from core.config import TRAINING_FEATURES_FILE, NEW_GAMES_FILE
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("build_features")


def build_features_for_new_games(new_games_file: str = NEW_GAMES_FILE) -> str:
    """
    Build team-level features from player-level new_games.csv.
    Aggregates player stats into team averages/sums.
    Saves to TRAINING_FEATURES_FILE and returns path.
    """
    if not os.path.exists(new_games_file):
        raise FileNotFoundError(f"{new_games_file} not found. Run fetch_new_games.py first.")

    df = pd.read_csv(new_games_file)

    # Ensure required columns exist
    ensure_columns(df, {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "GAMES_PLAYED"}, "new game features")

    # --- Aggregate player stats into team-level features ---
    team_features = df.groupby("TEAM_ABBREVIATION").agg({
        "PTS": "mean",          # average points per player
        "AST": "mean",          # average assists
        "REB": "mean",          # average rebounds
        "GAMES_PLAYED": "mean"  # average games played
    }).reset_index()

    # Rename columns for clarity
    team_features.rename(columns={
        "TEAM_ABBREVIATION": "team",
        "PTS": "avg_pts",
        "AST": "avg_ast",
        "REB": "avg_reb",
        "GAMES_PLAYED": "avg_games_played"
    }, inplace=True)

    # --- Save features ---
    os.makedirs(os.path.dirname(TRAINING_FEATURES_FILE), exist_ok=True)
    team_features.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Features saved to {TRAINING_FEATURES_FILE} ({len(team_features)} rows)")

    # ✅ Return the path, not the DataFrame
    return TRAINING_FEATURES_FILE


if __name__ == "__main__":
    build_features_for_new_games()