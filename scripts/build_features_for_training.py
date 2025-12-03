# ============================================================
# File: scripts/build_features_for_training.py
# Purpose: Build training features from historical games with labels
# ============================================================

import os
import pandas as pd
from core.config import TRAINING_FEATURES_FILE, HISTORICAL_GAMES_FILE
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("build_features_training")


def build_features_for_training(historical_file: str = HISTORICAL_GAMES_FILE) -> str:
    """
    Build game-level features from historical games.
    Aggregates player stats into team averages, merges home vs away,
    and adds a 'label' column (1 if home team won, else 0).
    Saves to TRAINING_FEATURES_FILE and returns path.
    """
    if not os.path.exists(historical_file):
        raise FileNotFoundError(f"{historical_file} not found. Provide historical games data.")

    df = pd.read_csv(historical_file)

    # ✅ Do NOT require decimal_odds for training
    ensure_columns(df, {
        "PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_HOME", "TEAM_AWAY",
        "PTS", "AST", "REB", "GAMES_PLAYED", "HOME_WIN"
    }, "historical game features")

    # Aggregate player stats into team-level
    team_features = df.groupby("TEAM_ABBREVIATION").agg({
        "PTS": "mean", "AST": "mean", "REB": "mean", "GAMES_PLAYED": "mean"
    }).reset_index().rename(columns={
        "TEAM_ABBREVIATION": "team",
        "PTS": "avg_pts", "AST": "avg_ast", "REB": "avg_reb", "GAMES_PLAYED": "avg_games_played"
    })

    # Merge into game-level rows
    games = df[["TEAM_HOME", "TEAM_AWAY", "HOME_WIN"]].drop_duplicates()
    merged = []
    for _, row in games.iterrows():
        home = team_features[team_features["team"] == row["TEAM_HOME"]].iloc[0]
        away = team_features[team_features["team"] == row["TEAM_AWAY"]].iloc[0]
        merged.append({
            "game_id": f"{row['TEAM_HOME']}_vs_{row['TEAM_AWAY']}",
            "home_team": row["TEAM_HOME"], "away_team": row["TEAM_AWAY"],
            "home_avg_pts": home["avg_pts"], "home_avg_ast": home["avg_ast"],
            "home_avg_reb": home["avg_reb"], "home_avg_games_played": home["avg_games_played"],
            "away_avg_pts": away["avg_pts"], "away_avg_ast": away["avg_ast"],
            "away_avg_reb": away["avg_reb"], "away_avg_games_played": away["avg_games_played"],
            "label": 1 if row["HOME_WIN"] == 1 else 0
        })
    features_df = pd.DataFrame(merged)

    os.makedirs(os.path.dirname(TRAINING_FEATURES_FILE), exist_ok=True)
    features_df.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Training features built ({len(features_df)} rows) → {TRAINING_FEATURES_FILE}")

    return TRAINING_FEATURES_FILE


if __name__ == "__main__":
    build_features_for_training()