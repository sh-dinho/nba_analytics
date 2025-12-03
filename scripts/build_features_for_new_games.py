# ============================================================
# File: scripts/build_features_for_new_games.py
# Purpose: Build features for today's games (prediction path)
# ============================================================

import os
import pandas as pd
from core.config import TRAINING_FEATURES_FILE, NEW_GAMES_FILE
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("build_features")


def build_features_for_new_games(new_games_file: str = NEW_GAMES_FILE) -> str:
    """
    Build game-level features for today's games.
    Aggregates player stats into team averages, merges home vs away,
    and carries forward decimal_odds for EV calculations.
    """
    if not os.path.exists(new_games_file):
        raise FileNotFoundError(f"{new_games_file} not found.")

    df = pd.read_csv(new_games_file)

    # Ensure required columns exist
    ensure_columns(df, {
        "PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_HOME", "TEAM_AWAY",
        "PTS", "AST", "REB", "GAMES_PLAYED", "decimal_odds"
    }, "new game features")

    # Aggregate player stats into team-level
    team_features = df.groupby("TEAM_ABBREVIATION").agg({
        "PTS": "mean", "AST": "mean", "REB": "mean", "GAMES_PLAYED": "mean"
    }).reset_index().rename(columns={
        "TEAM_ABBREVIATION": "team",
        "PTS": "avg_pts", "AST": "avg_ast", "REB": "avg_reb", "GAMES_PLAYED": "avg_games_played"
    })

    # Merge into game-level rows
    games = df[["TEAM_HOME", "TEAM_AWAY", "decimal_odds"]].drop_duplicates()
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
            # ✅ carry forward decimal_odds
            "decimal_odds": row["decimal_odds"]
        })
    features_df = pd.DataFrame(merged)

    os.makedirs(os.path.dirname(TRAINING_FEATURES_FILE), exist_ok=True)
    features_df.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Features built for new games ({len(features_df)} rows) → {TRAINING_FEATURES_FILE}")

    return TRAINING_FEATURES_FILE


if __name__ == "__main__":
    build_features_for_new_games()