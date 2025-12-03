# ============================================================
# File: scripts/build_features_for_new_games.py
# Purpose: Build team-level features for new games
# ============================================================

import pandas as pd
import os
from core.config import NEW_GAMES_FILE, NEW_GAMES_FEATURES_FILE
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("build_features_for_new_games")


def build_features_for_new_games(new_games_file: str = NEW_GAMES_FILE) -> str:
    """
    Build team-level features for new games from player-level stats.
    Saves to NEW_GAMES_FEATURES_FILE and returns path.
    """
    if not os.path.exists(new_games_file):
        raise FileNotFoundError(f"{new_games_file} not found. Run fetch_new_games first.")

    df = pd.read_csv(new_games_file)

    # Ensure required columns exist
    required = ["PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_HOME", "TEAM_AWAY",
                "PTS", "AST", "REB", "GAMES_PLAYED"]
    if "decimal_odds" in df.columns:
        required.append("decimal_odds")
    ensure_columns(df, required, "new games")

    # Aggregate team-level averages
    features = []
    games = df[["TEAM_HOME", "TEAM_AWAY"]].drop_duplicates()

    for _, row in games.iterrows():
        home_team = row["TEAM_HOME"]
        away_team = row["TEAM_AWAY"]

        home_players = df[df["TEAM_ABBREVIATION"] == home_team]
        away_players = df[df["TEAM_ABBREVIATION"] == away_team]

        game_features = {
            "game_id": f"{home_team}_vs_{away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "home_avg_pts": home_players["PTS"].mean(),
            "home_avg_ast": home_players["AST"].mean(),
            "home_avg_reb": home_players["REB"].mean(),
            "home_avg_games_played": home_players["GAMES_PLAYED"].mean(),
            "away_avg_pts": away_players["PTS"].mean(),
            "away_avg_ast": away_players["AST"].mean(),
            "away_avg_reb": away_players["REB"].mean(),
            "away_avg_games_played": away_players["GAMES_PLAYED"].mean(),
        }

        # Carry forward odds if available
        if "decimal_odds" in df.columns:
            game_features["decimal_odds"] = df.loc[df["TEAM_HOME"] == home_team, "decimal_odds"].iloc[0]

        features.append(game_features)

    features_df = pd.DataFrame(features)

    # ✅ Save to NEW_GAMES_FEATURES_FILE
    features_df.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
    logger.info(f"✅ Features built for new games ({len(features_df)} rows) → {NEW_GAMES_FEATURES_FILE}")

    return str(NEW_GAMES_FEATURES_FILE)


if __name__ == "__main__":
    build_features_for_new_games()