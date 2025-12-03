# ============================================================
# File: scripts/build_features_for_new_games.py
# Purpose: Build features for upcoming games, including team averages and odds
# ============================================================

import pandas as pd
from pathlib import Path
from core.config import NEW_GAMES_FILE, PLAYER_STATS_FILE, NEW_GAMES_FEATURES_FILE
from core.log_config import setup_logger
from core.exceptions import DataError, PipelineError

logger = setup_logger("build_features_for_new_games")


def compute_team_averages(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team-level averages from player stats.
    Returns a DataFrame with team averages for pts, ast, reb, games played.
    """
    required_cols = ["team", "pts", "ast", "reb", "games_played"]
    missing = [c for c in required_cols if c not in player_stats.columns]
    if missing:
        raise DataError(f"Player stats missing required columns: {missing}")

    team_avgs = (
        player_stats.groupby("team")[["pts", "ast", "reb", "games_played"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "pts": "avg_pts",
                "ast": "avg_ast",
                "reb": "avg_reb",
                "games_played": "avg_games_played",
            }
        )
    )
    return team_avgs


def main():
    # Load new games
    if not Path(NEW_GAMES_FILE).exists():
        raise FileNotFoundError(f"{NEW_GAMES_FILE} not found.")
    new_games = pd.read_csv(NEW_GAMES_FILE)
    if new_games.empty:
        raise DataError("No new games found.")

    # Load player stats
    if not Path(PLAYER_STATS_FILE).exists():
        raise FileNotFoundError(f"{PLAYER_STATS_FILE} not found.")
    player_stats = pd.read_csv(PLAYER_STATS_FILE)
    if player_stats.empty:
        raise DataError("Player stats file is empty.")

    # Compute team averages
    team_avgs = compute_team_averages(player_stats)

    # Merge averages into new games (home + away)
    features = (
        new_games.merge(team_avgs, left_on="home_team", right_on="team", how="left")
        .rename(
            columns={
                "avg_pts": "home_avg_pts",
                "avg_ast": "home_avg_ast",
                "avg_reb": "home_avg_reb",
                "avg_games_played": "home_avg_games_played",
            }
        )
        .drop(columns=["team"])
    )

    features = (
        features.merge(team_avgs, left_on="away_team", right_on="team", how="left")
        .rename(
            columns={
                "avg_pts": "away_avg_pts",
                "avg_ast": "away_avg_ast",
                "avg_reb": "away_avg_reb",
                "avg_games_played": "away_avg_games_played",
            }
        )
        .drop(columns=["team"])
    )

    # If odds are included in NEW_GAMES_FILE, keep them
    if "decimal_odds" not in features.columns and "decimal_odds" in new_games.columns:
        features["decimal_odds"] = new_games["decimal_odds"]

    # Save features
    try:
        features.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
        logger.info(f"✅ Features saved to {NEW_GAMES_FEATURES_FILE} ({len(features)} rows)")
    except Exception as e:
        raise PipelineError(f"Failed to save features: {e}")


if __name__ == "__main__":
    main()