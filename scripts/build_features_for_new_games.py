# ============================================================
# File: scripts/build_features_for_new_games.py
# Purpose: Build features for upcoming games, including rolling team averages, odds, and OU lines
# ============================================================

import argparse
import pandas as pd
import datetime
from pathlib import Path

from core.paths import NEW_GAMES_FILE, PLAYER_STATS_FILE, DATA_DIR, LOGS_DIR, ensure_dirs
from core.config import NEW_GAMES_FEATURES_FILE, ROLLING_WINDOW
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError, PipelineError

logger = init_global_logger()

NEW_GAMES_FEATURES_LOG = LOGS_DIR / "new_games_features.log"


def compute_team_rolling_averages(player_stats: pd.DataFrame, window: int) -> pd.DataFrame:
    required_cols = ["team", "game_date", "pts", "ast", "reb", "games_played"]
    missing = [c for c in required_cols if c not in player_stats.columns]
    if missing:
        raise DataError(f"Player stats missing required columns: {missing}")

    player_stats["game_date"] = pd.to_datetime(player_stats["game_date"])
    player_stats = player_stats.sort_values(["team", "game_date"])

    team_avgs = (
        player_stats.groupby("team")[["pts", "ast", "reb", "games_played"]]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index()
    )

    latest_avgs = (
        team_avgs.groupby("team")
        .tail(1)
        .rename(
            columns={
                "pts": "avg_pts",
                "ast": "avg_ast",
                "reb": "avg_reb",
                "games_played": "avg_games_played",
            }
        )
    )
    return latest_avgs


def build_features_for_new_games(window: int = ROLLING_WINDOW) -> str:
    ensure_dirs(strict=False)

    if not Path(NEW_GAMES_FILE).exists():
        raise FileError("New games file not found", file_path=str(NEW_GAMES_FILE))
    new_games = pd.read_csv(NEW_GAMES_FILE)
    if new_games.empty:
        raise DataError("No new games found.")

    if not Path(PLAYER_STATS_FILE).exists():
        raise FileError("Player stats file not found", file_path=str(PLAYER_STATS_FILE))
    player_stats = pd.read_csv(PLAYER_STATS_FILE)
    if player_stats.empty:
        raise DataError("Player stats file is empty.")

    team_avgs = compute_team_rolling_averages(player_stats, window)

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

    if "decimal_odds" in new_games.columns and "decimal_odds" not in features.columns:
        features["decimal_odds"] = new_games["decimal_odds"]

    if "ou_line" in new_games.columns and "ou_line" not in features.columns:
        features["ou_line"] = new_games["ou_line"]

    try:
        features.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
        logger.info(f"✅ New game features saved to {NEW_GAMES_FEATURES_FILE} ({len(features)} rows)")
    except Exception as e:
        raise PipelineError(f"Failed to save features: {e}")

    # Append summary log
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "window": window,
        "rows": len(features),
    }])
    try:
        if NEW_GAMES_FEATURES_LOG.exists():
            summary_entry.to_csv(NEW_GAMES_FEATURES_LOG, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(NEW_GAMES_FEATURES_LOG, index=False)
        logger.info(f"📈 New game features summary appended to {NEW_GAMES_FEATURES_LOG}")
    except Exception as e:
        logger.warning(f"Failed to append new game features summary: {e}")

    return str(NEW_GAMES_FEATURES_FILE)


def print_latest_summary():
    """Print the latest summary entry without regenerating features."""
    if not NEW_GAMES_FEATURES_LOG.exists():
        logger.error("No summary log found.")
        return
    try:
        df = pd.read_csv(NEW_GAMES_FEATURES_LOG)
        if df.empty:
            logger.warning("Summary log is empty.")
            return
        latest = df.tail(1).iloc[0].to_dict()
        logger.info(f"📊 Latest summary: {latest}")
    except Exception as e:
        logger.error(f"Failed to read summary log: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for upcoming games")
    parser.add_argument("--window", type=int, default=ROLLING_WINDOW,
                        help="Rolling window size for team averages")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print the latest summary log entry without regenerating features")
    args = parser.parse_args()

    if args.summary_only:
        print_latest_summary()
    else:
        logger.info(f"🛠️ Building new game features with rolling window={args.window}")
        build_features_for_new_games(window=args.window)
