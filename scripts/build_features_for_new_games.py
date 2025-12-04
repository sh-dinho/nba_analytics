# ============================================================
# File: scripts/build_features_for_new_games.py
# Purpose: Build features for upcoming games, including rolling team averages, odds, and OU lines
# ============================================================

import argparse
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.plot_combined_feature_trends import plot_combined_feature_trends

from core.paths import (
    NEW_GAMES_FILE,
    PLAYER_STATS_FILE,
    NEW_GAMES_FEATURES_FILE,
    LOGS_DIR,
    FEATURES_LOG_FILE,
    ensure_dirs,
)
from core.config import ROLLING_WINDOW, log_config_snapshot
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError, PipelineError

logger = init_global_logger()

# Dedicated log for new games features
NEW_GAMES_FEATURES_LOG = LOGS_DIR / "new_games_features_summary.csv"


def _require_columns(df: pd.DataFrame, required: list[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{name} missing required columns: {missing}")


def compute_team_rolling_averages(player_stats: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling team averages from player stats."""
    required_cols = ["team", "game_date", "pts", "ast", "reb", "games_played"]
    _require_columns(player_stats, required_cols, "Player stats")

    player_stats = player_stats.copy()
    player_stats["game_date"] = pd.to_datetime(player_stats["game_date"], errors="coerce")
    player_stats = player_stats.sort_values(["team", "game_date"])

    team_daily = (
        player_stats.groupby(["team", "game_date"], as_index=False)[["pts", "ast", "reb", "games_played"]]
        .sum()
    )

    team_avgs = (
        team_daily.groupby("team")[["pts", "ast", "reb", "games_played"]]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index()
    )

    latest = (
        team_avgs.groupby("team", as_index=False)
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
    return latest[["team", "avg_pts", "avg_ast", "avg_reb", "avg_games_played"]]


def log_feature_summary(summary_entry: pd.DataFrame):
    """Append summary entry to both new games log and unified features log."""
    try:
        if NEW_GAMES_FEATURES_LOG.exists():
            summary_entry.to_csv(NEW_GAMES_FEATURES_LOG, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(NEW_GAMES_FEATURES_LOG, index=False)
        logger.info(f"📈 New game features summary appended to {NEW_GAMES_FEATURES_LOG}")
    except Exception as e:
        logger.warning(f"Failed to append new game features summary: {e}")

    try:
        if FEATURES_LOG_FILE.exists():
            summary_entry.to_csv(FEATURES_LOG_FILE, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(FEATURES_LOG_FILE, index=False)
        logger.info(f"📈 Unified features summary appended to {FEATURES_LOG_FILE}")
    except Exception as e:
        logger.warning(f"Failed to append unified features summary: {e}")


def build_features_for_new_games(window: int = ROLLING_WINDOW) -> str:
    ensure_dirs(strict=False)
    log_config_snapshot()  # record configuration state

    if not Path(NEW_GAMES_FILE).exists():
        raise FileError("New games file not found", file_path=str(NEW_GAMES_FILE))
    new_games = pd.read_csv(NEW_GAMES_FILE)
    if new_games.empty:
        raise DataError("No new games found.")

    team_map = {
        "home": "home_team",
        "away": "away_team",
        "teamhome": "home_team",
        "teamaway": "away_team",
        "home_team": "home_team",
        "away_team": "away_team",
    }
    normalized_cols = {c: team_map.get(c.lower(), c) for c in new_games.columns}
    new_games = new_games.rename(columns=normalized_cols)

    _require_columns(new_games, ["home_team", "away_team"], "New games")

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

    for col in ["decimal_odds", "ou_line", "moneyline_home", "moneyline_away"]:
        if col in new_games.columns and col not in features.columns:
            features[col] = new_games[col]

    avg_cols = [
        "home_avg_pts", "home_avg_ast", "home_avg_reb", "home_avg_games_played",
        "away_avg_pts", "away_avg_ast", "away_avg_reb", "away_avg_games_played",
    ]
    for c in avg_cols:
        if c in features.columns:
            features[c] = features[c].fillna(features[c].mean())

    try:
        features.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
        logger.info(f"✅ New game features saved to {NEW_GAMES_FEATURES_FILE} ({len(features)} rows)")
    except Exception as e:
        raise PipelineError(f"Failed to save features: {e}")

    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "window": window,
        "rows": len(features),
        "has_odds": int("decimal_odds" in features.columns or "moneyline_home" in features.columns),
        "has_ou_line": int("ou_line" in features.columns),
    }])

    log_feature_summary(summary_entry)
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


def plot_feature_trends():
    """Plot trends of new game features over time."""
    if not NEW_GAMES_FEATURES_LOG.exists():
        logger.warning("No new games features log found.")
        return ""
    df = pd.read_csv(NEW_GAMES_FEATURES_LOG)
    if df.empty:
        logger.warning("Features log is empty.")
        return ""

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["rows"], marker="o", label="Rows")
    ax.plot(df["timestamp"], df["window"], marker="x", label="Rolling Window")
    ax.set_title("New Game Features Trends")
    ax.set_xlabel("Run Timestamp")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    trend_path = LOGS_DIR / "new_games_features_trends.png"
    plt.tight_layout()
    plt.savefig(trend_path)
    plt.close()
    logger.info(f"📊 Feature trends saved → {trend_path}")
    return str(trend_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for upcoming games")
    parser.add_argument("--window", type=int, default=ROLLING_WINDOW,
                        help="Rolling window size for team averages")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print the latest summary log entry without regenerating features")
    parser.add_argument("--plot-trends", action="store_true",
                        help="Plot new game feature trends over time")
    parser.add_argument("--plot-combined-trends", action="store_true",
                        help="Plot combined training + new games trends")
    args = parser.parse_args()

    if args.summary_only:
        print_latest_summary()
    elif args.plot_trends:
        plot_feature_trends()
    elif args.plot_combined_trends:
        plot_combined_feature_trends()
    else:
        logger.info(f"🛠️ Building new game features with rolling window={args.window}")
        build_features_for_new_games(window=args.window)
