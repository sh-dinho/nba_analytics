# ============================================================
# File: scripts/build_features_for_training.py
# Purpose: Build training features from historical games
#          Outputs both team-level and player-level features
# ============================================================

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import core.config as config

from core.paths import (
    HISTORICAL_GAMES_FILE,
    TRAINING_FEATURES_FILE,
    DATA_DIR,
    LOGS_DIR,
    FEATURES_LOG_FILE,
    ensure_dirs,
)
from core.config import USE_ROLLING_AVG, ROLLING_WINDOW, log_config_snapshot
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError

logger = init_global_logger()

TRAINING_FEATURES_LOG = LOGS_DIR / "training_features_summary.csv"
PLAYER_FEATURES_FILE = DATA_DIR / "player_features.csv"


def log_feature_summary(summary_entry: pd.DataFrame):
    """Append summary entry to both training log and unified features log."""
    try:
        if TRAINING_FEATURES_LOG.exists():
            summary_entry.to_csv(TRAINING_FEATURES_LOG, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(TRAINING_FEATURES_LOG, index=False)
        logger.info(f"📈 Training features summary appended to {TRAINING_FEATURES_LOG}")
    except Exception as e:
        logger.warning(f"Failed to append training features summary: {e}")

    try:
        if FEATURES_LOG_FILE.exists():
            summary_entry.to_csv(FEATURES_LOG_FILE, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(FEATURES_LOG_FILE, index=False)
        logger.info(f"📈 Unified features summary appended to {FEATURES_LOG_FILE}")
    except Exception as e:
        logger.warning(f"Failed to append unified features summary: {e}")


def build_features_for_training() -> str:
    ensure_dirs(strict=False)
    log_config_snapshot()  # record configuration state

    logger.info("Loading historical games...")
    if not HISTORICAL_GAMES_FILE.exists():
        raise FileError("Historical games file not found", file_path=str(HISTORICAL_GAMES_FILE))

    df = pd.read_csv(HISTORICAL_GAMES_FILE)
    if df.empty:
        raise DataError("Historical games file is empty.")

    logger.info(f"Detected columns: {list(df.columns)}")

    # Normalize column names
    rename_map = {
        "TEAM_HOME": "home_team",
        "TEAM_AWAY": "away_team",
        "PTS": "pts",
        "AST": "ast",
        "REB": "reb",
        "GAMES_PLAYED": "games_played",
        "HOME_WIN": "home_win",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["home_team", "away_team", "pts", "ast", "reb", "games_played", "home_win"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"Historical games file missing required columns: {missing}")

    # --- Team-level aggregation ---
    team_totals = (
        df.groupby(["home_team", "away_team"])[["pts", "ast", "reb", "games_played"]]
        .sum()
        .reset_index()
    )

    # Labels
    team_totals["label"] = df.groupby(["home_team", "away_team"])["home_win"].max().values

    # Split into home/away stats
    home_stats = team_totals.rename(
        columns={
            "pts": "home_pts",
            "ast": "home_ast",
            "reb": "home_reb",
            "games_played": "home_games_played",
        }
    )
    away_stats = team_totals.rename(
        columns={
            "pts": "away_pts",
            "ast": "away_ast",
            "reb": "away_reb",
            "games_played": "away_games_played",
        }
    )

    # Merge home and away stats
    features = pd.merge(
        home_stats[["home_team", "away_team", "home_pts", "home_ast", "home_reb", "home_games_played", "label"]],
        away_stats[["home_team", "away_team", "away_pts", "away_ast", "away_reb", "away_games_played"]],
        on=["home_team", "away_team"],
    )

    # Margin of victory
    features["margin"] = features["home_pts"] - features["away_pts"]

    # Outcome category
    def categorize(row):
        if row["margin"] >= 10 and row["label"] == 1:
            return "home_blowout"
        elif row["margin"] < 10 and row["label"] == 1:
            return "home_close"
        elif row["margin"] <= -10 and row["label"] == 0:
            return "away_blowout"
        else:
            return "away_close"

    features["outcome_category"] = features.apply(categorize, axis=1)

    # Rolling vs season averages for team stats
    if USE_ROLLING_AVG:
        logger.info(f"Using rolling averages (window={ROLLING_WINDOW}) for team features")
        for col in ["home_pts", "home_ast", "home_reb"]:
            features[f"home_avg_{col.split('_')[1]}"] = (
                features.groupby("home_team")[col].transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
            )
        for col in ["away_pts", "away_ast", "away_reb"]:
            features[f"away_avg_{col.split('_')[1]}"] = (
                features.groupby("away_team")[col].transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
            )
    else:
        logger.info("Using season averages for team features")
        for col in ["home_pts", "home_ast", "home_reb"]:
            features[f"home_avg_{col.split('_')[1]}"] = features.groupby("home_team")[col].transform("mean")
        for col in ["away_pts", "away_ast", "away_reb"]:
            features[f"away_avg_{col.split('_')[1]}"] = features.groupby("away_team")[col].transform("mean")

    # Save team features
    features.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Team training features built ({len(features)} rows) → {TRAINING_FEATURES_FILE}")

    # --- Player-level rolling/season averages ---
    if USE_ROLLING_AVG:
        logger.info(f"Computing player-level rolling averages (window={ROLLING_WINDOW})")
        df = df.sort_values("games_played")
        player_roll = (
            df.groupby("PLAYER_NAME")[["pts", "ast", "reb"]]
            .rolling(ROLLING_WINDOW, min_periods=1)
            .mean()
            .reset_index()
        )
        df["player_avg_pts"] = player_roll["pts"]
        df["player_avg_ast"] = player_roll["ast"]
        df["player_avg_reb"] = player_roll["reb"]
    else:
        logger.info("Computing player-level season averages")
        df["player_avg_pts"] = df.groupby("PLAYER_NAME")["pts"].transform("mean")
        df["player_avg_ast"] = df.groupby("PLAYER_NAME")["ast"].transform("mean")
        df["player_avg_reb"] = df.groupby("PLAYER_NAME")["reb"].transform("mean")

    # Save player features
    player_features = df[["PLAYER_NAME", "TEAM_ABBREVIATION", "games_played",
                          "player_avg_pts", "player_avg_ast", "player_avg_reb"]]
    player_features.to_csv(PLAYER_FEATURES_FILE, index=False)
    logger.info(f"✅ Player features built ({len(player_features)} rows) → {PLAYER_FEATURES_FILE}")

    # Append summary log
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "rolling" if USE_ROLLING_AVG else "season"
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "mode": mode,
        "window": ROLLING_WINDOW if USE_ROLLING_AVG else None,
        "team_rows": len(features),
        "player_rows": len(player_features),
    }])

    log_feature_summary(summary_entry)
    return str(TRAINING_FEATURES_FILE)


def plot_training_feature_trends():
    """Plot trends of training feature builds over time."""
    if not TRAINING_FEATURES_LOG.exists():
        logger.warning("No training features log found.")
        return ""
    df = pd.read_csv(TRAINING_FEATURES_LOG)
    if df.empty:
        logger.warning("Training features log is empty.")
        return ""

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["team_rows"], marker="o", label="Team Rows")
    ax.plot(df["timestamp"], df["player_rows"], marker="x", label="Player Rows")
    ax.set_title("Training Features Trends")
    ax.set_xlabel("Run Timestamp")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

        trend_path = LOGS_DIR / "training_features_trends.png"
    plt.tight_layout()
    plt.savefig(trend_path)
    plt.close()
    logger.info(f"📊 Training feature trends saved → {trend_path}")
    return str(trend_path)



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Build training features from historical games")
    parser.add_argument("--rolling", action="store_true", help="Use rolling averages (last N games)")
    parser.add_argument("--season", action="store_true", help="Use season averages")
    parser.add_argument("--window", type=int, default=None, help="Rolling window size (default from config)")
    parser.add_argument("--plot-trends", action="store_true", help="Plot training feature trends over time")
    parser.add_argument("--plot-combined-trends", action="store_true", help="Plot combined training + new games trends")
    args = parser.parse_args()

    if args.rolling:
        config.USE_ROLLING_AVG = True
    if args.season:
        config.USE_ROLLING_AVG = False
    if args.window is not None:
        config.ROLLING_WINDOW = args.window

    mode = "rolling" if config.USE_ROLLING_AVG else "season"
    logger.info(f"🛠️ CLI override: using {mode} averages (window={config.ROLLING_WINDOW if config.USE_ROLLING_AVG else 'N/A'})")

    if args.plot_trends:
        plot_training_feature_trends()
    elif args.plot_combined_trends:
        plot_combined_feature_trends()
    else:
        build_features_for_training()
