# ============================================================
# File: src/features/feature_engineering.py
# Purpose: Generate features for NBA games with rolling stats + home/away flag + sanity checks + CSV append
# ============================================================

import pandas as pd
import logging
import os
from datetime import datetime

logger = logging.getLogger("features.feature_engineering")


def generate_features_for_games(
    df: pd.DataFrame, stats_out="data/results/feature_stats.csv"
) -> pd.DataFrame:
    """
    Generate features for NBA games.
    Handles TEAM_NAME/TEAM_ABBREVIATION gracefully, adds rolling stats, home/away flag,
    logs sanity checks, and appends distributions to CSV for historical tracking.
    """

    # --- Target column ---
    if "WL" in df.columns:
        df["win"] = df["WL"].apply(lambda x: 1 if str(x).upper() == "W" else 0)
        logger.info("Using WL column to generate win target.")
    elif "win" in df.columns:
        logger.info("Win column already present.")
    else:
        logger.error("No WL or win column found. Cannot generate target.")
        return pd.DataFrame()

    # --- Team identifier ---
    team_col = None
    if "TEAM_NAME" in df.columns:
        team_col = "TEAM_NAME"
    elif "TEAM_ABBREVIATION" in df.columns:
        team_col = "TEAM_ABBREVIATION"

    # --- Base features ---
    features = pd.DataFrame()
    features["GAME_ID"] = df["GAME_ID"]
    if team_col:
        features["TEAM"] = df[team_col]
    if "MATCHUP" in df.columns:
        features["MATCHUP"] = df["MATCHUP"]
        # Home/away flag: '@' means away, otherwise home
        features["home_game"] = df["MATCHUP"].apply(lambda x: 0 if "@" in str(x) else 1)
        # Log distribution
        home_counts = features["home_game"].value_counts().to_dict()
        logger.info("Home/Away distribution: %s", home_counts)
    if "GAME_DATE" in df.columns:
        features["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    features["win"] = df["win"]

    # --- Sanity check: wins vs losses ---
    win_counts = features["win"].value_counts().to_dict()
    logger.info("Win/Loss distribution: %s", win_counts)

    # --- Export sanity checks (append mode) ---
    stats = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metric": "home_game",
                "home": home_counts.get(1, 0),
                "away": home_counts.get(0, 0),
            },
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metric": "win",
                "wins": win_counts.get(1, 0),
                "losses": win_counts.get(0, 0),
            },
        ]
    )

    os.makedirs(os.path.dirname(stats_out), exist_ok=True)
    if os.path.exists(stats_out):
        stats.to_csv(stats_out, mode="a", header=False, index=False)
    else:
        stats.to_csv(stats_out, index=False)
    logger.info("Feature sanity stats appended to %s", stats_out)

    # --- Rolling stats ---
    if "PTS" in df.columns:
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])
        df["rolling_pts"] = df.groupby("TEAM_ID")["PTS"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        features["rolling_pts"] = df["rolling_pts"]

    if "PTS_OPP" in df.columns:
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])
        df["rolling_pts_allowed"] = df.groupby("TEAM_ID")["PTS_OPP"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        features["rolling_pts_allowed"] = df["rolling_pts_allowed"]

    # Win streak feature
    df["win_streak"] = df.groupby("TEAM_ID")["win"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).sum()
    )
    features["win_streak"] = df["win_streak"]

    logger.info("Generated features with shape %s", features.shape)
    return features
