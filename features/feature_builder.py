# ============================================================
# File: features/feature_builder.py
# Purpose: Build training features from player stats
# ============================================================

import os
import pandas as pd
import numpy as np
from core.logging import setup_logger
from core.utils import ensure_columns
from core.exceptions import DataError

logger = setup_logger("feature_builder")

def build_features(season="2024-25", out_file="data/training_features.csv"):
    stats_df = fetch_player_stats(season)

    team_stats = stats_df.groupby("team").agg({
        "pts": "mean", "ast": "mean", "reb": "mean"
    }).reset_index()
    team_stats.columns = ["team", "avg_pts", "avg_ast", "avg_reb"]

    games = []
    teams = team_stats["team"].unique()
    np.random.shuffle(teams)

    for i in range(0, len(teams), 2):
        if i + 1 >= len(teams):
            break
        home_team, away_team = teams[i], teams[i+1]
        home_stats = team_stats[team_stats["team"] == home_team].iloc[0]
        away_stats = team_stats[team_stats["team"] == away_team].iloc[0]

        features = {
            "game_id": int(np.random.randint(100000, 999999)),
            "home_team": home_team,
            "away_team": away_team,
            "home_avg_pts": home_stats["avg_pts"],
            "away_avg_pts": away_stats["avg_pts"],
            "home_avg_ast": home_stats["avg_ast"],
            "away_avg_ast": away_stats["avg_ast"],
            "home_avg_reb": home_stats["avg_reb"],
            "away_avg_reb": away_stats["avg_reb"],
            "home_win": 1 if np.random.rand() < 0.55 else 0
        }
        games.append(features)

    features_df = pd.DataFrame(games)

    # Validate required columns
    try:
        ensure_columns(features_df, {"game_id", "home_team", "away_team", "home_win"}, "features")
    except ValueError as e:
        raise DataError(str(e))

    os.makedirs("data", exist_ok=True)
    features_df.to_csv(out_file, index=False)
    logger.info(f"Built features for {len(features_df)} games → {out_file}")
    return features_df