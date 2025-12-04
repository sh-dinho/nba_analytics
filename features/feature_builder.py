# ============================================================
# File: features/feature_builder.py
# Purpose: Build training features from player stats
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from core.log_config import init_global_logger
from core.exceptions import DataError, FileError
from core.paths import TRAINING_FEATURES_FILE
from core.training_cache import archive_training
from core.player_stats_cache import load_player_stats

logger = init_global_logger()

def build_features(season: str = "2024-25", out_file: Path = TRAINING_FEATURES_FILE):
    # Load player stats (from cache/file)
    stats_df = load_player_stats()

    # Normalize columns expected from nba_api to lowercase aliases for feature aggregation
    # nba_api columns typically: TEAM_ABBREVIATION, PTS, AST, REB
    if not {"TEAM_ABBREVIATION", "PTS", "AST", "REB"}.issubset(set(stats_df.columns)):
        raise DataError("Missing required player stat columns for feature building", dataset="player_stats.csv")

    stats_df = stats_df.rename(columns={
        "TEAM_ABBREVIATION": "team",
        "PTS": "pts",
        "AST": "ast",
        "REB": "reb"
    })

    team_stats = stats_df.groupby("team").agg({"pts": "mean", "ast": "mean", "reb": "mean"}).reset_index()
    team_stats.columns = ["team", "avg_pts", "avg_ast", "avg_reb"]

    games = []
    teams = team_stats["team"].unique()
    np.random.shuffle(teams)

    for i in range(0, len(teams), 2):
        if i + 1 >= len(teams):
            break
        home_team, away_team = teams[i], teams[i + 1]
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
            "home_win": 1 if np.random.rand() < 0.55 else 0,
        }
        games.append(features)

    features_df = pd.DataFrame(games)

    # Validate required columns
    expected = {"game_id", "home_team", "away_team", "home_win"}
    missing = expected - set(features_df.columns)
    if missing:
        raise DataError(f"Features missing required columns: {missing}", dataset="training_features")

    # Archive before saving and persist
    try:
        archive_training()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(out_file, index=False)
        logger.info(f"✅ Built features for {len(features_df)} games → {out_file}")
    except Exception as e:
        raise FileError(f"Failed to save training features: {out_file}", file_path=str(out_file)) from e

    return features_df
