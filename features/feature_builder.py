# ============================================================
# File: features/feature_builder.py
# Purpose: Build feature sets for training and prediction
# ============================================================

import pandas as pd
import numpy as np
from nba_core.paths import (
    HISTORICAL_GAMES_FILE,
    PLAYER_GAMES_FILE,
    TRAINING_FEATURES_FILE,
    NEW_GAMES_FILE,
    NEW_GAMES_FEATURES_FILE,
    PLAYER_FEATURES_FILE,
)
from nba_core.log_config import init_global_logger

logger = init_global_logger("feature_builder")


def _team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build team-level features and target column."""
    # Target: whether team won (1) or lost (0)
    df["label"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    # Point differential
    if {"PTS", "PLUS_MINUS"} <= set(df.columns):
        df["point_diff"] = df["PTS"] + df["PLUS_MINUS"]

    # Rolling averages (last 5 games)
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["avg_points_last5"] = (
        df.groupby("TEAM_ID")["PTS"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    df["avg_point_diff_last5"] = (
        df.groupby("TEAM_ID")["point_diff"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    return df


def _player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build player-level features."""
    df["efficiency"] = df["PTS"] + df["REB"] + df["AST"]

    df["fg_pct"] = np.where(df["FGA"] > 0, df["FGM"] / df["FGA"], 0.0)
    df["ft_pct"] = np.where(df["FTA"] > 0, df["FTM"] / df["FTA"], 0.0)
    df["three_pct"] = np.where(df["FG3A"] > 0, df["FG3M"] / df["FG3A"], 0.0)

    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    for col in ["PTS", "REB", "AST", "efficiency"]:
        df[f"{col}_last5"] = df.groupby("PLAYER_ID")[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    return df


def build_features(training: bool = True, player: bool = True) -> None:
    """Build features for training or prediction."""
    if training:
        logger.info("🔧 Building training features...")
        if not HISTORICAL_GAMES_FILE.exists():
            logger.warning(f"⚠️ Historical games file missing: {HISTORICAL_GAMES_FILE}")
            return

        df = pd.read_csv(HISTORICAL_GAMES_FILE)
        df = _team_features(df)

        if player and PLAYER_GAMES_FILE.exists():
            player_df = pd.read_csv(PLAYER_GAMES_FILE)
            player_df = _player_features(player_df)
            player_df.to_csv(PLAYER_FEATURES_FILE, index=False)
            logger.info(f"🗂️ Player features saved → {PLAYER_FEATURES_FILE}")

        df.to_csv(TRAINING_FEATURES_FILE, index=False)
        logger.info(f"🗂️ Training features saved → {TRAINING_FEATURES_FILE}")

    else:
        logger.info("🔧 Building prediction features...")
        if not NEW_GAMES_FILE.exists():
            logger.warning(f"⚠️ Upcoming games file missing: {NEW_GAMES_FILE}")
            return

        df = pd.read_csv(NEW_GAMES_FILE)
        df = _team_features(df)

        if player and PLAYER_GAMES_FILE.exists():
            player_df = pd.read_csv(PLAYER_GAMES_FILE)
            player_df = _player_features(player_df)
            team_eff = player_df.groupby("TEAM_ID")["efficiency"].mean().reset_index()
            df = df.merge(team_eff, on="TEAM_ID", how="left")

        df.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
        logger.info(f"🗂️ Prediction features saved → {NEW_GAMES_FEATURES_FILE}")
