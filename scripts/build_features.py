# ============================================================
# File: scripts/build_features.py
# Purpose: Generate training and prediction features
# Author: <your name / org>
# Last Updated: 2025-02-21
#
# Notes:
# - Produces team-level and player-level features
# - Supports rolling averages or season averages
# - Aligned with CI/CD pipeline (fetch → build_features → train)
# ============================================================

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core.paths import (
    HISTORICAL_GAMES_FILE,
    NEW_GAMES_FILE,
    TRAINING_FEATURES_FILE,
    PLAYER_FEATURES_FILE,
    NEW_GAMES_FEATURES_FILE,
    ensure_dirs,
)
from core.config import USE_ROLLING_AVG, ROLLING_WINDOW, log_config_snapshot
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError

logger = init_global_logger()


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

COLUMN_MAP = {
    "home": "home_team",
    "away": "away_team",
    "homescore": "home_pts",
    "awayscore": "away_pts",
    "date": "date",
    "gameid": "game_id",
    "pts_home": "home_pts",
    "pts_away": "away_pts",
    "teamhome": "home_team",
    "teamaway": "away_team",
    "home_points": "home_pts",
    "away_points": "away_pts",
    "ast": "ast",
    "reb": "reb",
    "pts": "pts",
    "games_played": "games_played",
    "homewin": "homewin",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and apply known mappings."""
    original_cols = df.columns.tolist()
    df.columns = [col.lower().replace(" ", "").replace("_", "") for col in df.columns]

    normalized_map = {
        k.lower().replace(" ", "").replace("_", ""): v for k, v in COLUMN_MAP.items()
    }

    applied_renames = {col: normalized_map[col] for col in df.columns if col in normalized_map}
    df = df.rename(columns=applied_renames)

    logger.info(f"Column mapping applied: {applied_renames}")

    unmapped = [c for c in df.columns if c not in applied_renames.values()]
    if unmapped:
        logger.warning(f"⚠️ Unmapped columns: {unmapped}")

    return df


# ============================================================
# TEAM FEATURES
# ============================================================

def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create team-level features for training data."""
    required = ["home_team", "away_team"]
    for col in required:
        if col not in df.columns:
            raise DataError(f"Missing required column: {col}")

    df = df.copy()

    # Order by date for correct rolling averages
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    feats = pd.DataFrame({
        "game_id": df.get("game_id", df.index),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
    })

    stat_pairs = [
        ("home_pts", "home_avg_pts", "home_team"),
        ("away_pts", "away_avg_pts", "away_team"),
        ("home_reb", "home_avg_reb", "home_team"),
        ("away_reb", "away_avg_reb", "away_team"),
        ("home_ast", "home_avg_ast", "home_team"),
        ("away_ast", "away_avg_ast", "away_team"),
    ]

    logger.info(
        f"Building team features using {'rolling window '+str(ROLLING_WINDOW) if USE_ROLLING_AVG else 'season averages'}"
    )

    for raw_col, out_col, team_col in stat_pairs:
        if raw_col in df.columns:
            if USE_ROLLING_AVG:
                feats[out_col] = (
                    df.groupby(team_col)[raw_col]
                    .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
                )
            else:
                feats[out_col] = df.groupby(team_col)[raw_col].transform("mean")

    # Labels
    if "homewin" in df.columns:
        feats["label"] = df["homewin"].astype(int)
    elif "home_pts" in df.columns and "away_pts" in df.columns:
        feats["label"] = (df["home_pts"] > df["away_pts"]).astype(int)
    else:
        raise DataError("Cannot compute label: need homewin or score columns.")

    # Margin-based targets
    if "home_pts" in df.columns and "away_pts" in df.columns:
        feats["margin"] = (df["home_pts"] - df["away_pts"])
        feats["outcome_category"] = feats["margin"].apply(
            lambda m: (
                "home_blowout" if m >= 10 else
                "home_close" if m > 0 else
                "away_close" if m > -10 else
                "away_blowout"
            )
        )

    logger.info(f"Team features built: {feats.shape[0]} rows, {feats.shape[1]} columns")
    return feats


# ============================================================
# PLAYER FEATURES
# ============================================================

def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build player-level rolling or season averages."""
    if "player_name" not in df.columns:
        logger.info("No player data found. Skipping player feature generation.")
        return pd.DataFrame()

    df = df.copy()

    required = ["pts", "ast", "reb"]
    for col in required:
        if col not in df.columns:
            raise DataError(f"Missing player column: {col}")

    if USE_ROLLING_AVG:
        logger.info(f"Computing player rolling averages (window={ROLLING_WINDOW})")

        df = df.sort_values("games_played") if "games_played" in df.columns else df

        roll = (
            df.groupby("player_name")[["pts", "ast", "reb"]]
            .rolling(ROLLING_WINDOW, min_periods=1)
            .mean()
            .reset_index()
        )
        df["player_avg_pts"] = roll["pts"]
        df["player_avg_ast"] = roll["ast"]
        df["player_avg_reb"] = roll["reb"]

    else:
        logger.info("Computing player season averages")
        df["player_avg_pts"] = df.groupby("player_name")["pts"].transform("mean")
        df["player_avg_ast"] = df.groupby("player_name")["ast"].transform("mean")
        df["player_avg_reb"] = df.groupby("player_name")["reb"].transform("mean")

    keep_cols = [
        "player_name",
        "team_abbreviation" if "team_abbreviation" in df.columns else "team",
        "games_played" if "games_played" in df.columns else None,
        "player_avg_pts",
        "player_avg_ast",
        "player_avg_reb",
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in df.columns]

    player_features = df[keep_cols].drop_duplicates("player_name")

    logger.info(f"Player features built: {len(player_features)} players")
    return player_features


# ============================================================
# NEW GAME FEATURES (PREDICTION MODE)
# ============================================================

def build_new_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create team-only features for new upcoming games."""
    return pd.DataFrame({
        "game_id": df.get("game_id", df.index),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
    })


# ============================================================
# MAIN BUILDER
# ============================================================

def build_features(training: bool, player: bool):
    ensure_dirs(strict=False)
    log_config_snapshot()

    if training:
        # -----------------------------
        # Load historical training data
        # -----------------------------
        if not HISTORICAL_GAMES_FILE.exists():
            raise FileError("Historical games file not found", file_path=str(HISTORICAL_GAMES_FILE))

        df = pd.read_csv(HISTORICAL_GAMES_FILE)
        df = normalize_columns(df)

        # TEAM FEATURES
        team_features = build_team_features(df)
        team_features.to_csv(TRAINING_FEATURES_FILE, index=False)
        logger.info(f"Saved training team features → {TRAINING_FEATURES_FILE}")

        # PLAYER FEATURES
        if player:
            player_features = build_player_features(df)
            player_features.to_csv(PLAYER_FEATURES_FILE, index=False)
            logger.info(f"Saved player features → {PLAYER_FEATURES_FILE}")

    else:
        # -----------------------------
        # Build inference features
        # -----------------------------
        if not NEW_GAMES_FILE.exists():
            raise FileError("New games file not found", file_path=str(NEW_GAMES_FILE))

        df = pd.read_csv(NEW_GAMES_FILE)
        df = normalize_columns(df)

        features = build_new_game_features(df)
        features.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
        logger.info(f"Saved new game features → {NEW_GAMES_FEATURES_FILE}")


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build NBA training & inference features")
    parser.add_argument("--training", action="store_true", help="Build training features")
    parser.add_argument("--player", action="store_true", help="Build player features")
    args = parser.parse_args()

    build_features(training=args.training, player=args.player)


if __name__ == "__main__":
    main()
