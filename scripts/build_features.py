# ============================================================
# File: scripts/build_features.py
# Purpose: Build training or prediction features
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
    LOGS_DIR,
    ensure_dirs,
)
from core.config import USE_ROLLING_AVG, ROLLING_WINDOW, log_config_snapshot
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError

logger = init_global_logger()


COLUMN_MAP = {
    "Home": "home_team",
    "Away": "away_team",
    "HomeScore": "home_pts",
    "AwayScore": "away_pts",
    "Date": "date",
    "GameID": "game_id",
    "teamhome": "home_team",
    "teamaway": "away_team",
    "home_points": "home_pts",
    "away_points": "away_pts",
    "pts_home": "home_pts",
    "pts_away": "away_pts",
    "TEAM_HOME": "home_team",
    "TEAM_AWAY": "away_team",
    "PTS": "pts",
    "AST": "ast",
    "REB": "reb",
    "GAMES_PLAYED": "games_played",
    "HOME_WIN": "homewin",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase and map known variations."""
    df.columns = [col.strip().lower().replace(" ", "").replace("_", "") for col in df.columns]
    normalized_map = {k.lower().replace(" ", "").replace("_", ""): v for k, v in COLUMN_MAP.items()}
    applied = {col: normalized_map[col] for col in df.columns if col in normalized_map}
    df = df.rename(columns=applied)

    logger.info(f"Column mapping applied: {applied}")
    unmapped = [col for col in df.columns if col not in normalized_map.values()]
    if unmapped:
        logger.warning(f"Unmapped columns: {unmapped}")

    logger.info(f"Normalized columns: {list(df.columns)}")
    return df


def build_features(rounds: int = 10, training: bool = False, player: bool = False):
    """Build features for training or prediction (team and/or player)."""
    ensure_dirs(strict=False)
    log_config_snapshot() 

    if training:
        logger.info("Loading historical games...")
        if not HISTORICAL_GAMES_FILE.exists():
            raise FileError("Historical games file not found", file_path=str(HISTORICAL_GAMES_FILE))

        df = pd.read_csv(HISTORICAL_GAMES_FILE)
        df = normalize_columns(df)

        required = ["home_team", "away_team"]
        for col in required:
            if col not in df.columns:
                raise DataError(f"Missing required column: {col}")

        # Sort by date for rolling averages
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")

        # --- TEAM FEATURES ---
        if not player:  # combined mode builds both team + player
            features = pd.DataFrame({
                "game_id": df["game_id"] if "game_id" in df.columns else df.index,
                "home_team": df["home_team"],
                "away_team": df["away_team"],
            })

            if USE_ROLLING_AVG:
                rounds = ROLLING_WINDOW
                logger.info(f"Using rolling averages (window={rounds}) for team features")
                for col in ["home_pts", "away_pts", "home_reb", "away_reb", "home_ast", "away_ast"]:
                    if col in df.columns:
                        team_col = "home_team" if col.startswith("home_") else "away_team"
                        features[f"{col.replace('home_', 'home_avg_').replace('away_', 'away_avg_')}"] = (
                            df.groupby(team_col)[col].transform(lambda x: x.rolling(rounds, min_periods=1).mean())
                        )
            else:
                logger.info("Using season averages for team features")
                for col in ["home_pts", "away_pts", "home_reb", "away_reb", "home_ast", "away_ast"]:
                    if col in df.columns:
                        team_col = "home_team" if col.startswith("home_") else "away_team"
                        features[f"{col.replace('home_', 'home_avg_').replace('away_', 'away_avg_')}"] = (
                            df.groupby(team_col)[col].transform("mean")
                        )

            if "homewin" in df.columns:
                features["label"] = df["homewin"].astype(int)
            elif "home_pts" in df.columns and "away_pts" in df.columns:
                features["label"] = (df["home_pts"] > df["away_pts"]).astype(int)
            else:
                raise DataError("No valid label source found (homewin or points).")

            if "home_pts" in df.columns and "away_pts" in df.columns:
                features["margin"] = (df["home_pts"] - df["away_pts"]).astype(int)
                features["outcome_category"] = features["margin"].apply(
                    lambda m: "home_blowout" if m >= 10 else
                              "home_close" if m > 0 else
                              "away_close" if m > -10 else
                              "away_blowout"
                )

            features.to_csv(TRAINING_FEATURES_FILE, index=False)
            logger.info(f"✅ Training features saved to {TRAINING_FEATURES_FILE} ({len(features)} rows)")

        # --- PLAYER FEATURES ---
        if player or not player:  # combined mode builds player features too
            if "player_name" in df.columns:
                if USE_ROLLING_AVG:
                    logger.info(f"Computing player-level rolling averages (window={rounds})")
                    df = df.sort_values("games_played") if "games_played" in df.columns else df
                    player_roll = (
                        df.groupby("player_name")[["pts", "ast", "reb"]]
                        .rolling(rounds, min_periods=1)
                        .mean()
                        .reset_index()
                    )
                    df["player_avg_pts"] = player_roll["pts"]
                    df["player_avg_ast"] = player_roll["ast"]
                    df["player_avg_reb"] = player_roll["reb"]
                else:
                    logger.info("Computing player-level season averages")
                    df["player_avg_pts"] = df.groupby("player_name")["pts"].transform("mean")
                    df["player_avg_ast"] = df.groupby("player_name")["ast"].transform("mean")
                    df["player_avg_reb"] = df.groupby("player_name")["reb"].transform("mean")

                player_features = df[["player_name", "team_abbreviation", "games_played",
                                      "player_avg_pts", "player_avg_ast", "player_avg_reb"]]
                player_features.to_csv(PLAYER_FEATURES_FILE, index=False)
                logger.info(f"✅ Player features saved to {PLAYER_FEATURES_FILE} ({len(player_features)} rows)")

    else:
        logger.info("Loading new games...")
        if not NEW_GAMES_FILE.exists():
            raise FileError("New games file not found", file_path=str(NEW_GAMES_FILE))

        df = pd.read_csv(NEW_GAMES_FILE)
        df = normalize_columns(df)

        required = ["home_team", "away_team"]
        for col in required:
            if col not in df.columns:
                raise DataError(f"Missing required column: {col}")

        features = pd.DataFrame({
            "game_id": df["game_id"] if "game_id" in df.columns else df.index,
            "home_team": df["home_team"],
            "away_team": df["away_team"],
        })

        features.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
        logger.info(f"✅ New game features saved to {NEW_GAMES_FEATURES_FILE} ({len(features)} rows)")


def plot_feature_trends(file: Path, feature_cols: list[str]):
    """Plot trends of selected features over time."""
    if not file.exists():
        logger.warning(f"⚠️ Feature file not found: {file}")
        return ""
    df = pd.read_csv(file)
    if df.empty:
        logger.warning("Feature file is empty.")
        return ""

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in feature_cols:
            if col in df.columns:
                ax.plot(df["date"], df[col], label=col)
        ax.set_title("Feature Trends Over Time")
        ax.legend()