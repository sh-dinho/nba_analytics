# ============================================================
# File: scripts/build_features.py
# Purpose: Build training features from player stats and game results
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from core.config import TRAINING_FEATURES_FILE, PLAYER_STATS_FILE, GAME_RESULTS_FILE
from core.log_config import setup_logger
from core.exceptions import DataError
from core.utils import ensure_columns

logger = setup_logger("build_features")


def _prepare_labels(results: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare labels for merging.
    - If results have PLAYER_NAME + home_win, return labels keyed by PLAYER_NAME.
    - Else if results have home_team, away_team, home_win, return team-level labels keyed by TEAM_ABBREVIATION.
      Creates a long format with two rows per game:
        - home_team with team_win = home_win
        - away_team with team_win = 1 - home_win
    """
    # Player-level labels
    if {"PLAYER_NAME", "home_win"}.issubset(results.columns):
        if results["home_win"].isnull().all():
            raise DataError("Labels present but all home_win are null in results.")
        labels = results[["PLAYER_NAME", "home_win"]].copy()
        labels = labels.rename(columns={"home_win": "label"})
        labels["merge_key"] = "player"
        return labels

    # Team-level labels
    required = {"home_team", "away_team", "home_win"}
    if required.issubset(results.columns):
        if results["home_win"].isnull().all():
            raise DataError("Labels present but all home_win are null in results.")

        # Build long format: two rows per game
        home = results[["home_team", "home_win"]].copy()
        home = home.rename(columns={"home_team": "TEAM_ABBREVIATION"})
        home["team_win"] = home["home_win"]

        away = results[["away_team", "home_win"]].copy()
        away = away.rename(columns={"away_team": "TEAM_ABBREVIATION"})
        away["team_win"] = 1 - away["home_win"]

        labels = pd.concat([home[["TEAM_ABBREVIATION", "team_win"]],
                            away[["TEAM_ABBREVIATION", "team_win"]]], ignore_index=True)
        labels = labels.rename(columns={"team_win": "label"})
        labels["merge_key"] = "team"
        return labels

    raise DataError(
        "Results must contain either PLAYER_NAME + home_win or home_team + away_team + home_win."
    )


def build_features() -> pd.DataFrame:
    """Build training features with robust label merging."""
    TRAINING_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not PLAYER_STATS_FILE.exists():
        raise FileNotFoundError(f"{PLAYER_STATS_FILE} not found. Run fetch_player_stats.py first.")

    # Load player stats
    df = pd.read_csv(PLAYER_STATS_FILE)

    # Validate input schema (include TEAM_ABBREVIATION so team-level fallback works)
    required_input_cols = {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "GAMES_PLAYED"}
    missing = required_input_cols - set(df.columns)
    if missing:
        raise DataError(f"Missing columns in player stats: {missing}")

    # Engineered features
    df["PTS_per_AST"] = np.where(df["AST"] > 0, df["PTS"] / df["AST"], 0.0)
    df["REB_rate"] = np.where(df["GAMES_PLAYED"] > 0, df["REB"] / df["GAMES_PLAYED"], 0.0)

    # Labels
    if GAME_RESULTS_FILE.exists():
        results = pd.read_csv(GAME_RESULTS_FILE)
        try:
            labels = _prepare_labels(results)
        except Exception as e:
            logger.error(f"❌ Failed to prepare labels from results: {e}")
            raise

        # Merge labels
        if labels["merge_key"].iloc[0] == "player":
            df = df.merge(labels[["PLAYER_NAME", "label"]], on="PLAYER_NAME", how="left")
        else:  # team-level
            df = df.merge(labels[["TEAM_ABBREVIATION", "label"]], on="TEAM_ABBREVIATION", how="left")

        if "label" not in df.columns or df["label"].isnull().all():
            logger.warning("⚠️ Labels not successfully merged. Training data may be incomplete.")
    else:
        raise FileNotFoundError(f"Missing ACTUAL game results file: {GAME_RESULTS_FILE}")

    # Final validation
    ensure_columns(df, {"PLAYER_NAME", "PTS", "AST", "REB", "label"}, "training features")

    df.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Features saved to {TRAINING_FEATURES_FILE} ({len(df)} rows)")
    return df


def build_features_for_new_games(new_games_file) -> pd.DataFrame:
    """Build features for new games (no labels)."""
    new_games_file = Path(new_games_file)
    if not new_games_file.exists():
        raise FileNotFoundError(f"{new_games_file} not found. Run fetch_new_games.py first.")

    df = pd.read_csv(new_games_file)

    # Validate necessary columns for feature engineering
    ensure_columns(df, {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "GAMES_PLAYED"}, "new game features")

    # Features
    df["PTS_per_AST"] = np.where(df["AST"] > 0, df["PTS"] / df["AST"], 0.0)
    df["REB_rate"] = np.where(df["GAMES_PLAYED"] > 0, df["REB"] / df["GAMES_PLAYED"], 0.0)

    logger.info(f"✅ Features built for new games ({len(df)} rows)")
    return df


# Alias for pipeline compatibility
main = build_features