# ============================================================
# File: scripts/build_features.py
# Purpose: Build training or prediction features
# ============================================================

import argparse
import os
import pandas as pd
from core.config import (
    HISTORICAL_GAMES_FILE,
    NEW_GAMES_FILE,
    BASE_DATA_DIR,
    ensure_dirs,
    USE_ROLLING_AVG,
    ROLLING_WINDOW,
)
from core.log_config import setup_logger

logger = setup_logger("build_features")

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


def build_features(rounds=10, training=False):
    ensure_dirs()

    if training:
        logger.info("Loading historical games...")
        try:
            df = pd.read_csv(HISTORICAL_GAMES_FILE)
        except FileNotFoundError:
            logger.error(f"File not found: {HISTORICAL_GAMES_FILE}")
            return

        df = normalize_columns(df)

        required = ["home_team", "away_team"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # --- TEAM FEATURES ---
        features = pd.DataFrame({
            "game_id": df["game_id"] if "game_id" in df.columns else df.index,
            "home_team": df["home_team"],
            "away_team": df["away_team"],
        })

        # Rolling vs season averages
        if USE_ROLLING_AVG:
            rounds = ROLLING_WINDOW
            logger.info(f"Using rolling averages (window={rounds}) for team features")
            for col in ["home_pts", "away_pts", "home_reb", "away_reb", "home_ast", "away_ast"]:
                if col in df:
                    features[f"{col.replace('home_', 'home_avg_').replace('away_', 'away_avg_')}"] = (
                        df.groupby(col.split('_')[0] + "_team")[col].transform(lambda x: x.rolling(rounds, min_periods=1).mean())
                    )
        else:
            logger.info("Using season averages for team features")
            for col in ["home_pts", "away_pts", "home_reb", "away_reb", "home_ast", "away_ast"]:
                if col in df:
                    features[f"{col.replace('home_', 'home_avg_').replace('away_', 'away_avg_')}"] = (
                        df.groupby(col.split('_')[0] + "_team")[col].transform("mean")
                    )

        # Labels
        if "homewin" in df.columns:
            features["label"] = df["homewin"].astype(int)
        elif "home_pts" in df and "away_pts" in df:
            features["label"] = (df["home_pts"] > df["away_pts"]).astype(int)
        else:
            raise ValueError("No valid label source found (homewin or points).")

        # Margin and outcome category
        if "home_pts" in df and "away_pts" in df:
            features["margin"] = (df["home_pts"] - df["away_pts"]).astype(int)
            features["outcome_category"] = features["margin"].apply(
                lambda m: "home_blowout" if m >= 10 else
                          "home_close" if m > 0 else
                          "away_close" if m > -10 else
                          "away_blowout"
            )

        # Diagnostics
        label_counts = features["label"].value_counts()
        logger.info(f"📊 Label distribution: {label_counts.to_dict()}")
        if features["label"].nunique() < 2:
            logger.warning("⚠️ Only one class present in labels. Training will fail unless data includes both outcomes.")
        if "margin" in features:
            logger.info(f"📊 Margin stats: min={features['margin'].min()}, max={features['margin'].max()}, mean={features['margin'].mean():.2f}")

        out_file = os.path.join(BASE_DATA_DIR, "training_features.csv")
        features.to_csv(out_file, index=False)
        logger.info(f"✅ Training features saved to {out_file} ({len(features)} rows)")

        # --- PLAYER FEATURES ---
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

            player_out_file = os.path.join(BASE_DATA_DIR, "player_features.csv")
            player_features = df[["player_name", "team_abbreviation", "games_played",
                                  "player_avg_pts", "player_avg_ast", "player_avg_reb"]]
            player_features.to_csv(player_out_file, index=False)
            logger.info(f"✅ Player features saved to {player_out_file} ({len(player_features)} rows)")

    else:
        logger.info("Loading new games...")
        try:
            df = pd.read_csv(NEW_GAMES_FILE)
        except FileNotFoundError:
            logger.error(f"File not found: {NEW_GAMES_FILE}")
            return

        df = normalize_columns(df)

        required = ["home_team", "away_team"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        features = pd.DataFrame({
            "game_id": df["game_id"] if "game_id" in df.columns else df.index,
            "home_team": df["home_team"],
            "away_team": df["away_team"],
        })

        out_file = os.path.join(BASE_DATA_DIR, "new_games_features.csv")
        features.to_csv(out_file, index=False)
        logger.info(f"✅ New game features saved to {out_file} ({len(features)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for training or prediction")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()

    build_features(rounds=args.rounds, training=args.training)