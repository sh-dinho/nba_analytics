# ============================================================
# File: scripts/build_features.py
# Purpose: Build training or prediction features
# ============================================================

import argparse
import os
import pandas as pd
from core.config import HISTORICAL_GAMES_FILE, NEW_GAMES_FILE, BASE_DATA_DIR, ensure_dirs
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
    "pts_away": "away_pts"
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase and map known variations."""
    df.columns = [col.strip().lower().replace(" ", "").replace("_", "") for col in df.columns]
    normalized_map = {k.lower().replace(" ", "").replace("_", ""): v for k, v in COLUMN_MAP.items()}
    df = df.rename(columns={col: normalized_map[col] for col in df.columns if col in normalized_map})

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

        # Build features with per-team rolling averages
        features = pd.DataFrame({
            "game_id": df.index,
            "home_team": df["home_team"],
            "away_team": df["away_team"],
        })

        if "home_pts" in df:
            features["home_avg_pts"] = df.groupby("home_team")["home_pts"].transform(lambda x: x.rolling(rounds, min_periods=1).mean())
        if "away_pts" in df:
            features["away_avg_pts"] = df.groupby("away_team")["away_pts"].transform(lambda x: x.rolling(rounds, min_periods=1).mean())
        if "home_reb" in df:
            features["home_avg_reb"] = df.groupby("home_team")["home_reb"].transform(lambda x: x.rolling(rounds, min_periods=1).mean())
        if "away_reb" in df:
            features["away_avg_reb"] = df.groupby("away_team")["away_reb"].transform(lambda x: x.rolling(rounds, min_periods=1).mean())
        if "home_ast" in df:
            features["home_avg_ast"] = df.groupby("home_team")["home_ast"].transform(lambda x: x.rolling(rounds, min_periods=1).mean())
        if "away_ast" in df:
            features["away_avg_ast"] = df.groupby("away_team")["away_ast"].transform(lambda x: x.rolling(rounds, min_periods=1).mean())

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

        out_file = os.path.join(BASE_DATA_DIR, "training_features.csv")
        features.to_csv(out_file, index=False)
        logger.info(f"✅ Training features saved to {out_file} ({len(features)} rows)")

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
            "game_id": df.index,
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