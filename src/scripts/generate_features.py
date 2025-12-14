# ============================================================
# File: src/scripts/generate_features.py
# Purpose: Generate enriched features from historical NBA schedule
# Version: 1.1 (schema consistency, rolling alignment, numeric casting, clearer logging)
# ============================================================

import logging
import os
import pandas as pd
from src.schemas import FEATURE_COLUMNS, normalize_features

logger = logging.getLogger("scripts.generate_features")
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Starting feature generation from historical NBA games...")

    input_file_parquet = "data/cache/historical_schedule.parquet"
    input_file_csv = "data/cache/schedule.csv"
    output_file = "data/cache/features_full.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = None

    # Try parquet first
    if os.path.exists(input_file_parquet):
        logger.info("Loading schedule from %s", input_file_parquet)
        try:
            df = pd.read_parquet(input_file_parquet)
        except Exception as e:
            logger.error("Failed to read parquet file: %s", e)
            df = None
    elif os.path.exists(input_file_csv):
        logger.info("Loading schedule from %s", input_file_csv)
        try:
            df = pd.read_csv(input_file_csv)
        except Exception as e:
            logger.error("Failed to read CSV file: %s", e)
            df = None
    else:
        logger.warning("No schedule file found. Writing empty features file.")
        pd.DataFrame(columns=FEATURE_COLUMNS).to_parquet(output_file, index=False)
        return

    if df is None or df.empty:
        logger.warning("Schedule data is empty. Writing empty features file.")
        pd.DataFrame(columns=FEATURE_COLUMNS).to_parquet(output_file, index=False)
        return

    # Build features
    logger.info("Building features from schedule data (rows: %d)", len(df))

    features = pd.DataFrame()

    # GAME_ID
    features["GAME_ID"] = (
        df["GAME_ID"] if "GAME_ID" in df.columns else pd.Series(dtype="int64")
    )

    # SEASON
    if "SEASON" in df.columns:
        features["SEASON"] = df["SEASON"]
    elif "SEASON_ID" in df.columns:
        features["SEASON"] = df["SEASON_ID"]
    else:
        features["SEASON"] = pd.Series(dtype="int64")

    # Points scored/allowed
    features["PTS"] = df["PTS"] if "PTS" in df.columns else pd.Series(dtype="float64")
    features["PTS_OPP"] = (
        df["PTS_OPP"] if "PTS_OPP" in df.columns else pd.Series(dtype="float64")
    )

    # WIN target (numeric) + WL
    if "WIN" in df.columns:
        features["WIN"] = df["WIN"]
        features["WL"] = df["WL"] if "WL" in df.columns else pd.Series(dtype="object")
    elif "WL" in df.columns:
        features["WIN"] = df["WL"].apply(lambda x: 1 if str(x).upper() == "W" else 0)
        features["WL"] = df["WL"]
    else:
        features["WIN"] = pd.Series(dtype="int64")
        features["WL"] = pd.Series(dtype="object")

    # Rolling averages and streaks
    if {"TEAM_ID", "GAME_DATE"}.issubset(df.columns):
        df_sorted = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

        if "PTS" in df_sorted.columns:
            df_sorted["AVG_PTS_LAST3"] = df_sorted.groupby("TEAM_ID")["PTS"].transform(
                lambda x: x.shift().rolling(3, min_periods=1).mean()
            )
        else:
            df_sorted["AVG_PTS_LAST3"] = pd.Series(dtype="float64")

        if "PTS_OPP" in df_sorted.columns:
            df_sorted["AVG_PTS_ALLOWED_LAST3"] = df_sorted.groupby("TEAM_ID")[
                "PTS_OPP"
            ].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
        else:
            df_sorted["AVG_PTS_ALLOWED_LAST3"] = pd.Series(dtype="float64")

        if "WL" in df_sorted.columns:
            df_sorted["win_flag"] = df_sorted["WL"].apply(
                lambda x: 1 if str(x).upper() == "W" else 0
            )
            df_sorted["WIN_STREAK"] = df_sorted.groupby("TEAM_ID")[
                "win_flag"
            ].transform(lambda x: x.shift().rolling(5, min_periods=1).sum())
        else:
            df_sorted["WIN_STREAK"] = pd.Series(dtype="float64")

        # Align back to original index
        features = features.join(
            df_sorted[["AVG_PTS_LAST3", "AVG_PTS_ALLOWED_LAST3", "WIN_STREAK"]]
        )
    else:
        missing_cols = {"TEAM_ID", "GAME_DATE"} - set(df.columns)
        logger.warning(
            "Missing columns for rolling stats: %s. Skipping rolling stats and win streaks.",
            missing_cols,
        )
        features["AVG_PTS_LAST3"] = pd.Series(dtype="float64")
        features["AVG_PTS_ALLOWED_LAST3"] = pd.Series(dtype="float64")
        features["WIN_STREAK"] = pd.Series(dtype="float64")

    # Cast numeric features to float
    numeric_cols = [
        "PTS",
        "PTS_OPP",
        "WIN",
        "AVG_PTS_LAST3",
        "AVG_PTS_ALLOWED_LAST3",
        "WIN_STREAK",
    ]
    for col in numeric_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    # Normalize and save
    features = normalize_features(features)
    try:
        features.to_parquet(output_file, index=False)
        logger.info("Features saved to %s (rows: %d)", output_file, len(features))
    except Exception as e:
        logger.error("Failed to save features: %s", e)


if __name__ == "__main__":
    main()
