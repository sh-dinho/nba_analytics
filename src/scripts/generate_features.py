# ============================================================
# File: src/scripts/generate_features.py
# Purpose: Generate enriched features from historical NBA schedule
# ============================================================

import logging
import os
import pandas as pd

logger = logging.getLogger("scripts.generate_features")
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Starting feature generation from historical NBA games...")

    input_file_parquet = "data/cache/historical_schedule.parquet"
    input_file_csv = "data/cache/schedule.csv"
    output_file = "data/cache/features_full.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = None

    # --- Try parquet first ---
    if os.path.exists(input_file_parquet):
        logger.info("Loading schedule from %s", input_file_parquet)
        df = pd.read_parquet(input_file_parquet)
    elif os.path.exists(input_file_csv):
        logger.info("Loading schedule from %s", input_file_csv)
        df = pd.read_csv(input_file_csv)
    else:
        logger.warning("No schedule file found. Writing empty features file.")
        empty_cols = ["GAME_ID", "SEASON", "PTS", "PTS_OPP", "WL"]
        pd.DataFrame(columns=empty_cols).to_parquet(output_file, index=False)
        return

    if df is None or df.empty:
        logger.warning("Schedule data is empty. Writing empty features file.")
        empty_cols = ["GAME_ID", "SEASON", "PTS", "PTS_OPP", "WL"]
        pd.DataFrame(columns=empty_cols).to_parquet(output_file, index=False)
        return

    # --- Build features ---
    logger.info("Building features from schedule data (rows: %d)", len(df))

    features = pd.DataFrame()
    features["GAME_ID"] = df.get("GAME_ID", pd.Series(dtype="int"))
    features["SEASON"] = df.get("SEASON_ID", pd.Series(dtype="int"))

    # Points scored/allowed if available
    if "PTS" in df.columns:
        features["PTS"] = df["PTS"]
    if "PTS_OPP" in df.columns:
        features["PTS_OPP"] = df["PTS_OPP"]

    # Win/Loss flag
    if "WL" in df.columns:
        features["WL"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    # Rolling averages (last 3 games per team)
    if "TEAM_ID" in df.columns and "PTS" in df.columns:
        features["AVG_PTS_LAST3"] = df.groupby("TEAM_ID")["PTS"].transform(
            lambda x: x.shift().rolling(3, min_periods=1).mean()
        )
    if "TEAM_ID" in df.columns and "PTS_OPP" in df.columns:
        features["AVG_PTS_ALLOWED_LAST3"] = df.groupby("TEAM_ID")["PTS_OPP"].transform(
            lambda x: x.shift().rolling(3, min_periods=1).mean()
        )

    # Win streaks
    if "TEAM_ID" in df.columns and "WL" in df.columns:
        features["WIN_STREAK"] = df.groupby("TEAM_ID")["WL"].transform(
            lambda x: x.eq("W").astype(int).groupby(x.ne("W").cumsum()).cumsum()
        )

    # --- Save features ---
    features.to_parquet(output_file, index=False)
    logger.info("Features saved to %s (rows: %d)", output_file, len(features))


if __name__ == "__main__":
    main()
