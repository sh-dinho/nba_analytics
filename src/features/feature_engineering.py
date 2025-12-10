# ============================================================
# File: src/features/feature_engineering.py
# Purpose: Generate NBA game features for prediction
# Project: nba_analysis
# Version: 1.5 (adds point spread, over/under, top scorers, rolling features)
# ============================================================

import pandas as pd
import logging
import datetime

def generate_features_for_games(game_data_list):
    """
    Generate features for a list of games.
    Ensures GAME_ID and TEAM_ID columns are always present.
    Filters out incomplete rows where GAME_ID or TEAM_ID is missing.
    Adds rolling team features, betting features, and player-level features.
    Returns a DataFrame with the full expected schema.
    """
    features = []

    for i, game in enumerate(game_data_list):
        # Normalize keys to uppercase for consistency
        normalized = {k.upper(): v for k, v in game.items()}
        df = pd.DataFrame([normalized])

        # Print the columns for debugging
        print(f"Columns in game {i}: {df.columns.tolist()}")

        # Check if required columns exist
        if 'GAME_ID' not in df.columns or 'TEAM_ID' not in df.columns:
            logging.warning(f"Missing GAME_ID or TEAM_ID at index {i}. Skipping this game.")
            continue  # Skip this game if essential columns are missing

        # Ensure GAME_ID and TEAM_ID are present
        if pd.isna(df["GAME_ID"].iloc[0]) or pd.isna(df["TEAM_ID"].iloc[0]):
            logging.warning(f"Skipping game at index {i} due to missing GAME_ID or TEAM_ID.")
            continue  # Skip this game

        # Guarantee GAME_ID and TEAM_ID are valid
        if "GAME_ID" not in df.columns:
            logging.warning(f"Missing GAME_ID at index {i}, assigning placeholder.")
            df["GAME_ID"] = f"unknown_game_{i}"

        if "TEAM_ID" not in df.columns:
            logging.warning(f"Missing TEAM_ID at index {i}, assigning placeholder.")
            df["TEAM_ID"] = -1

        # Add prediction_date and unique_id
        df["prediction_date"] = datetime.date.today().isoformat()
        df["unique_id"] = (
            df["GAME_ID"].astype(str)
            + "_"
            + df["TEAM_ID"].astype(str)
            + "_"
            + df["prediction_date"].astype(str)
        )

        # Initialize rolling/betting/player features with defaults
        df["RollingPTS_5"] = 0
        df["RollingWinPct_10"] = 0
        df["RestDays"] = 0
        df["TeamWinPctToDate"] = 0
        df["OppWinPctToDate"] = 0
        df["PointSpread"] = 0          # home minus away
        df["OverUnder"] = 0
        df["Players20Pts"] = 0         # number of players scoring 20+

        features.append(df)

    valid_features = [df for df in features if not df.empty]
    if not valid_features:
        logging.warning("No valid features generated.")
        expected_cols = [
            "GAME_ID", "TEAM_ID", "unique_id", "prediction_date",
            "RollingPTS_5", "RollingWinPct_10", "RestDays",
            "TeamWinPctToDate", "OppWinPctToDate",
            "PointSpread", "OverUnder", "Players20Pts"
        ]
        return pd.DataFrame(columns=expected_cols)

    full_df = pd.concat(valid_features, ignore_index=True)

    # --- Rolling team features ---
    if "DATE" in full_df.columns:
        full_df["DATE"] = pd.to_datetime(full_df["DATE"], errors="coerce")
        full_df = full_df.sort_values(["TEAM_ID", "DATE"])

        if "POINTS" in full_df.columns:
            full_df["RollingPTS_5"] = full_df.groupby("TEAM_ID")["POINTS"].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

        if "TARGET" in full_df.columns:
            full_df["RollingWinPct_10"] = full_df.groupby("TEAM_ID")["TARGET"].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )

        full_df["RestDays"] = full_df.groupby("TEAM_ID")["DATE"].diff().dt.days.fillna(0)

        if "TARGET" in full_df.columns:
            full_df["CumulativeWins"] = full_df.groupby("TEAM_ID")["TARGET"].cumsum()
            full_df["CumulativeGames"] = full_df.groupby("TEAM_ID").cumcount() + 1
            full_df["TeamWinPctToDate"] = (full_df["CumulativeWins"] / full_df["CumulativeGames"]).fillna(0)

        # Opponent win pct
        if "OPPONENT_TEAM_ID" in full_df.columns:
            opp_df = full_df[["TEAM_ID", "DATE", "TeamWinPctToDate"]].rename(
                columns={"TEAM_ID": "OPPONENT_TEAM_ID", "TeamWinPctToDate": "OppWinPctToDate"}
            )
            full_df = pd.merge_asof(
                full_df.sort_values("DATE"),
                opp_df.sort_values("DATE"),
                by="OPPONENT_TEAM_ID",
                on="DATE",
                direction="backward"
            )

    # --- Betting and player features ---
    if "HOME_SCORE" in full_df.columns and "AWAY_SCORE" in full_df.columns:
        full_df["PointSpread"] = full_df["HOME_SCORE"] - full_df["AWAY_SCORE"]

    if "OVER_UNDER" in full_df.columns:
        full_df["OverUnder"] = full_df["OVER_UNDER"]

    if "PLAYER_POINTS" in full_df.columns:
        # Count players scoring 20+ points
        def count_20pts(pp):
            if isinstance(pp, dict):
                return sum(1 for p in pp.values() if p >= 20)
            elif isinstance(pp, list):
                return sum(1 for p in pp if p >= 20)
            return 0
        full_df["Players20Pts"] = full_df["PLAYER_POINTS"].apply(count_20pts)

    return full_df.reset_index(drop=True)
