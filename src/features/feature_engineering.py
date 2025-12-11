# ============================================================
# File: src/features/feature_engineering.py
# Purpose: Generate NBA game features for prediction
# Project: nba_analysis
# Version: 2.0 (merged game_features + feature_engineering)
# ============================================================

import datetime
import logging
import pandas as pd
import numpy as np


def generate_features_for_games(game_data_list, players_min_points: int = 20):
    """
    Generate features for a list of games.
    Ensures GAME_ID and TEAM_ID columns are always present.
    Adds rolling team features, betting features, and player-level features.
    Returns a DataFrame with the full expected schema.
    """
    features = []

    for i, game in enumerate(game_data_list):
        normalized = {k.upper(): v for k, v in game.items()}
        df = pd.DataFrame([normalized])

        # Required columns
        df["GAME_ID"] = df.get("GAME_ID", f"unknown_game_{i}")
        df["TEAM_ID"] = df.get("TEAM_ID", -1)
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        df["prediction_date"] = datetime.date.today().isoformat()
        df["unique_id"] = (
            df["GAME_ID"] + "_" + df["TEAM_ID"].astype(str) + "_" + df["prediction_date"]
        )

        # Betting & player features
        df["PointSpread"] = df.get("POINT_SPREAD", np.nan)
        df["OverUnder"] = df.get("OVER_UNDER", np.nan)

        # Handle player scoring (dict/list or simple flag)
        if "PLAYER_POINTS" in df.columns:
            def count_20pts(pp):
                if isinstance(pp, dict):
                    return sum(1 for p in pp.values() if p >= players_min_points)
                elif isinstance(pp, list):
                    return sum(1 for p in pp if p >= players_min_points)
                return 0
            df["Players20PlusPts"] = df["PLAYER_POINTS"].apply(count_20pts)
        else:
            df["Players20PlusPts"] = df.get("TOP_SCORER_20PTS", np.nan)

        features.append(df)

    if not features:
        logging.warning("No valid features generated.")
        return pd.DataFrame(
            columns=[
                "GAME_ID", "TEAM_ID", "unique_id", "prediction_date",
                "RollingPTS_5", "RollingWinPct_10", "RestDays",
                "TeamWinPctToDate", "OppWinPctToDate",
                "PointSpread", "OverUnder", "Players20PlusPts",
            ]
        )

    full_df = pd.concat(features, ignore_index=True)

    # --- Rolling team features ---
    if "DATE" in full_df.columns:
        full_df["DATE"] = pd.to_datetime(full_df["DATE"], errors="coerce")
        full_df = full_df.sort_values(["TEAM_ID", "DATE"])

        if "POINTS" in full_df.columns:
            full_df["RollingPTS_5"] = full_df.groupby("TEAM_ID")["POINTS"].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
        else:
            full_df["RollingPTS_5"] = np.nan

        if "TARGET" in full_df.columns:
            full_df["RollingWinPct_10"] = full_df.groupby("TEAM_ID")["TARGET"].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )
        else:
            full_df["RollingWinPct_10"] = np.nan

        full_df["RestDays"] = full_df.groupby("TEAM_ID")["DATE"].diff().dt.days.fillna(0)

        if "TARGET" in full_df.columns:
            full_df["CumulativeWins"] = full_df.groupby("TEAM_ID")["TARGET"].cumsum()
            full_df["CumulativeGames"] = full_df.groupby("TEAM_ID").cumcount() + 1
            full_df["TeamWinPctToDate"] = (
                full_df["CumulativeWins"] / full_df["CumulativeGames"]
            ).fillna(0)
        else:
            full_df["TeamWinPctToDate"] = np.nan

        if "OPPONENT_TEAM_ID" in full_df.columns:
            opp_df = full_df[["TEAM_ID", "DATE", "TeamWinPctToDate"]].rename(
                columns={"TEAM_ID": "OPPONENT_TEAM_ID", "TeamWinPctToDate": "OppWinPctToDate"}
            )
            full_df = pd.merge_asof(
                full_df.sort_values("DATE"),
                opp_df.sort_values("DATE"),
                by="OPPONENT_TEAM_ID",
                on="DATE",
                direction="backward",
            ).reset_index(drop=True)
        else:
            full_df["OppWinPctToDate"] = np.nan
    else:
        full_df["RollingPTS_5"] = np.nan
        full_df["RollingWinPct_10"] = np.nan
        full_df["RestDays"] = np.nan
        full_df["TeamWinPctToDate"] = np.nan
        full_df["OppWinPctToDate"] = np.nan

    return full_df.reset_index(drop=True)
