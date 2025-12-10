# ============================================================
# File: src/prediction_engine/game_features.py
# Purpose: Generate features for NBA games (rolling stats + betting + top scorers)
# Project: nba_analysis
# Version: 1.1
# ============================================================

import pandas as pd
import datetime
import logging

def generate_features_for_games(game_data_list):
    """
    Generate features for a list of games.
    Includes rolling stats, point spread, over/under, top scorer 20+ pts.
    """
    features = []

    for i, game in enumerate(game_data_list):
        normalized = {k.upper(): v for k,v in game.items()}
        df = pd.DataFrame([normalized])

        # Required columns
        df["GAME_ID"] = df.get("GAME_ID", f"unknown_game_{i}")
        df["TEAM_ID"] = df.get("TEAM_ID", -1)
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        df["prediction_date"] = datetime.date.today().isoformat()
        df["unique_id"] = df["GAME_ID"] + "_" + df["TEAM_ID"].astype(str) + "_" + df["prediction_date"]

        # Betting & top scorer features
        df["PointSpread"] = df.get("POINT_SPREAD", 0)
        df["OverUnder"] = df.get("OVER_UNDER", 0)
        df["TopScorer20Pts"] = df.get("TOP_SCORER_20PTS", 0)

        features.append(df)

    if not features:
        return pd.DataFrame(columns=[
            "GAME_ID","TEAM_ID","unique_id","prediction_date",
            "RollingPTS_5","RollingWinPct_10","RestDays",
            "TeamWinPctToDate","OppWinPctToDate",
            "PointSpread","OverUnder","TopScorer20Pts"
        ])

    full_df = pd.concat(features, ignore_index=True)

    # Optional: compute rolling stats if DATE, POINTS, TARGET columns exist
    if "DATE" in full_df.columns:
        full_df["DATE"] = pd.to_datetime(full_df["DATE"], errors="coerce")
        full_df = full_df.sort_values(["TEAM_ID","DATE"])
        if "POINTS" in full_df.columns:
            full_df["RollingPTS_5"] = full_df.groupby("TEAM_ID")["POINTS"].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
        else:
            full_df["RollingPTS_5"] = 0
        if "TARGET" in full_df.columns:
            full_df["RollingWinPct_10"] = full_df.groupby("TEAM_ID")["TARGET"].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )
        else:
            full_df["RollingWinPct_10"] = 0
        full_df["RestDays"] = full_df.groupby("TEAM_ID")["DATE"].diff().dt.days.fillna(0)
        if "TARGET" in full_df.columns:
            full_df["CumulativeWins"] = full_df.groupby("TEAM_ID")["TARGET"].cumsum()
            full_df["CumulativeGames"] = full_df.groupby("TEAM_ID").cumcount() + 1
            full_df["TeamWinPctToDate"] = (full_df["CumulativeWins"]/full_df["CumulativeGames"]).fillna(0)
        else:
            full_df["TeamWinPctToDate"] = 0
        if "OPPONENT_TEAM_ID" in full_df.columns:
            opp_df = full_df[["TEAM_ID","DATE","TeamWinPctToDate"]].rename(
                columns={"TEAM_ID":"OPPONENT_TEAM_ID","TeamWinPctToDate":"OppWinPctToDate"}
            )
            full_df = pd.merge_asof(
                full_df.sort_values("DATE"),
                opp_df.sort_values("DATE"),
                by="OPPONENT_TEAM_ID",
                on="DATE",
                direction="backward"
            )
        else:
            full_df["OppWinPctToDate"] = 0
    else:
        full_df["RollingPTS_5"] = 0
        full_df["RollingWinPct_10"] = 0
        full_df["RestDays"] = 0
        full_df["TeamWinPctToDate"] = 0
        full_df["OppWinPctToDate"] = 0

    return full_df.reset_index(drop=True)
