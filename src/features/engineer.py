# ============================================================
# File: src/features/engineer.py
# Purpose: Prepare features for ML model
# ============================================================

import pandas as pd


def prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare schedule dataframe with features for ML model.
    """
    if schedule.empty:
        return pd.DataFrame()

    df = schedule.copy()

    # Example features (you can extend)
    df["feat_home_adv"] = df["HOME_TEAM_WINS_LAST5"] - df["AWAY_TEAM_WINS_LAST5"]
    df["feat_win_streak"] = df.get("WIN_STREAK", 0)
    df["feat_avg_pts_diff"] = df.get("HOME_TEAM_AVG_PTS", 0) - df.get(
        "AWAY_TEAM_AVG_PTS", 0
    )

    # Convert to int where needed
    df["feat_home_adv"] = df["feat_home_adv"].astype("Int64")
    df["feat_win_streak"] = df["feat_win_streak"].astype("Int64")
    df["feat_avg_pts_diff"] = df["feat_avg_pts_diff"].astype("Int64")

    return df
