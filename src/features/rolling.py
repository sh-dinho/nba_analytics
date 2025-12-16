# ============================================================
# File: src/features/rolling.py
# Purpose: Compute rolling features for teams
# ============================================================

import pandas as pd


def compute_rolling_features(
    master_schedule: pd.DataFrame, window: int = 3
) -> pd.DataFrame:
    """
    Compute rolling win rates for each team.
    """
    df = master_schedule.copy()
    df["WIN"] = df.get("WIN", 0)  # Ensure WIN column exists

    rolling_stats = []

    for team in pd.unique(df["HOME_TEAM"].append(df["AWAY_TEAM"])):
        team_games = df[
            (df["HOME_TEAM"] == team) | (df["AWAY_TEAM"] == team)
        ].sort_values("GAME_DATE")
        team_games[f"{team}_rolling_win"] = (
            team_games["WIN"].rolling(window, min_periods=1).mean()
        )
        rolling_stats.append(team_games)

    if rolling_stats:
        return pd.concat(rolling_stats).drop_duplicates("GAME_ID")
    return df
