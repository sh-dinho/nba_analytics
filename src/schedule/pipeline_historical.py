# ============================================================
# File: src/schedule/pipeline_historical.py
# Purpose: Feature engineering, strength, predicted win, rankings (v2.0)
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

from pathlib import Path

import pandas as pd

from src.utils.common import save_dataframe


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic game-level features:
    - homeWin: 1 if homeScore > awayScore, else 0
    - point_diff: homeScore - awayScore
    """
    df = df.copy()

    df["homeWin"] = (df["homeScore"] > df["awayScore"]).astype(int)
    df["point_diff"] = df["homeScore"] - df["awayScore"]

    return df


def add_team_strength(
    df: pd.DataFrame,
    streak_weight: float = 0.3,
    diff_weight: float = 0.4,
    winpct_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Compute team strength based on:
    - cumulative win/loss streaks
    - point differential
    - historical win percentage

    Assumes one row per game with:
    - homeTeam, awayTeam
    - homeWin (0/1)
    - point_diff
    """
    df = df.copy()

    # Win pct by team to date (simple aggregate over all rows)
    win_pct_home = df.groupby("homeTeam")["homeWin"].transform("mean").fillna(0.5)
    win_pct_away = (
        df.groupby("awayTeam")["homeWin"].transform(lambda s: 1 - s.mean()).fillna(0.5)
    )

    # Rolling/home streak: cumulative number of wins per home team
    df["streak_home"] = df.groupby("homeTeam")["homeWin"].transform(
        lambda s: s.cumsum()
    )

    # Away streak: cumulative count of losses per away team
    # (~homeWin.astype(bool)) is True for losses → 1, then cumsum.
    df["streak_away"] = df.groupby("awayTeam")["homeWin"].transform(
        lambda s: (~s.astype(bool)).cumsum()
    )

    # Strength scores (clipping point_diff to reduce outlier impact)
    df["strength_home"] = (
        streak_weight * df["streak_home"]
        + diff_weight * df["point_diff"].clip(-30, 30)
        + winpct_weight * win_pct_home * 100
    )

    df["strength_away"] = (
        streak_weight * df["streak_away"]
        + diff_weight * (-df["point_diff"]).clip(-30, 30)
        + winpct_weight * win_pct_away * 100
    )

    return df


def add_predicted_win(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple logistic proxy for home win probability based on
    strength difference.
    """
    df = df.copy()

    sd = (df["strength_home"] - df["strength_away"]).clip(-100, 100)
    df["predicted_win"] = 1 / (1 + (2.718281828459045 ** (-sd / 15)))

    return df


def generate_daily_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a simple team ranking based on average predicted_win
    over all games, aggregating home and away.
    """
    # Home contribution
    home_rank = (
        df.groupby("homeTeam")["predicted_win"]
        .mean()
        .reset_index()
        .rename(columns={"homeTeam": "team", "predicted_win": "score"})
    )

    # Away contribution
    away_rank = (
        df.groupby("awayTeam")["predicted_win"]
        .mean()
        .reset_index()
        .rename(columns={"awayTeam": "team", "predicted_win": "score"})
    )

    # Combine and average
    rank = (
        pd.concat([home_rank, away_rank], axis=0)
        .groupby("team")["score"]
        .mean()
        .reset_index()
    )

    rank["rank"] = rank["score"].rank(ascending=False, method="dense").astype(int)
    return rank.sort_values(["rank", "team"]).reset_index(drop=True)


def save_rankings(
    df: pd.DataFrame,
    path: Path = Path("data/cache/rankings.parquet"),
) -> None:
    """
    Save rankings snapshot to a parquet file.
    """
    save_dataframe(df, path)


def merge_rankings_history() -> pd.DataFrame:
    """
    For v2.0, simply return an empty DataFrame.
    Hook for future extension to maintain historical rankings over time.
    """
    return pd.DataFrame()
