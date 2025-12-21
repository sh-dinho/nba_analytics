# ============================================================
# File: src/ingestion/transform.py
# Purpose: Convert wide-format games → long-format ML rows
# ============================================================

import pandas as pd


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts wide-format games into long-format ML rows.

    Input columns:
        game_id, date, home_team, away_team,
        home_score, away_score, status, season

    Output columns:
        game_id, date, team, opponent, is_home,
        points_for, points_against, won
    """

    if df.empty:
        return df

    home = df.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    ).copy()
    home["is_home"] = 1

    away = df.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    ).copy()
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)

    # Compute win/loss if scores exist
    if "points_for" in long_df.columns and "points_against" in long_df.columns:
        long_df["won"] = (long_df["points_for"] > long_df["points_against"]).astype(int)
    else:
        long_df["won"] = None

    return long_df[
        [
            "game_id",
            "date",
            "team",
            "opponent",
            "is_home",
            "points_for",
            "points_against",
            "won",
        ]
    ]
