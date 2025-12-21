# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Convert canonical wide-format schedule into
#              long-format team-level rows for modeling.
# ============================================================

from __future__ import annotations

import pandas as pd
from loguru import logger


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert canonical wide-format schedule into long-format team-level rows.

    Expected input schema (one row per game):
        game_id       : str
        date          : datetime64[ns]
        season        : int
        season_type   : str
        home_team     : str
        away_team     : str
        home_score    : float or None
        away_score    : float or None
        status        : {"scheduled","final","unknown"}

    Output schema (two rows per game):
        game_id
        date
        season
        season_type
        team
        opponent
        is_home
        points_for
        points_against
        won
        status
        game_number   (per-team sequential game index within season)
    """

    required = {
        "game_id",
        "date",
        "season",
        "season_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status",
    }

    missing = required - set(df_wide.columns)
    if missing:
        raise ValueError(f"wide_to_long(): Missing required columns: {missing}")

    if df_wide.empty:
        logger.info("wide_to_long(): received empty DataFrame.")
        return pd.DataFrame(
            columns=[
                "game_id",
                "date",
                "season",
                "season_type",
                "team",
                "opponent",
                "is_home",
                "points_for",
                "points_against",
                "won",
                "status",
                "game_number",
            ]
        )

    df = df_wide.copy()

    # ----------------------------------------------------------------------
    # Build home rows
    # ----------------------------------------------------------------------
    home = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "date": df["date"],
            "season": df["season"],
            "season_type": df["season_type"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": 1,
            "points_for": df["home_score"],
            "points_against": df["away_score"],
            "status": df["status"],
        }
    )

    # ----------------------------------------------------------------------
    # Build away rows
    # ----------------------------------------------------------------------
    away = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "date": df["date"],
            "season": df["season"],
            "season_type": df["season_type"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "points_for": df["away_score"],
            "points_against": df["home_score"],
            "status": df["status"],
        }
    )

    # ----------------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------------
    long_df = pd.concat([home, away], ignore_index=True)

    # ----------------------------------------------------------------------
    # Compute win/loss (only for final games)
    # ----------------------------------------------------------------------
    def compute_win(row):
        if row["status"] != "final":
            return None
        if pd.isna(row["points_for"]) or pd.isna(row["points_against"]):
            return None
        return int(row["points_for"] > row["points_against"])

    long_df["won"] = long_df.apply(compute_win, axis=1)

    # ----------------------------------------------------------------------
    # Sort for game_number assignment
    # ----------------------------------------------------------------------
    long_df = long_df.sort_values(["team", "season", "date", "game_id"])

    # ----------------------------------------------------------------------
    # Assign per-team game_number within each season
    # ----------------------------------------------------------------------
    long_df["game_number"] = (
        long_df.groupby(["team", "season"]).cumcount().astype("int64")
    )

    # ----------------------------------------------------------------------
    # Final sanity checks
    # ----------------------------------------------------------------------
    before = len(long_df)
    long_df = long_df.dropna(subset=["team", "opponent"])
    dropped = before - len(long_df)
    if dropped:
        logger.warning(
            f"wide_to_long(): dropped {dropped} rows with missing team/opponent."
        )

    # Enforce column order
    long_df = long_df[
        [
            "game_id",
            "date",
            "season",
            "season_type",
            "team",
            "opponent",
            "is_home",
            "points_for",
            "points_against",
            "won",
            "status",
            "game_number",
        ]
    ]

    return long_df
