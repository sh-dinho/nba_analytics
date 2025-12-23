from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Project: NBA Analytics & Betting Engine
# Module: Wide → Long Converter (Legacy / Fallback)
# File: src/ingestion/wide_to_long.py
# Author: Sadiq
#
# Description:
#     Converts canonical WIDE-format schedule rows into
#     TEAM-GAME (long) rows for modeling.
#
# IMPORTANT (v4):
#     - The v4 normalizer already outputs TEAM-GAME rows.
#     - This module is ONLY used as a fallback if a future
#       ingestion source provides WIDE-format data.
#     - If input is already long, it is passed through safely.
# ============================================================

import pandas as pd
from loguru import logger


# ------------------------------------------------------------
# Schema helpers
# ------------------------------------------------------------

LONG_COLUMNS = {
    "game_id",
    "date",
    "team",
    "opponent",
    "is_home",
    "score",
    "opponent_score",
    "season",
}

WIDE_COLUMNS = {
    "game_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "season",
}

# NEW: minimal scoreboard schema (no season, different score names)
MINIMAL_COLUMNS = {
    "game_id",
    "date",
    "home_team",
    "away_team",
    "score_home",
    "score_away",
}


def _is_long_format(df: pd.DataFrame) -> bool:
    return LONG_COLUMNS.issubset(df.columns)


def _is_wide_format(df: pd.DataFrame) -> bool:
    return WIDE_COLUMNS.issubset(df.columns)


def _is_minimal_format(df: pd.DataFrame) -> bool:
    return MINIMAL_COLUMNS.issubset(df.columns)


# ------------------------------------------------------------
# Main conversion function
# ------------------------------------------------------------


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert WIDE-format schedule rows into TEAM-GAME (long) rows.

    Behavior:
        - If df is already LONG → return as-is
        - If df is WIDE → convert to LONG
        - If df is MINIMAL → convert to LONG (fallback)
        - Otherwise → raise explicit error

    Returns:
        DataFrame in TEAM-GAME (long) format
    """
    if df.empty:
        return df

    # --------------------------------------------------------
    # Case 1: Already long (v4 canonical)
    # --------------------------------------------------------
    if _is_long_format(df):
        logger.debug("wide_to_long: input already long-format; passing through")
        return df.copy()

    # --------------------------------------------------------
    # Case 2: Proper wide format → convert
    # --------------------------------------------------------
    if _is_wide_format(df):
        logger.info("wide_to_long: converting wide-format data to team-game rows")

        status = df["status"] if "status" in df.columns else None
        schema_version = (
            df["schema_version"] if "schema_version" in df.columns else None
        )

        home = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "date": df["date"],
                "team": df["home_team"],
                "opponent": df["away_team"],
                "is_home": 1,
                "score": df["home_score"],
                "opponent_score": df["away_score"],
                "season": df["season"],
                "status": status,
                "schema_version": schema_version,
            }
        )

        away = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "date": df["date"],
                "team": df["away_team"],
                "opponent": df["home_team"],
                "is_home": 0,
                "score": df["away_score"],
                "opponent_score": df["home_score"],
                "season": df["season"],
                "status": status,
                "schema_version": schema_version,
            }
        )

        long_df = pd.concat([home, away], ignore_index=True)

        return long_df[
            [
                "game_id",
                "date",
                "team",
                "opponent",
                "is_home",
                "score",
                "opponent_score",
                "season",
                "status",
                "schema_version",
            ]
        ]

    # --------------------------------------------------------
    # NEW Case 3: Minimal scoreboard schema (fallback)
    # --------------------------------------------------------
    if _is_minimal_format(df):
        logger.warning(
            "wide_to_long: using MINIMAL scoreboard schema fallback "
            "(no season, score_home/score_away)."
        )

        # infer season if possible
        if "season" in df.columns:
            season = df["season"]
        else:
            # best-effort: season = year of date
            season = pd.to_datetime(df["date"]).dt.year

        home = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "date": df["date"],
                "team": df["home_team"],
                "opponent": df["away_team"],
                "is_home": 1,
                "score": df["score_home"],
                "opponent_score": df["score_away"],
                "season": season,
                "status": None,
                "schema_version": "minimal",
            }
        )

        away = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "date": df["date"],
                "team": df["away_team"],
                "opponent": df["home_team"],
                "is_home": 0,
                "score": df["score_away"],
                "opponent_score": df["score_home"],
                "season": season,
                "status": None,
                "schema_version": "minimal",
            }
        )

        long_df = pd.concat([home, away], ignore_index=True)

        return long_df[
            [
                "game_id",
                "date",
                "team",
                "opponent",
                "is_home",
                "score",
                "opponent_score",
                "season",
                "status",
                "schema_version",
            ]
        ]

    # --------------------------------------------------------
    # Unknown schema → explicit error
    # --------------------------------------------------------
    raise ValueError(
        "wide_to_long received DataFrame with unknown schema. "
        f"Columns={list(df.columns)}"
    )
