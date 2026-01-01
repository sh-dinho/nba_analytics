from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Wide to Long Converter
# File: src/ingestion/normalizer/wide_to_long.py
# Author: Sadiq
#
# Description:
#     Converts canonical WIDE-format schedule rows into TEAM-GAME
#     (long) rows for ingestion. Handles standard wide schema,
#     minimal schema, and passes through already-long canonical
#     data safely.
# ============================================================

import pandas as pd
from loguru import logger

from src.ingestion.normalizer.season import infer_season_label
from src.ingestion.normalizer.team_names import to_tricode


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
}

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


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert WIDE-format schedule rows into TEAM-GAME (long) rows.

    Behavior:
      - If df is already LONG → return copy with normalized date dtype
      - If df is WIDE → convert and infer season
      - If df is MINIMAL → convert and infer season
      - Else → raise explicit error
    """
    if df.empty:
        return df

    # Already long
    if _is_long_format(df):
        logger.debug("wide_to_long: input already long-format; passing through.")
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        return out

    # Standard wide schema
    if _is_wide_format(df):
        logger.info("wide_to_long: converting wide-format data to team-game rows.")

        status = df.get("status")
        schema_version = "scoreboard_v3"

        dates = pd.to_datetime(df["date"])
        seasons = dates.dt.date.map(infer_season_label)

        home = pd.DataFrame(
            {
                "game_id": df["game_id"].astype(str),
                "date": dates,
                "team": df["home_team"].map(to_tricode),
                "opponent": df["away_team"].map(to_tricode),
                "is_home": 1,
                "score": df["home_score"],
                "opponent_score": df["away_score"],
                "season": seasons,
                "status": status,
                "schema_version": schema_version,
            }
        )

        away = pd.DataFrame(
            {
                "game_id": df["game_id"].astype(str),
                "date": dates,
                "team": df["away_team"].map(to_tricode),
                "opponent": df["home_team"].map(to_tricode),
                "is_home": 0,
                "score": df["away_score"],
                "opponent_score": df["home_score"],
                "season": seasons,
                "status": status,
                "schema_version": schema_version,
            }
        )

        long_df = pd.concat([home, away], ignore_index=True)
        long_df["date"] = long_df["date"].dt.date
        return long_df

    # Minimal schema fallback
    if _is_minimal_format(df):
        logger.warning(
            "wide_to_long: using MINIMAL scoreboard schema fallback "
            "(score_home/score_away, season inferred)."
        )

        dates = pd.to_datetime(df["date"])
        seasons = dates.dt.date.map(infer_season_label)

        home = pd.DataFrame(
            {
                "game_id": df["game_id"].astype(str),
                "date": dates,
                "team": df["home_team"].map(to_tricode),
                "opponent": df["away_team"].map(to_tricode),
                "is_home": 1,
                "score": df["score_home"],
                "opponent_score": df["score_away"],
                "season": seasons,
                "status": None,
                "schema_version": "minimal",
            }
        )

        away = pd.DataFrame(
            {
                "game_id": df["game_id"].astype(str),
                "date": dates,
                "team": df["away_team"].map(to_tricode),
                "opponent": df["home_team"].map(to_tricode),
                "is_home": 0,
                "score": df["score_away"],
                "opponent_score": df["score_home"],
                "season": seasons,
                "status": None,
                "schema_version": "minimal",
            }
        )

        long_df = pd.concat([home, away], ignore_index=True)
        long_df["date"] = long_df["date"].dt.date
        return long_df

    raise ValueError(
        "wide_to_long: received DataFrame with unknown schema. "
        f"Columns={list(df.columns)}"
    )