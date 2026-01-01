from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Scoreboard Normalizer
# File: src/ingestion/normalizer/scoreboard_normalizer.py
# Author: Sadiq
#
# Description:
#     Normalizes ScoreboardV3 raw responses into canonical WIDE
#     game rows for ingestion. Handles alias resolution,
#     minimal schema fallback, dtype coercion, and safe parsing.
# ============================================================

import pandas as pd
from loguru import logger


ALIASES = {
    "date": ["gameDateEst", "gameDate"],
    "home_team": ["homeTeamName"],
    "away_team": ["awayTeamName", "visitorTeamName"],
    "home_score": ["homeScore", "homeTeamScore"],
    "away_score": ["awayScore", "visitorTeamScore"],
}

# Minimal fallback schema for extremely degraded API responses
MINIMAL_SCHEMA = {
    "gameId",
    "date",
    "home_team",
    "away_team",
    "score_home",
    "score_away",
}


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_scoreboard_to_wide(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ScoreboardV3 raw data into WIDE canonical rows.

    Output columns:
        game_id
        date
        home_team
        away_team
        home_score
        away_score
        status
        schema_version
    """
    if df_raw.empty:
        return pd.DataFrame()

    if "gameId" not in df_raw.columns:
        logger.error(
            "normalize_scoreboard_to_wide received DataFrame without gameId. "
            f"Columns={df_raw.columns.tolist()}"
        )
        return pd.DataFrame()

    # Alias resolution
    date_col = _first_existing(df_raw, ALIASES["date"])
    home_team_col = _first_existing(df_raw, ALIASES["home_team"])
    away_team_col = _first_existing(df_raw, ALIASES["away_team"])
    home_score_col = _first_existing(df_raw, ALIASES["home_score"])
    away_score_col = _first_existing(df_raw, ALIASES["away_score"])

    # Minimal fallback
    if home_team_col is None or away_team_col is None:
        if MINIMAL_SCHEMA.issubset(df_raw.columns):
            logger.warning(
                "normalize_scoreboard_to_wide: using MINIMAL scoreboard schema fallback."
            )
            wide = pd.DataFrame(
                {
                    "game_id": df_raw["gameId"].astype(str),
                    "date": pd.to_datetime(df_raw["date"], errors="coerce"),
                    "home_team": df_raw["home_team"].astype(str),
                    "away_team": df_raw["away_team"].astype(str),
                    "home_score": pd.to_numeric(df_raw["score_home"], errors="coerce"),
                    "away_score": pd.to_numeric(df_raw["score_away"], errors="coerce"),
                    "status": df_raw.get("gameStatusText"),
                    "schema_version": "minimal",
                }
            )
        else:
            logger.error(
                "normalize_scoreboard_to_wide could not resolve required fields via "
                "aliases or minimal schema. "
                f"Columns={df_raw.columns.tolist()}"
            )
            return pd.DataFrame()
    else:
        wide = pd.DataFrame(
            {
                "game_id": df_raw["gameId"].astype(str),
                "date": pd.to_datetime(df_raw[date_col], errors="coerce"),
                "home_team": df_raw[home_team_col].astype(str),
                "away_team": df_raw[away_team_col].astype(str),
                "home_score": (
                    pd.to_numeric(df_raw[home_score_col], errors="coerce")
                    if home_score_col
                    else pd.NA
                ),
                "away_score": (
                    pd.to_numeric(df_raw[away_score_col], errors="coerce")
                    if away_score_col
                    else pd.NA
                ),
                "status": df_raw.get("gameStatusText"),
                "schema_version": "scoreboard_v3",
            }
        )

    # Drop invalid rows
    wide = wide.dropna(subset=["game_id", "date", "home_team", "away_team"])
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")

    # Normalize status
    wide["status"] = wide["status"].astype(str).str.lower()

    return wide.reset_index(drop=True)