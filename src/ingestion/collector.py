# ============================================================
# 🏀 NBA Analytics v3
# Module: Ingestion Collector
# File: src/ingestion/collector.py
# Author: Sadiq
#
# Description:
#     Fetches daily scoreboard data from nba_api (ScoreboardV3),
#     applies retry logic, and returns a raw normalized dataframe
#     ready for validation and persistence.
# ============================================================

from __future__ import annotations

import time
from datetime import date
from typing import Callable

import pandas as pd
from loguru import logger
from nba_api.stats.endpoints import ScoreboardV3


def _retry(func: Callable[[], pd.DataFrame], max_retries: int = 3) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exc = e
            wait = 2**attempt
            logger.warning(
                f"Ingestion retry {attempt + 1}/{max_retries} failed: {e}. "
                f"Sleeping {wait}s before next attempt."
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _fetch_scoreboard_raw(target_date: date) -> pd.DataFrame:
    """
    Low-level wrapper around ScoreboardV3. Returns a dataframe with the
    raw 'gameHeader' data if available, otherwise an empty dataframe.
    """
    logger.info(f"📅 Fetching ScoreboardV3 for {target_date}")
    sb = ScoreboardV3(game_date=target_date.strftime("%Y-%m-%d"))
    data_sets = sb.get_dict().get("resultSets", []) or sb.get_normalized_dict()

    # Handle both old-style ("resultSets") and normalized-style keys
    game_header_df = None

    if isinstance(data_sets, dict) and "GameHeader" in data_sets:
        game_header_df = pd.DataFrame(data_sets["GameHeader"])
    elif isinstance(data_sets, list):
        for ds in data_sets:
            if ds.get("name") == "GameHeader":
                game_header_df = pd.DataFrame(
                    ds.get("rowSet", []), columns=ds.get("headers", [])
                )
                break

    if game_header_df is None or game_header_df.empty:
        logger.warning(
            f"No valid game header found in ScoreboardV3 datasets for {target_date}"
        )
        return pd.DataFrame()

    return game_header_df


def fetch_scoreboard(target_date: date) -> pd.DataFrame:
    """
    Public ingestion entry point for a single date.
    Returns a normalized dataframe with columns:
    ['game_id', 'date', 'home_team', 'away_team',
     'home_score', 'away_score', 'status', 'season']
    """
    df_raw = _retry(lambda: _fetch_scoreboard_raw(target_date))

    if df_raw.empty:
        return df_raw

    # Normalize columns depending on nba_api schema
    # Expect columns like: 'GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', etc.
    cols = {c.upper(): c for c in df_raw.columns}

    def _get(col: str, default=None):
        return (
            df_raw[cols.get(col, col)]
            if cols.get(col, col) in df_raw.columns
            else default
        )

    df = pd.DataFrame(
        {
            "game_id": _get("GAME_ID"),
            "date": pd.to_datetime(
                _get("GAME_DATE_EST") or _get("GAME_DATE") or target_date
            ).dt.date,
            "home_team": _get("HOME_TEAM_NAME")
            or _get("HOME_TEAM_ABBREVIATION")
            or _get("HOME_TEAM_ID"),
            "away_team": _get("VISITOR_TEAM_NAME")
            or _get("VISITOR_TEAM_ABBREVIATION")
            or _get("VISITOR_TEAM_ID"),
            "home_score": _get("PTS_HOME") or _get("PTS") or 0,
            "away_score": _get("PTS_VISITOR") or _get("PTS") or 0,
            "status": _get("GAME_STATUS_TEXT") or "scheduled",
        }
    )

    # Basic season inference: season spanning from October to June
    df["season"] = df["date"].apply(
        lambda d: (
            f"{d.year}-{d.year + 1}" if d.month >= 10 else f"{d.year - 1}-{d.year}"
        )
    )

    return df
