from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Fallback Schedule → Team-Game Rows
# File: src/ingestion/fallback_schedule.py
#
# Description:
#     Uses season_schedule.parquet to synthesize team-game
#     rows for dates where the primary NBA API ingestion
#     has not populated LONG_SNAPSHOT.
#
#     Output rows have:
#       - game_id
#       - date
#       - team (tricode)
#       - opponent (tricode)
#       - is_home (1/0)
# ============================================================

from datetime import date as Date
from typing import List

import pandas as pd
from loguru import logger

from src.config.paths import SEASON_SCHEDULE_PATH, LONG_SNAPSHOT


def _load_season_schedule() -> pd.DataFrame:
    if not SEASON_SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"Season schedule not found at {SEASON_SCHEDULE_PATH}. "
            f"Run schedule_scraper.build_season_schedule(...) first "
            f"or let schedule_refresh.refresh_schedule_if_needed() handle it."
        )
    df = pd.read_parquet(SEASON_SCHEDULE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_games_for_date(pred_date: Date) -> pd.DataFrame:
    df = _load_season_schedule()
    todays = df[df["date"].dt.date == pred_date].copy()
    if todays.empty:
        logger.warning(
            f"[FallbackSchedule] No games in season schedule for {pred_date}."
        )
    return todays


def build_team_game_rows_for_date(pred_date: Date) -> pd.DataFrame:
    """
    Build minimal team-game rows for pred_date from the season schedule.

    Columns:
      - game_id
      - date
      - team
      - opponent
      - is_home
    """
    games = get_games_for_date(pred_date)
    if games.empty:
        return pd.DataFrame()

    rows: List[dict] = []

    for _, r in games.iterrows():
        game_id = r["game_id"]
        d = r["date"]

        home = r["home_team"]
        away = r["away_team"]

        rows.append(
            {
                "game_id": game_id,
                "date": d,
                "team": home,
                "opponent": away,
                "is_home": 1,
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "date": d,
                "team": away,
                "opponent": home,
                "is_home": 0,
            }
        )

    df_team = pd.DataFrame(rows)
    df_team["date"] = pd.to_datetime(df_team["date"])

    return df_team


def ensure_long_snapshot_has_date(pred_date: Date) -> pd.DataFrame:
    """
    Ensure LONG_SNAPSHOT has team-game rows for pred_date.
    If missing, synthesize from season schedule and append.
    Returns the rows for pred_date (from long snapshot after update).
    """
    # Load existing long snapshot (or create empty)
    if LONG_SNAPSHOT.exists():
        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"])
    else:
        long_df = pd.DataFrame()

    todays_long = (
        long_df[long_df["date"].dt.date == pred_date].copy()
        if not long_df.empty
        else pd.DataFrame()
    )

    if not todays_long.empty:
        logger.info(
            f"[FallbackSchedule] LONG_SNAPSHOT already has rows for {pred_date} "
            f"(rows={len(todays_long)}). No fallback needed."
        )
        return todays_long

    # Build fallback rows
    fallback_rows = build_team_game_rows_for_date(pred_date)
    if fallback_rows.empty:
        logger.warning(
            f"[FallbackSchedule] No fallback games available for {pred_date}. "
            f"Cannot synthesize long rows."
        )
        return pd.DataFrame()

    logger.warning(
        f"[FallbackSchedule] LONG_SNAPSHOT missing rows for {pred_date}. "
        f"Synthesizing {len(fallback_rows)} rows from season schedule."
    )

    # Append and save
    updated = pd.concat([long_df, fallback_rows], ignore_index=True)
    updated.to_parquet(LONG_SNAPSHOT, index=False)

    todays_long_updated = updated[updated["date"].dt.date == pred_date].copy()
    logger.success(
        f"[FallbackSchedule] LONG_SNAPSHOT updated with fallback rows for {pred_date} "
        f"(rows={len(todays_long_updated)})."
    )
    return todays_long_updated
