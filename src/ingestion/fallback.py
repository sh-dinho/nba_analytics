from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Ingestion Fallbacks (ESPN)
# File: src/ingestion/fallback.py
# Author: Sadiq
# ============================================================

import pandas as pd
from loguru import logger


def fill_missing_games_with_espn(
    schedule_df: pd.DataFrame, espn_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing games in the canonical schedule using ESPN schedule data.

    Assumptions:
        - schedule_df: canonical team-game rows from ScoreboardV3 pipeline
            Required columns:
                - game_id
                - date
                - team
                - opponent
        - espn_df: wide ESPN schedule from normalize_espn_schedule
            Required columns:
                - game_id
                - date
                - home_team
                - away_team
                - game_time
                - season
                - source

    Strategy:
        - Identify ESPN games whose game_id does NOT appear in schedule_df.game_id
        - For those games, create synthetic team-game rows (home/away)
        - Mark them with source "espn_fallback"
        - Append to schedule_df and return merged result
    """
    if schedule_df.empty and espn_df.empty:
        logger.warning("[Fallback] Both schedule_df and espn_df are empty.")
        return schedule_df

    if espn_df.empty:
        logger.info("[Fallback] ESPN schedule is empty; nothing to fill.")
        return schedule_df

    required_sched_cols = {"game_id", "date", "team", "opponent"}
    missing_sched = required_sched_cols - set(schedule_df.columns)
    if missing_sched:
        logger.error(
            f"[Fallback] schedule_df missing required columns: {sorted(missing_sched)}. "
            f"Columns present: {schedule_df.columns.tolist()}"
        )
        return schedule_df

    required_espn_cols = {
        "game_id",
        "date",
        "home_team",
        "away_team",
        "game_time",
        "season",
        "source",
    }
    missing_espn = required_espn_cols - set(espn_df.columns)
    if missing_espn:
        logger.error(
            f"[Fallback] espn_df missing required columns: {sorted(missing_espn)}. "
            f"Columns present: {espn_df.columns.tolist()}"
        )
        return schedule_df

    # Canonical schedule might have multiple rows per game (team-game).
    existing_game_ids = set(schedule_df["game_id"].astype(str).unique())
    espn_game_ids = set(espn_df["game_id"].astype(str).unique())

    missing_ids = espn_game_ids - existing_game_ids
    if not missing_ids:
        logger.info("[Fallback] No missing games found to fill from ESPN.")
        return schedule_df

    missing_espn = espn_df[espn_df["game_id"].astype(str).isin(missing_ids)].copy()
    if missing_espn.empty:
        logger.info("[Fallback] No ESPN rows matched missing game_ids.")
        return schedule_df

    logger.info(
        f"[Fallback] Filling {len(missing_espn)} missing games from ESPN "
        f"(unique game_ids={len(missing_ids)})."
    )

    # Build synthetic team-game rows from wide ESPN schedule
    home_rows = pd.DataFrame(
        {
            "game_id": missing_espn["game_id"],
            "date": missing_espn["date"],
            "team": missing_espn["home_team"],
            "opponent": missing_espn["away_team"],
            "is_home": 1,
            "score": pd.NA,
            "opponent_score": pd.NA,
            "status": "SCHEDULED",
            "season": missing_espn["season"],
            "source": "espn_fallback",
            "game_time": missing_espn["game_time"],
        }
    )

    away_rows = pd.DataFrame(
        {
            "game_id": missing_espn["game_id"],
            "date": missing_espn["date"],
            "team": missing_espn["away_team"],
            "opponent": missing_espn["home_team"],
            "is_home": 0,
            "score": pd.NA,
            "opponent_score": pd.NA,
            "status": "SCHEDULED",
            "season": missing_espn["season"],
            "source": "espn_fallback",
            "game_time": missing_espn["game_time"],
        }
    )

    fallback_long = pd.concat([home_rows, away_rows], ignore_index=True)

    # Align columns with schedule_df where possible
    # (keep extra columns like game_time/source if schedule_df doesn't yet have them)
    merged = pd.concat([schedule_df, fallback_long], ignore_index=True, sort=False)

    # Drop duplicates if any conflict (prefer ScoreboardV3 data over fallback)
    # We keep 'first', assuming schedule_df was appended first
    merged = merged.drop_duplicates(subset=["game_id", "team"], keep="first")

    # Ensure date is date type
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date

    merged = merged.sort_values(["date", "game_id", "team"]).reset_index(drop=True)
    logger.success(
        f"[Fallback] After ESPN fill: rows={len(merged)} "
        f"(fallback_rows={len(fallback_long)})"
    )

    return merged
