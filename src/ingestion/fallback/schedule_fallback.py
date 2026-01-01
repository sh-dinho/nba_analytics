from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Season Schedule Fallback
# File: src/ingestion/fallback/schedule_fallback.py
# Author: Sadiq
#
# Description:
#     Uses the canonical season schedule snapshot to fill in
#     missing team-game rows when ScoreboardV3 fails to return
#     a game (rare but possible).
# ============================================================

from datetime import date
import pandas as pd
from loguru import logger

from src.config.paths import SCHEDULE_SNAPSHOT
from src.ingestion.fallback.base import FallbackSource
from src.ingestion.normalizer.team_names import to_tricode
from src.ingestion.normalizer.season import infer_season_label


class SeasonScheduleFallback(FallbackSource):
    """
    Fallback using the canonical season schedule snapshot.
    """

    def can_fill(self, day: date, df: pd.DataFrame) -> bool:
        """
        True if:
          - schedule snapshot exists
          - snapshot contains games for this date
          - df is missing some of those games
        """
        if not SCHEDULE_SNAPSHOT.exists():
            return False

        sched = pd.read_parquet(SCHEDULE_SNAPSHOT)
        sched_day = sched[sched["date"] == pd.to_datetime(day).date()]

        if sched_day.empty:
            return False

        missing_ids = set(sched_day["game_id"]) - set(df["game_id"])
        return len(missing_ids) > 0

    def fill(self, day: date, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing games using the season schedule snapshot.
        """
        sched = pd.read_parquet(SCHEDULE_SNAPSHOT)
        sched_day = sched[sched["date"] == pd.to_datetime(day).date()]

        existing_ids = set(df["game_id"])
        missing = sched_day[~sched_day["game_id"].isin(existing_ids)]

        if missing.empty:
            return df

        logger.warning(
            f"[Fallback] Filling {len(missing)} missing games for {day} "
            f"from season schedule snapshot."
        )

        fallback_rows = []
        for _, row in missing.iterrows():
            game_id = str(row["game_id"])
            date_val = pd.to_datetime(row["date"]).date()
            season = infer_season_label(date_val)

            home = {
                "game_id": game_id,
                "date": date_val,
                "team": to_tricode(row["home_team"]),
                "opponent": to_tricode(row["away_team"]),
                "is_home": 1,
                "score": pd.NA,
                "opponent_score": pd.NA,
                "season": season,
                "status": "scheduled",
                "schema_version": "fallback_schedule",
            }

            away = {
                "game_id": game_id,
                "date": date_val,
                "team": to_tricode(row["away_team"]),
                "opponent": to_tricode(row["home_team"]),
                "is_home": 0,
                "score": pd.NA,
                "opponent_score": pd.NA,
                "season": season,
                "status": "scheduled",
                "schema_version": "fallback_schedule",
            }

            fallback_rows.extend([home, away])

        fallback_df = pd.DataFrame(fallback_rows)
        return pd.concat([df, fallback_df], ignore_index=True)