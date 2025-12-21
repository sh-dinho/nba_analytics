# ============================================================
# Full-History Ingestor (using nba_api)
# Creates canonical snapshots:
#   - schedule_snapshot.parquet
#   - long_snapshot.parquet
# ============================================================

from __future__ import annotations

import pandas as pd
from pathlib import Path
from loguru import logger

from nba_api.stats.endpoints import LeagueGameLog
import time

from src.config.paths import (
    SCHEDULE_SNAPSHOT,
    LONG_SNAPSHOT,
)

RETRY_DELAY = 5
MAX_RETRIES = 5
START_YEAR = 2020


# ------------------------------------------------------------
# Fetching helpers
# ------------------------------------------------------------


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch a full NBA season using nba_api."""
    frames = []

    for season_type in ["Regular Season", "Playoffs"]:
        df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Fetching {season_type} for {season} (attempt {attempt})")
                log = LeagueGameLog(season=season, season_type_all_star=season_type)
                df = log.get_data_frames()[0]
                if df.empty:
                    df = None
                else:
                    df["season_type"] = season_type
                break
            except Exception as e:
                logger.warning(f"Error fetching {season_type} {season}: {e}")
                time.sleep(RETRY_DELAY)

        if df is not None:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def fetch_all_seasons(start_year=START_YEAR) -> pd.DataFrame:
    """Fetch all seasons from start_year → current year."""
    frames = []
    current_year = pd.Timestamp.today().year

    for year in range(start_year, current_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        logger.info(f"Fetching full season {season}")
        df = fetch_season(season)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------


def normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nba_api schema → canonical schedule schema."""
    if df.empty:
        return df

    df = df.rename(
        columns={
            "GAME_DATE": "date",
            "MATCHUP": "matchup",
            "PTS": "points",
            "TEAM_NAME": "team",
            "GAME_ID": "game_id",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    def parse_matchup(row):
        m = row["matchup"]
        if " vs. " in m:
            home, away = m.split(" vs. ")
        elif " @ " in m:
            away, home = m.split(" @ ")
        else:
            home = row["team"]
            away = None
        return pd.Series([home, away])

    df[["home_team", "away_team"]] = df.apply(parse_matchup, axis=1)

    df["home_score"] = df.apply(
        lambda r: r["points"] if r["team"] == r["home_team"] else None, axis=1
    )
    df["away_score"] = df.apply(
        lambda r: r["points"] if r["team"] == r["away_team"] else None, axis=1
    )

    agg = {
        "date": ("date", "first"),
        "home_team": ("home_team", "first"),
        "away_team": ("away_team", "first"),
        "home_score": ("home_score", "max"),
        "away_score": ("away_score", "max"),
        "season_type": ("season_type", "first"),
    }

    df = df.groupby("game_id").agg(**agg).reset_index()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["home_team", "away_team"])

    return df


# ------------------------------------------------------------
# Long-format builder
# ------------------------------------------------------------


def build_long_format(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide schedule → long format for modeling."""
    if schedule_df.empty:
        return schedule_df

    home = schedule_df.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "score",
            "away_score": "opponent_score",
        }
    ).assign(is_home=1)

    away = schedule_df.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "score",
            "home_score": "opponent_score",
        }
    ).assign(is_home=0)

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
            "season_type",
        ]
    ]


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------


def run_full_history_ingestion() -> dict:
    logger.info("=== Full-History Ingestion Start ===")

    raw = fetch_all_seasons()
    schedule = normalize_schedule(raw)
    long_df = build_long_format(schedule)

    SCHEDULE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    LONG_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)

    schedule.to_parquet(SCHEDULE_SNAPSHOT, index=False)
    long_df.to_parquet(LONG_SNAPSHOT, index=False)

    logger.success("Full-history ingestion complete.")

    return {
        "schedule_rows": len(schedule),
        "long_rows": len(long_df),
        "snapshot_paths": {
            "schedule": str(SCHEDULE_SNAPSHOT),
            "long": str(LONG_SNAPSHOT),
        },
    }


if __name__ == "__main__":
    run_full_history_ingestion()
