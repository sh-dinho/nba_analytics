from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Data Quality Dashboard
# Author: Sadiq
# ============================================================

import pandas as pd
from loguru import logger
from datetime import timedelta
from src.config.paths import SCHEDULE_SNAPSHOT


def load_data():
    if not SCHEDULE_SNAPSHOT.exists():
        raise FileNotFoundError("schedule.parquet not found.")

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def dashboard():
    df = load_data()

    logger.info("📊 === DATA QUALITY DASHBOARD (v4) ===")

    # --------------------------------------------------------
    # Basic stats
    # --------------------------------------------------------
    logger.info(f"Total team-game rows: {len(df)}")
    unique_games = df["game_id"].nunique()
    logger.info(f"Total unique games: {unique_games}")

    logger.info(f"Date range: {df['date'].min()} → {df['date'].max()}")
    logger.info(f"Seasons: {sorted(df['season'].unique())}")

    # --------------------------------------------------------
    # Missing game days (not missing calendar days)
    # --------------------------------------------------------
    counts = df.groupby("date").size()
    min_d, max_d = counts.index.min(), counts.index.max()
    all_days = [min_d + timedelta(days=i) for i in range((max_d - min_d).days + 1)]

    missing_days = [
        d
        for d in all_days
        if d not in counts.index
        and (
            (d - timedelta(days=1)) in counts.index
            or (d + timedelta(days=1)) in counts.index
        )
    ]

    logger.info(f"Missing game days: {len(missing_days)}")

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------
    dupes = df[df.duplicated(subset=["game_id", "team"], keep=False)]
    logger.info(f"Duplicate team-game rows: {len(dupes)}")

    # --------------------------------------------------------
    # Score sanity
    # --------------------------------------------------------
    neg_scores = df[(df["score"] < 0) | (df["opponent_score"] < 0)]
    logger.info(f"Negative score rows: {len(neg_scores)}")

    # --------------------------------------------------------
    # Games per month
    # --------------------------------------------------------
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month")["game_id"].nunique()

    logger.info("Games per month:")
    for m, c in monthly.items():
        logger.info(f"  {m}: {c}")

    # --------------------------------------------------------
    # Games per team
    # --------------------------------------------------------
    team_counts = df["team"].value_counts()
    logger.info("Games per team:")
    for team, count in team_counts.items():
        logger.info(f"  {team}: {count}")

    logger.success("🏁 Data quality dashboard complete.")


def run_ingestion_health_check(df: pd.DataFrame):
    logger.info("🔍 Running ingestion health check (v4)...")

    required_cols = [
        "game_id",
        "date",
        "team",
        "opponent",
        "is_home",
        "score",
        "opponent_score",
        "season",
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.error(f"❌ Missing required columns: {missing}")
    else:
        logger.info("✔ Schema OK")

    # Duplicate team-game rows
    dupes = df[df.duplicated(subset=["game_id", "team"], keep=False)]
    if not dupes.empty:
        logger.warning(f"⚠ Duplicate game_id/team rows: {len(dupes)}")
    else:
        logger.info("✔ No duplicate team-game rows")

    # Missing game days
    counts = df.groupby("date").size()
    min_d, max_d = counts.index.min(), counts.index.max()
    all_days = [min_d + timedelta(days=i) for i in range((max_d - min_d).days + 1)]

    missing_days = [
        d
        for d in all_days
        if d not in counts.index
        and (
            (d - timedelta(days=1)) in counts.index
            or (d + timedelta(days=1)) in counts.index
        )
    ]

    if missing_days:
        logger.warning(f"⚠ Missing {len(missing_days)} game days")
    else:
        logger.info("✔ No missing game days")

    # Score sanity
    if (df["score"] < 0).any() or (df["opponent_score"] < 0).any():
        logger.error("❌ Negative scores detected")
    else:
        logger.info("✔ Score sanity OK")

    logger.success("🏁 Ingestion health check complete.")
