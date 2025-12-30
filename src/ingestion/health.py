# ============================================================
# 🏀 NBA Analytics v4
# Module: Ingestion Health Checks
# File: src/ingestion/health.py
# Author: Sadiq
#
# Description:
#     Validates the canonical team-game snapshot for schema
#     correctness, duplicates, and basic consistency.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from loguru import logger

from src.config.paths import SCHEDULE_SNAPSHOT


@dataclass
class IngestionHealthReport:
    ok: bool
    errors: List[str]
    warnings: List[str]
    rows: int
    unique_games: int
    seasons: List[str]


def run_ingestion_health_check() -> IngestionHealthReport:
    if not SCHEDULE_SNAPSHOT.exists():
        msg = f"Canonical snapshot not found at {SCHEDULE_SNAPSHOT}"
        logger.error(msg)
        return IngestionHealthReport(
            ok=False, errors=[msg], warnings=[], rows=0, unique_games=0, seasons=[]
        )

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    errors: List[str] = []
    warnings: List[str] = []

    expected_cols = {
        "game_id",
        "date",
        "team",
        "opponent",
        "is_home",
        "score",
        "opponent_score",
        "season",
    }

    missing = expected_cols - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    # Check duplicates: each game_id should appear exactly twice
    games_per_id = df.groupby("game_id").size()
    bad_counts = games_per_id[games_per_id != 2]
    if not bad_counts.empty:
        errors.append(
            f"Game IDs not appearing twice: {bad_counts.index.tolist()[:10]} "
            f"(showing first 10)"
        )

    # Check home/away pair consistency
    home_counts = df.groupby("game_id")["is_home"].sum()
    if not (home_counts == 1).all():
        errors.append("Each game_id must have exactly one home team (is_home==1).")

    # Simple score sanity (if played)
    played = df.dropna(subset=["score", "opponent_score"])
    neg_scores = played[(played["score"] < 0) | (played["opponent_score"] < 0)]
    if not neg_scores.empty:
        errors.append("Found negative scores in played games.")

    report = IngestionHealthReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        rows=len(df),
        unique_games=df["game_id"].nunique(),
        seasons=sorted(df["season"].dropna().astype(str).unique().tolist()),
    )

    if report.ok:
        logger.success(
            f"[IngestionHealth] OK — rows={report.rows}, "
            f"games={report.unique_games}, seasons={report.seasons}"
        )
    else:
        logger.error(f"[IngestionHealth] FAILED: {errors}")

    return report
