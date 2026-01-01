from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Long Snapshot Inspector
# File: src/ingestion/maintenance/inspect_long_snapshot.py
# Author: Sadiq
#
# Description:
#     Diagnostic tool for inspecting the canonical long-format
#     snapshot. Detects schema drift, duplicate rows, missing
#     values, asymmetry, score mismatches, and ingestion anomalies.
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT
from src.ingestion.validator.checks import (
    find_incomplete_games,
    find_asymmetry,
    find_score_mismatches,
)


def inspect_long_snapshot() -> dict:
    logger.info("=== Inspecting LONG_SNAPSHOT ===")
    logger.info(f"Path: {LONG_SNAPSHOT}")

    if not LONG_SNAPSHOT.exists():
        logger.error("LONG_SNAPSHOT does not exist.")
        return {"ok": False, "error": "Snapshot missing"}

    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ------------------------------------------------------------
    # Column categories
    # ------------------------------------------------------------
    columns = df.columns.tolist()

    identity_cols = [
        c for c in columns
        if any(key in c.lower() for key in
               ["id", "team", "opponent", "date", "home", "away", "status"])
    ]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in columns if c not in numeric_cols]
    nan_cols = df.columns[df.isna().any()].tolist()

    # ------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------
    incomplete = find_incomplete_games(df)
    asym = find_asymmetry(df)
    mismatches = find_score_mismatches(df)

    duplicate_rows = int(df.duplicated(subset=["game_id", "team"]).sum())

    # Season inconsistencies (only if season exists)
    if "season" in df.columns:
        season_incons = (
            df.groupby("game_id")["season"]
            .nunique()
            .loc[lambda s: s > 1]
            .index.tolist()
        )
    else:
        season_incons = []

    # Cross-date duplicates
    cross_date = (
        df.groupby("game_id")["date"]
        .nunique()
        .loc[lambda s: s > 1]
        .index.tolist()
    )

    # ------------------------------------------------------------
    # Build structured report
    # ------------------------------------------------------------
    report = {
        "ok": True,
        "rows": len(df),
        "columns": columns,
        "identity_columns": identity_cols,
        "numeric_columns": numeric_cols,
        "non_numeric_columns": non_numeric_cols,
        "nan_columns": nan_cols,
        "duplicate_team_game_rows": duplicate_rows,
        "incomplete_games": list(incomplete.index),
        "asymmetry_games": list(asym),
        "score_mismatches": list(mismatches),
        "season_inconsistencies": season_incons,
        "cross_date_duplicates": cross_date,
    }

    if (
        duplicate_rows > 0
        or len(incomplete) > 0
        or len(asym) > 0
        or len(season_incons) > 0
        or len(cross_date) > 0
    ):
        report["ok"] = False

    # ------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total rows: {report['rows']}")
    logger.info(f"Total columns: {len(columns)}")
    logger.info(f"Numeric columns: {len(numeric_cols)}")
    logger.info(f"Non-numeric columns: {len(non_numeric_cols)}")
    logger.info(f"Identity-like columns: {len(identity_cols)}")
    logger.info(f"Columns with NaNs: {nan_cols}")

    if duplicate_rows > 0:
        logger.warning(f"Duplicate team-game rows: {duplicate_rows}")

    if len(incomplete) > 0:
        logger.warning(f"Incomplete games: {list(incomplete.index[:20])}")

    if len(asym) > 0:
        logger.warning(f"Opponent asymmetry: {list(asym[:20])}")

    if len(mismatches) > 0:
        logger.warning(f"Score mismatches: {list(mismatches[:20])}")

    if len(season_incons) > 0:
        logger.warning(f"Season inconsistencies: {season_incons[:20]}")

    if len(cross_date) > 0:
        logger.warning(f"Cross-date duplicates: {cross_date[:20]}")

    logger.info("\nRecommended identity columns to drop before prediction:")
    logger.info(identity_cols)

    return report


if __name__ == "__main__":
    inspect_long_snapshot()