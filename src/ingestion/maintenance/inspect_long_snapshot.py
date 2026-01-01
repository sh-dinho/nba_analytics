from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Long Snapshot Inspector
# File: src/ingestion/maintenance/inspect_long_snapshot.py
# Author: Sadiq
#
# Description:
#     Diagnostic tool for inspecting the canonical long-format
#     snapshot. Helps detect schema drift, duplicate rows,
#     inconsistent seasons, and other ingestion anomalies.
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT
from src.ingestion.normalizer.canonicalizer import CANONICAL_COLUMNS
from src.ingestion.validator.checks import (
    find_incomplete_games,
    find_asymmetry,
    find_score_mismatches,
)


def inspect_long_snapshot() -> None:
    """
    Print diagnostics about the long-format snapshot.
    """
    if not LONG_SNAPSHOT.exists():
        logger.error("[Inspector] LONG_SNAPSHOT does not exist.")
        return

    df = pd.read_parquet(LONG_SNAPSHOT)
    logger.info(f"[Inspector] Loaded {len(df)} rows from LONG_SNAPSHOT.")

    # ------------------------------------------------------------
    # Basic stats
    # ------------------------------------------------------------
    logger.info(f"[Inspector] Unique dates: {df['date'].nunique()}")
    logger.info(f"[Inspector] Unique games: {df['game_id'].nunique()}")

    # ------------------------------------------------------------
    # Schema drift detection
    # ------------------------------------------------------------
    snapshot_cols = set(df.columns)
    canonical_cols = set(CANONICAL_COLUMNS)

    missing_cols = canonical_cols - snapshot_cols
    extra_cols = snapshot_cols - canonical_cols

    if missing_cols:
        logger.error(f"[Inspector] Missing canonical columns: {missing_cols}")

    if extra_cols:
        logger.warning(f"[Inspector] Extra unexpected columns: {extra_cols}")

    # ------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------
    dup = df.duplicated(subset=["game_id", "team"])
    if dup.any():
        logger.warning(f"[Inspector] Duplicate team-game rows: {dup.sum()}")

    # ------------------------------------------------------------
    # Incomplete games
    # ------------------------------------------------------------
    incomplete = find_incomplete_games(df)
    if not incomplete.empty:
        logger.warning(f"[Inspector] Incomplete games: {dict(incomplete.head(20))}")

    # ------------------------------------------------------------
    # Asymmetry
    # ------------------------------------------------------------
    asym = find_asymmetry(df)
    if len(asym) > 0:
        logger.warning(f"[Inspector] Opponent asymmetry in games: {list(asym[:20])}")

    # ------------------------------------------------------------
    # Score mismatches
    # ------------------------------------------------------------
    mism = find_score_mismatches(df)
    if len(mism) > 0:
        logger.warning(f"[Inspector] Score mismatches: {list(mism[:20])}")

    # ------------------------------------------------------------
    # Schema version sanity check
    # ------------------------------------------------------------
    if "schema_version" in df.columns:
        versions = df["schema_version"].unique().tolist()
        logger.info(f"[Inspector] schema_version values: {versions}")

    logger.success("[Inspector] Inspection complete.")