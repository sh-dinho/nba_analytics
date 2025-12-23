from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Extract Canonical Team Names
# File: src/utils/extract_team_names.py
#
# Description:
#     Reads LONG_SNAPSHOT and prints all unique team names
#     exactly as they appear in your ingestion pipeline.
#
#     This ensures we build a perfect normalization map
#     (ESPN → NBA Stats → Your Canonical Names).
# ============================================================

import pandas as pd
from loguru import logger
from src.config.paths import LONG_SNAPSHOT


def extract_team_names() -> list[str]:
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(
            f"LONG_SNAPSHOT not found at {LONG_SNAPSHOT}. " f"Run ingestion first."
        )

    df = pd.read_parquet(LONG_SNAPSHOT)

    if "team" not in df.columns:
        raise KeyError(
            f"'team' column not found in LONG_SNAPSHOT. "
            f"Columns available: {list(df.columns)}"
        )

    teams = sorted(df["team"].dropna().unique().tolist())

    logger.info(f"Found {len(teams)} unique team names in LONG_SNAPSHOT:")
    for t in teams:
        logger.info(f"  - {t}")

    return teams


if __name__ == "__main__":
    extract_team_names()
