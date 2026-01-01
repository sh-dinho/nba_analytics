from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Validate Ingestion Team Names (Canonical)
# File: src/utils/validate_ingestion_team_names.py
# Author: Sadiq
#
# Description:
#     Checks any ingestion DataFrame for unknown team names
#     using the canonical normalization map.
#
#     Enhancements:
#       • multi-column support (team, opponent, home_team, away_team)
#       • case-insensitive + accent-insensitive normalization
#       • structured return format
#       • optional strict mode for ingestion QA
# ============================================================

import pandas as pd
from loguru import logger
from src.utils.team_names import normalize_team


def validate_ingestion_team_names(
    df: pd.DataFrame,
    columns=("team",),
    strict: bool = False,
) -> list[str]:
    """
    Validate team names in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing team names.
    columns : tuple[str]
        Columns to validate. Default: ("team",)
    strict : bool
        If True, raise an exception when unknown names are found.

    Returns
    -------
    list[str]
        Sorted list of unknown team name variants.
    """

    unknown = set()

    for col in columns:
        if col not in df.columns:
            logger.debug(f"[TeamNameValidation] Column '{col}' not found, skipping.")
            continue

        raw_values = df[col].dropna().unique()

        for name in raw_values:
            if normalize_team(name) is None:
                unknown.add(name)

    # Logging summary
    if unknown:
        logger.warning(f"[TeamNameValidation] Found {len(unknown)} unknown team names:")
        for u in sorted(unknown):
            logger.warning(f"  - {u}")

        if strict:
            raise ValueError(
                f"Unknown team names detected in strict mode: {sorted(unknown)}"
            )
    else:
        logger.info("[TeamNameValidation] All team names normalized successfully.")

    return sorted(unknown)


if __name__ == "__main__":
    print("This script validates team names in ingestion DataFrames.")