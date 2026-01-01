from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Team Name Normalization Report (Canonical)
# File: src/utils/team_name_normalization_report.py
# Author: Sadiq
#
# Description:
#     Generates a detailed report of:
#       • all unique team names in a DataFrame
#       • their normalized tricodes
#       • unknown names
#       • frequency counts
#
#     Optional:
#       • strict mode (raise on unknowns)
#       • multi-column mode (team + opponent)
#       • sorting options
# ============================================================

import pandas as pd
from loguru import logger
from src.utils.team_names import normalize_team


def team_name_normalization_report(
    df: pd.DataFrame,
    columns: list[str] | str = "team",
    strict: bool = False,
    sort_by: str = "count",   # "count", "raw_name", "normalized"
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
        raw_name | normalized | is_unknown | count

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing team names.
    columns : str or list[str]
        Column(s) to analyze. Default: "team".
    strict : bool
        If True, raise an exception when unknown names are found.
    sort_by : str
        Sorting key: "count", "raw_name", or "normalized".
    """

    # Normalize input to list
    if isinstance(columns, str):
        columns = [columns]

    # Validate columns
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # Collect all names from all columns
    names = pd.concat([df[col].dropna() for col in columns], ignore_index=True)

    # Build frequency table
    report = (
        names.value_counts()
        .rename("count")
        .reset_index()
        .rename(columns={"index": "raw_name"})
    )

    # Apply normalization
    report["normalized"] = report["raw_name"].apply(normalize_team)
    report["is_unknown"] = report["normalized"].isna()

    # Logging
    unknown_count = report["is_unknown"].sum()
    logger.info(f"Found {len(report)} unique team name variants across {columns}.")
    logger.info(f"Unknown variants: {unknown_count}")

    if unknown_count > 0:
        logger.warning("Unknown team names detected:")
        for name in report[report["is_unknown"]]["raw_name"]:
            logger.warning(f"  - {name}")

        if strict:
            raise ValueError("Unknown team names detected in strict mode.")

    # Sorting
    if sort_by == "count":
        report = report.sort_values("count", ascending=False)
    elif sort_by == "raw_name":
        report = report.sort_values("raw_name")
    elif sort_by == "normalized":
        report = report.sort_values("normalized")
    else:
        logger.warning(f"Unknown sort key '{sort_by}', skipping sorting.")

    return report.reset_index(drop=True)


if __name__ == "__main__":
    print("This script generates a normalization report for team names.")