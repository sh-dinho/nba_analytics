# ============================================================
# File: src/schemas/normalize.py
# Purpose: Schema normalization to enriched_schedule (v2.0)
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pandas as pd


def normalize(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    if schema != "enriched_schedule":
        raise ValueError("Only 'enriched_schedule' schema supported in v2.0.")
    df = df.copy()

    # Enforce canonical column names and types
    required = [
        "gameId",
        "seasonYear",
        "startDate",
        "homeTeam",
        "awayTeam",
        "homeScore",
        "awayScore",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    for c in ["homeScore", "awayScore"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
