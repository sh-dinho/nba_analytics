from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Canonicalizer
# File: src/ingestion/normalizer/canonicalizer.py
# Author: Sadiq
#
# Description:
#     Enforces the canonical team-game schema and dtypes for
#     ingestion. Ensures downstream consumers receive a
#     consistent, memory-efficient layout.
# ============================================================

import pandas as pd


CANONICAL_COLUMNS = [
    "game_id",
    "date",
    "team",
    "opponent",
    "is_home",
    "score",
    "opponent_score",
    "season",
    "status",
    "schema_version",
]


def canonicalize_team_game_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and enforce the canonical ingestion schema for
    team-game rows.

    Returns:
        DataFrame with consistent column order and dtypes.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Core identifiers
    out["game_id"] = out["game_id"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["team"] = out["team"].astype(str)
    out["opponent"] = out["opponent"].astype(str)

    # Memory-efficient numeric types
    out["is_home"] = out["is_home"].fillna(0).astype("int8")
    out["score"] = pd.to_numeric(out["score"], errors="coerce").astype("Int16")
    out["opponent_score"] = (
        pd.to_numeric(out["opponent_score"], errors="coerce").astype("Int16")
    )

    # Strings
    out["season"] = out["season"].astype(str)
    out["status"] = out["status"].astype(str)
    out["schema_version"] = out["schema_version"].astype(str)

    # Enforce canonical column order
    return out[CANONICAL_COLUMNS]