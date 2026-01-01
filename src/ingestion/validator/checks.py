from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Ingestion Validation Checks
# File: src/ingestion/validator/checks.py
# Author: Sadiq (Optimized)
#
# Description:
#     Vectorized validation helpers for canonical team-game rows.
#     Used by the ingestion validator to enforce structural and
#     statistical invariants.
# ============================================================

import pandas as pd


def find_incomplete_games(df: pd.DataFrame) -> pd.Series:
    """Games must have exactly 2 rows (home + away)."""
    counts = df.groupby("game_id").size()
    return counts[counts != 2]


def find_negative_scores(df: pd.DataFrame) -> bool:
    """Detect negative score values."""
    sc = df["score"].dropna()
    osc = df["opponent_score"].dropna()
    return (sc < 0).any() or (osc < 0).any()


def find_null_dates(df: pd.DataFrame) -> bool:
    """Detect missing date values."""
    return df["date"].isna().any()


def find_null_seasons(df: pd.DataFrame) -> bool:
    """Detect missing season labels."""
    return df["season"].isna().any()


def find_asymmetry(df: pd.DataFrame) -> pd.Index:
    """
    Vectorized detection of team/opponent asymmetry.
    Aligned rows must satisfy: Team A's opponent == Team B and vice versa.
    """
    if df.empty:
        return pd.Index([])

    # Filter for games with exactly two rows
    counts = df.groupby("game_id")["team"].transform("count")
    df_pairs = df[counts == 2].sort_values(["game_id", "is_home"])

    home_rows = df_pairs.iloc[::2].reset_index(drop=True)
    away_rows = df_pairs.iloc[1::2].reset_index(drop=True)

    mismatch = (
        (home_rows["team"] != away_rows["opponent"]) |
        (away_rows["team"] != home_rows["opponent"])
    )

    return pd.Index(home_rows.loc[mismatch, "game_id"].unique())


def find_score_mismatches(df: pd.DataFrame) -> pd.Index:
    """
    Vectorized score symmetry: home score == away opponent_score.
    """
    df_scores = df.dropna(subset=["score", "opponent_score"])
    if df_scores.empty:
        return pd.Index([])

    counts = df_scores.groupby("game_id")["team"].transform("count")
    df_pairs = df_scores[counts == 2].sort_values(["game_id", "is_home"])

    home_rows = df_pairs.iloc[::2].reset_index(drop=True)
    away_rows = df_pairs.iloc[1::2].reset_index(drop=True)

    mismatch = (
        (home_rows["score"] != away_rows["opponent_score"]) |
        (away_rows["score"] != home_rows["opponent_score"])
    )

    return pd.Index(home_rows.loc[mismatch, "game_id"].unique())