from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Ingestion Validation
# File: src/ingestion/validator.py
# Author: Sadiq
#
# Description:
#     Validates canonical team-game rows produced by the
#     ingestion pipeline after normalization.
# ============================================================

import pandas as pd
from loguru import logger


# ------------------------------------------------------------
# Canonical schema for team-game rows (v4)
# ------------------------------------------------------------

REQUIRED_COLUMNS = {
    "game_id",
    "date",
    "team",
    "opponent",
    "is_home",
    "score",
    "opponent_score",
    "season",
}

OPTIONAL_COLUMNS = {
    "status",
    "schema_version",
}


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------


def validate_ingestion_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate canonical team-game rows after normalization.

    v4 rules:
      - Required columns must exist
      - Optional columns allowed to be missing
      - Scores may be NaN for pre-game rows
      - No negative scores
      - No null dates
      - No duplicate (game_id, team)
      - Each game_id must have exactly 2 rows
      - Home/away symmetry must hold
      - Season must be non-null and string-like
    """
    if df.empty:
        return df

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Validator: Missing required columns: {missing}")

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------
    dupes = df[df.duplicated(subset=["game_id", "team"], keep=False)]
    if not dupes.empty:
        logger.warning(
            f"Validator: Duplicate game_id/team rows detected: "
            f"{dupes[['game_id','team']].drop_duplicates().to_dict(orient='records')}"
        )

    # --------------------------------------------------------
    # Score sanity (allow NaN for pre-game)
    # --------------------------------------------------------
    if (df["score"].dropna() < 0).any() or (df["opponent_score"].dropna() < 0).any():
        raise ValueError("Validator: Negative scores detected.")

    # --------------------------------------------------------
    # Date sanity
    # --------------------------------------------------------
    if df["date"].isna().any():
        raise ValueError("Validator: Null dates detected.")

    # --------------------------------------------------------
    # Season sanity
    # --------------------------------------------------------
    if df["season"].isna().any():
        raise ValueError("Validator: Null season values detected.")

    # Season format check (warn only)
    bad_seasons = df[~df["season"].astype(str).str.match(r"^\d{4}-\d{2}$")]
    if not bad_seasons.empty:
        logger.warning(
            f"Validator: Unexpected season format detected: "
            f"{bad_seasons['season'].unique()}"
        )

    # --------------------------------------------------------
    # Game completeness (each game_id must have exactly 2 rows)
    # --------------------------------------------------------
    game_counts = df.groupby("game_id").size()
    bad_games = game_counts[game_counts != 2]

    if not bad_games.empty:
        raise ValueError(
            f"Validator: Incomplete games detected (must have 2 rows): "
            f"{bad_games.to_dict()}"
        )

    # --------------------------------------------------------
    # Home/away symmetry
    # --------------------------------------------------------
    def bad_symmetry(g):
        if len(g) != 2:
            return True
        a, b = g.iloc[0], g.iloc[1]
        return not (a["team"] == b["opponent"] and b["team"] == a["opponent"])

    asym = df.groupby("game_id").apply(bad_symmetry)
    asym = asym[asym]

    if not asym.empty:
        raise ValueError(
            f"Validator: Opponent symmetry errors in games: {asym.index.tolist()}"
        )

    # --------------------------------------------------------
    # Score symmetry (warn only for pre-game rows)
    # --------------------------------------------------------
    def score_mismatch(g):
        if g["score"].isna().any() or g["opponent_score"].isna().any():
            return False  # pre-game rows allowed
        a, b = g.iloc[0], g.iloc[1]
        return not (
            a["score"] == b["opponent_score"] and b["score"] == a["opponent_score"]
        )

    mismatches = df.groupby("game_id").apply(score_mismatch)
    mismatches = mismatches[mismatches]

    if not mismatches.empty:
        logger.warning(
            f"Validator: Score symmetry mismatches detected: {mismatches.index.tolist()}"
        )

    return df
