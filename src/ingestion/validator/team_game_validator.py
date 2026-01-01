from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Team-Game Validator
# File: src/ingestion/validator/team_game_validator.py
# Author: Sadiq
#
# Description:
#     Strict validator for canonical team-game rows in ingestion.
#     Enforces core invariants (2 rows per game, symmetry,
#     no negative scores, no null dates/seasons) and produces a
#     structured ValidationReport for logging and monitoring.
# ============================================================

from dataclasses import dataclass
from typing import List

import pandas as pd
from loguru import logger

from src.ingestion.validator.checks import (
    find_incomplete_games,
    find_negative_scores,
    find_null_dates,
    find_null_seasons,
    find_asymmetry,
    find_score_mismatches,
)


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


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str]
    warnings: List[str]


def validate_team_game_df(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> ValidationReport:
    """
    Validate canonical team-game rows.

    - Required columns must exist.
    - No negative scores.
    - No null dates or seasons.
    - Each game_id must have exactly 2 rows.
    - Home/away symmetry must hold.
    - Score symmetry checked but only warned on.
    """
    if df.empty:
        return ValidationReport(ok=True, errors=[], warnings=["Empty DataFrame."])

    errors: List[str] = []
    warnings: List[str] = []

    # Required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Basic invariants
    if find_negative_scores(df):
        errors.append("Negative scores detected.")

    if find_null_dates(df):
        errors.append("Null dates detected.")

    if find_null_seasons(df):
        errors.append("Null season values detected.")

    # Game-level invariants
    bad_games = find_incomplete_games(df)
    if not bad_games.empty:
        errors.append(
            f"Incomplete games (must have exactly 2 rows): {dict(bad_games.head(20))}"
        )

    asym = find_asymmetry(df)
    if len(asym) > 0:
        errors.append(f"Opponent symmetry errors in games: {list(asym[:20])}")

    mismatches = find_score_mismatches(df)
    if len(mismatches) > 0:
        warnings.append(
            f"Score symmetry mismatches detected (ignoring pre-game rows): "
            f"{list(mismatches[:20])}"
        )

    ok = len(errors) == 0
    report = ValidationReport(ok=ok, errors=errors, warnings=warnings)

    # Logging
    if ok:
        logger.success("[Validator] Team-game DataFrame OK.")
    else:
        logger.error(f"[Validator] FAILED: {errors}")

    for w in warnings:
        logger.warning(f"[Validator] {w}")

    if not ok and raise_on_error:
        raise ValueError(f"Team-game validation failed: {errors}")

    return report