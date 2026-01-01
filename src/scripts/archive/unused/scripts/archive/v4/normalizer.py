from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Ingestion Normalizer
# File: src/ingestion/normalizer.py
# Author: Sadiq
# ============================================================

import pandas as pd
from loguru import logger


# ------------------------------------------------------------
# Column alias mapping for ScoreboardV3 variations
# ------------------------------------------------------------

ALIASES = {
    "date": ["gameDateEst", "gameDate"],
    "home_team": ["homeTeamName"],
    "away_team": ["awayTeamName", "visitorTeamName"],
    "home_score": ["homeScore", "homeTeamScore"],
    "away_score": ["awayScore", "visitorTeamScore"],
}

# NEW: minimal schema fallback
MINIMAL_SCHEMA = {
    "game_id",
    "date",
    "home_team",
    "away_team",
    "score_home",
    "score_away",
}


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ------------------------------------------------------------
# Main normalizer
# ------------------------------------------------------------


def normalize_schedule(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ScoreboardV3 raw data into canonical TEAM-GAME rows.
    Output schema:
        game_id
        date
        team
        opponent
        is_home
        score
        opponent_score
        season
        status
        schema_version
    """
    if df_raw.empty:
        return df_raw

    # Must contain gameId
    if "gameId" not in df_raw.columns:
        logger.error(
            "Normalizer received non-game DataFrame. "
            f"Columns={df_raw.columns.tolist()}"
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Try alias-based resolution (legacy/modern ScoreboardV3)
    # --------------------------------------------------------
    date_col = _first_existing(df_raw, ALIASES["date"])
    home_team_col = _first_existing(df_raw, ALIASES["home_team"])
    away_team_col = _first_existing(df_raw, ALIASES["away_team"])
    home_score_col = _first_existing(df_raw, ALIASES["home_score"])
    away_score_col = _first_existing(df_raw, ALIASES["away_score"])

    # If alias resolution fails, try minimal schema fallback
    if home_team_col is None or away_team_col is None:
        if MINIMAL_SCHEMA.issubset(df_raw.columns):
            logger.warning(
                "Normalizer: using MINIMAL scoreboard schema fallback "
                "(score_home / score_away)."
            )

            wide = pd.DataFrame(
                {
                    "game_id": df_raw["gameId"].astype(str),
                    "date": pd.to_datetime(df_raw["date"], errors="coerce").dt.date,
                    "home_team": df_raw["home_team"],
                    "away_team": df_raw["away_team"],
                    "home_score": pd.to_numeric(df_raw["score_home"], errors="coerce"),
                    "away_score": pd.to_numeric(df_raw["score_away"], errors="coerce"),
                    "status": df_raw.get("gameStatusText"),
                    "schema_version": "minimal",
                }
            )

        else:
            logger.error(
                "Normalizer could not resolve required fields via aliases or minimal schema. "
                f"Columns={df_raw.columns.tolist()}"
            )
            return pd.DataFrame()

    else:
        # --------------------------------------------------------
        # Build wide canonical rows (alias-based)
        # --------------------------------------------------------
        wide = pd.DataFrame(
            {
                "game_id": df_raw["gameId"].astype(str),
                "date": pd.to_datetime(df_raw[date_col], errors="coerce").dt.date,
                "home_team": df_raw[home_team_col],
                "away_team": df_raw[away_team_col],
                "home_score": (
                    pd.to_numeric(df_raw[home_score_col], errors="coerce")
                    if home_score_col
                    else pd.NA
                ),
                "away_score": (
                    pd.to_numeric(df_raw[away_score_col], errors="coerce")
                    if away_score_col
                    else pd.NA
                ),
                "status": df_raw.get("gameStatusText"),
                "schema_version": df_raw.get("schema_version"),
            }
        )

    wide = wide.dropna(subset=["game_id", "date", "home_team", "away_team"])
    if wide.empty:
        return wide

    # --------------------------------------------------------
    # Convert wide → team-game rows
    # --------------------------------------------------------

    home_rows = pd.DataFrame(
        {
            "game_id": wide["game_id"],
            "date": wide["date"],
            "team": wide["home_team"],
            "opponent": wide["away_team"],
            "is_home": 1,
            "score": wide["home_score"],
            "opponent_score": wide["away_score"],
            "status": wide["status"],
            "schema_version": wide["schema_version"],
        }
    )

    away_rows = pd.DataFrame(
        {
            "game_id": wide["game_id"],
            "date": wide["date"],
            "team": wide["away_team"],
            "opponent": wide["home_team"],
            "is_home": 0,
            "score": wide["away_score"],
            "opponent_score": wide["home_score"],
            "status": wide["status"],
            "schema_version": wide["schema_version"],
        }
    )

    df = pd.concat([home_rows, away_rows], ignore_index=True)

    # --------------------------------------------------------
    # Season inference
    # --------------------------------------------------------
    df["season"] = df["date"].apply(
        lambda d: (
            f"{d.year}-{str(d.year + 1)[-2:]}"
            if d.month >= 10
            else f"{d.year - 1}-{str(d.year)[-2:]}"
        )
    )

    return df
