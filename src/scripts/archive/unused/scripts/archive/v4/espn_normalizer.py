from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: ESPN Schedule Normalizer
# File: src/ingestion/espn_normalizer.py
# Author: Sadiq
# ============================================================

from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger

# If you already have shared team normalization utilities,
# you can import and reuse them here.
# Example:
# from src.utils.team_names import to_tricode

DEFAULT_SOURCE = "espn"


def _infer_season_from_date(d: date) -> str:
    """
    Infer NBA season string from a given date.

    - If month >= 10, season is YYYY-YY where YYYY is the current year
    - If month < 10, season is (YYYY-1)-YY where YYYY is the current year
    """
    if d.month >= 10:
        start_year = d.year
    else:
        start_year = d.year - 1
    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


def _normalize_team_name(raw_name: str) -> str:
    """
    Normalize raw ESPN team name to canonical representation.

    For now this is an identity function; wire it to your
    team-name → tricode mapping as needed.
    """
    if raw_name is None:
        return ""
    name = raw_name.strip()

    # TODO: integrate with your canonical team name / tricode mapping.
    # Example:
    # return to_tricode(name)

    return name


def normalize_espn_schedule(
    df_raw: pd.DataFrame, source: str = DEFAULT_SOURCE
) -> pd.DataFrame:
    """
    Normalize ESPN scraped schedule into a stable, wide-format schedule table.

    Expected input columns (from ESPN scraper):
        - date        (datetime or date)
        - home_team   (raw ESPN home team name)
        - away_team   (raw ESPN away team name)
        - game_time   (string, optional)

    Output schema:
        - game_id      (synthetic, stable)
        - date         (date)
        - home_team    (canonical name / tricode)
        - away_team    (canonical name / tricode)
        - game_time    (string, as scraped)
        - season       (season label, e.g. "2024-25")
        - source       (default "espn")
    """
    if df_raw.empty:
        logger.warning("[ESPNNormalizer] Received empty DataFrame.")
        return df_raw

    required_cols = {"date", "home_team", "away_team"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        logger.error(
            f"[ESPNNormalizer] Missing required columns: {sorted(missing)}. "
            f"Columns present: {df_raw.columns.tolist()}"
        )
        return pd.DataFrame()

    df = df_raw.copy()

    # Ensure date is datetime.date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "home_team", "away_team"])
    if df.empty:
        logger.warning("[ESPNNormalizer] All rows dropped after date/team coercion.")
        return df

    # Normalize team names
    df["home_team"] = df["home_team"].astype(str).apply(_normalize_team_name)
    df["away_team"] = df["away_team"].astype(str).apply(_normalize_team_name)

    # Infer season per row
    df["season"] = df["date"].apply(_infer_season_from_date)

    # Synthetic, stable game_id
    # Use date + away_team + home_team; relies on canonicalized names
    df["game_id"] = df.apply(
        lambda r: f"{r['date'].strftime('%Y%m%d')}_{r['away_team']}_{r['home_team']}",
        axis=1,
    )

    # Ensure game_time exists (even if None)
    if "game_time" not in df.columns:
        df["game_time"] = None

    # Tag source
    df["source"] = source

    # Select and order columns for stability
    df = df[
        [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "game_time",
            "season",
            "source",
        ]
    ]

    df = df.sort_values(["date", "game_time", "home_team", "away_team"]).reset_index(
        drop=True
    )

    logger.info(
        f"[ESPNNormalizer] Normalized ESPN schedule rows: {len(df)} "
        f"(seasons={sorted(df['season'].unique())})"
    )

    return df
