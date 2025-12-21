# ============================================================
# File: src/ingestion/normalizer.py
# Purpose: Clean and normalize raw NBA data to canonical schema
# Version: 4.1 (Automated, no manual maps)
# ============================================================

from __future__ import annotations

import re
from typing import Optional, Tuple

import pandas as pd
from loguru import logger


def _extract_abbrevs_from_matchup(matchup: str) -> Optional[Tuple[str, str]]:
    """
    Extract (away_abbrev, home_abbrev) from MATCHUP string.

    Examples:
        "LAL @ BOS"   -> ("LAL", "BOS")
        "BOS vs. LAL" -> ("LAL", "BOS")
    """
    if not isinstance(matchup, str):
        return None

    m = matchup.replace(".", "")
    m = re.sub(r"\(.*?\)", "", m).strip()  # remove OT etc
    m = re.sub(r"\s+", " ", m)  # normalize spaces

    parts = m.split(" ")
    if len(parts) != 3:
        return None

    left, sep, right = parts
    left = left.strip()
    right = right.strip()
    sep = sep.strip().lower()

    if sep == "@":
        away_abbrev, home_abbrev = left, right
    elif sep == "vs":
        home_abbrev, away_abbrev = left, right
    else:
        return None

    # Abbrevs are typically uppercase 2–4 chars, but be tolerant
    if not left or not right:
        return None

    return away_abbrev, home_abbrev


def _infer_season(d: pd.Timestamp) -> int:
    """
    Infer NBA season label from date.
    Example: games in Oct 2024–Jun 2025 → season 2024.
    """
    year = d.year
    if d.month >= 8:  # Aug–Dec: start of new season
        return year
    return year - 1


def normalize_schedule(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw schedule DataFrame to canonical format.

    Expected raw columns (minimum):
        GAME_ID    : unique game identifier
        GAME_DATE  : date string or datetime
        TEAM_NAME  : full team name (per team row)
        MATCHUP    : NBA-style matchup string (e.g. "LAL @ BOS", "BOS vs. LAL")

    Strongly preferred (if available from NBA stats API):
        TEAM_ID
        TEAM_ABBREVIATION
        PTS
        season_type

    Output schema (one row per game_id):
        game_id      : str
        date         : datetime64[ns]
        season       : int
        season_type  : str
        home_team    : str
        away_team    : str
        home_score   : float (NaN if not final)
        away_score   : float (NaN if not final)
        status       : {"scheduled","final","unknown"}
    """
    cols_out = [
        "game_id",
        "date",
        "season",
        "season_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status",
    ]

    if df_raw is None or df_raw.empty:
        logger.info("normalize_schedule(): received empty DataFrame.")
        return pd.DataFrame(columns=cols_out)

    required = {"GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"normalize_schedule(): Missing required columns: {missing}")

    df = df_raw.copy()

    # ----------------------------------------------------------------
    # Parse and validate date
    # ----------------------------------------------------------------
    df["date"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    invalid_dates = df[df["date"].isna()]
    if not invalid_dates.empty:
        logger.warning(
            f"normalize_schedule(): dropping {len(invalid_dates)} rows with invalid GAME_DATE"
        )
        df = df.dropna(subset=["date"])

    if df.empty:
        logger.warning("normalize_schedule(): all rows dropped due to invalid dates.")
        return pd.DataFrame(columns=cols_out)

    df["GAME_ID"] = df["GAME_ID"].astype(str)

    # ----------------------------------------------------------------
    # Try to use TEAM_ABBREVIATION if present, else derive from MATCHUP
    # ----------------------------------------------------------------
    has_abbrev = "TEAM_ABBREVIATION" in df.columns
    if has_abbrev:
        df["team_abbrev"] = df["TEAM_ABBREVIATION"].astype(str)
    else:
        # Derive abbrev from MATCHUP and TEAM_NAME position (fallback)
        # For each row, take the token in MATCHUP that matches the TEAM_NAME position.
        # This is heuristic but fully automated.
        def derive_abbrev(row):
            matchup = row["MATCHUP"]
            if not isinstance(matchup, str):
                return None
            m = matchup.replace(".", "")
            m = re.sub(r"\(.*?\)", "", m).strip()
            parts = m.split(" ")
            if len(parts) != 3:
                return None
            left, _, right = parts
            # We don't know which side is this team. Try matching TEAM_NAME substring.
            name = str(row["TEAM_NAME"])
            if name.split(" ")[-1] in left or name[:3].upper() == left.upper():
                return left
            if name.split(" ")[-1] in right or name[:3].upper() == right.upper():
                return right
            return None

        df["team_abbrev"] = df.apply(derive_abbrev, axis=1)

    # Parse matchup into away/home abbrevs
    parsed_abbrevs = df["MATCHUP"].apply(_extract_abbrevs_from_matchup)
    bad_matchups = parsed_abbrevs.isna().sum()
    if bad_matchups:
        logger.warning(
            f"normalize_schedule(): {bad_matchups} rows with unparseable MATCHUP; they will be dropped."
        )

    df["away_abbrev"] = parsed_abbrevs.apply(lambda x: x[0] if x else None)
    df["home_abbrev"] = parsed_abbrevs.apply(lambda x: x[1] if x else None)

    df = df.dropna(subset=["away_abbrev", "home_abbrev"])

    if df.empty:
        logger.warning(
            "normalize_schedule(): all rows dropped due to bad MATCHUP parsing."
        )
        return pd.DataFrame(columns=cols_out)

    # ----------------------------------------------------------------
    # Assign home/away team names automatically
    # ----------------------------------------------------------------
    # For each game_id, pick the TEAM_NAME corresponding to the home_abbrev / away_abbrev
    def pick_team_name(group: pd.DataFrame, target_abbrev: str) -> str:
        # prefer row whose team_abbrev matches target_abbrev, fall back to any row containing it in MATCHUP
        exact = group[group["team_abbrev"].str.upper() == target_abbrev.upper()]
        if not exact.empty:
            return exact["TEAM_NAME"].iloc[0]

        # fallback: match via MATCHUP tokens
        m = group[group["MATCHUP"].str.contains(target_abbrev, na=False)]
        if not m.empty:
            return m["TEAM_NAME"].iloc[0]

        # final fallback: just pick first TEAM_NAME (should be rare)
        return group["TEAM_NAME"].iloc[0]

    grouped = df.groupby("GAME_ID", group_keys=False)

    home_names = grouped.apply(lambda g: pick_team_name(g, g["home_abbrev"].iloc[0]))
    away_names = grouped.apply(lambda g: pick_team_name(g, g["away_abbrev"].iloc[0]))

    # ----------------------------------------------------------------
    # Scores (if present)
    # ----------------------------------------------------------------
    if "PTS" in df.columns:
        df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
    else:
        df["PTS"] = pd.NA

    def agg_scores(group: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        home_abbrev = group["home_abbrev"].iloc[0]
        away_abbrev = group["away_abbrev"].iloc[0]

        # home rows: team_abbrev matches home_abbrev
        home_rows = group[group["team_abbrev"].str.upper() == home_abbrev.upper()]
        away_rows = group[group["team_abbrev"].str.upper() == away_abbrev.upper()]

        home_score = pd.to_numeric(home_rows["PTS"], errors="coerce").max()
        away_score = pd.to_numeric(away_rows["PTS"], errors="coerce").max()

        if pd.isna(home_score):
            home_score = None
        if pd.isna(away_score):
            away_score = None

        return home_score, away_score

    scores = grouped.apply(agg_scores)
    home_scores = scores.apply(lambda t: t[0] if isinstance(t, tuple) else None)
    away_scores = scores.apply(lambda t: t[1] if isinstance(t, tuple) else None)

    # ----------------------------------------------------------------
    # Build canonical per-game DataFrame
    # ----------------------------------------------------------------
    game_dates = grouped["date"].first()

    df_clean = pd.DataFrame(
        {
            "game_id": game_dates.index.astype(str),
            "date": game_dates.values,
            "home_team": home_names.values,
            "away_team": away_names.values,
            "home_score": pd.to_numeric(home_scores, errors="coerce"),
            "away_score": pd.to_numeric(away_scores, errors="coerce"),
        }
    )

    # season_type if present, else "unknown"
    if "season_type" in df.columns:
        df_clean["season_type"] = grouped["season_type"].first().reindex(df_clean.index)
    else:
        df_clean["season_type"] = "unknown"

    df_clean["season"] = df_clean["date"].apply(_infer_season).astype("int64")

    # ----------------------------------------------------------------
    # Status
    # ----------------------------------------------------------------
    def infer_status(row) -> str:
        if pd.notnull(row["home_score"]) and pd.notnull(row["away_score"]):
            return "final"
        return "scheduled"

    df_clean["status"] = df_clean.apply(infer_status, axis=1)

    # Final sanity checks
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["home_team", "away_team"])
    dropped = before - len(df_clean)
    if dropped:
        logger.warning(
            f"normalize_schedule(): dropped {dropped} games with missing home/away teams after normalization."
        )

    df_clean = df_clean[
        [
            "game_id",
            "date",
            "season",
            "season_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "status",
        ]
    ]

    return df_clean
