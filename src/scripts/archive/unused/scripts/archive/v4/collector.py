from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Collector (ScoreboardV3 Fetcher)
# File: src/ingestion/collector.py
# Author: Sadiq
# ============================================================

import time
from datetime import date

import pandas as pd
from loguru import logger

from nba_api.stats.endpoints import ScoreboardV3
from nba_api.stats.library.http import NBAStatsHTTP


# ------------------------------------------------------------
# Lazy + resilient session initialization
# ------------------------------------------------------------

_session = None


def _get_session():
    """
    Initialize NBAStatsHTTP session lazily and safely.
    Never raise on import. Always return a usable session.
    """
    global _session
    if _session is not None:
        return _session

    try:
        http = NBAStatsHTTP()

        # Try common session attributes across versions
        if hasattr(http, "_session") and http._session is not None:
            _session = http._session
        elif hasattr(http, "_init_session"):
            _session = http._init_session()
        elif hasattr(http, "session"):
            _session = http.session
        else:
            logger.warning(
                "NBAStatsHTTP session attribute not found; using fallback session"
            )
            _session = NBAStatsHTTP().session

    except Exception as e:
        logger.error(f"NBAStatsHTTP initialization failed: {e}")
        logger.warning("Falling back to a plain requests.Session()")
        import requests

        _session = requests.Session()

    # Update headers to avoid Cloudflare blocks
    _session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nba.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    return _session


# ------------------------------------------------------------
# Retry wrapper
# ------------------------------------------------------------


def _retry(func, retries: int = 3, base_delay: float = 1.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            logger.warning(
                f"Retry {attempt + 1}/{retries} failed: {e}. Sleeping {base_delay}s."
            )
            time.sleep(base_delay)
            base_delay *= 2

    logger.error("All retries failed. Returning empty DataFrame.")
    return pd.DataFrame()


# ------------------------------------------------------------
# Schema detection
# ------------------------------------------------------------


def detect_schema(df: pd.DataFrame) -> str:
    cols = set(df.columns)

    if {"homeTeamScore", "visitorTeamScore"} & cols:
        return "v3_legacy"
    if {"homeScore", "awayScore"} & cols:
        return "v3_modern"
    if "games" in cols:
        return "v3_header"

    return "unknown"


# ------------------------------------------------------------
# Core fetch
# ------------------------------------------------------------


def fetch_scoreboard(target_date: date) -> pd.DataFrame:
    """
    Fetch ScoreboardV3 for a given date.
    Uses lazy session initialization and retry logic.
    """

    def _fetch():
        session = _get_session()  # noqa: F841 (ensures headers/session set up)

        logger.debug(f"Fetching ScoreboardV3 for {target_date}")

        sb = ScoreboardV3(
            game_date=target_date.strftime("%Y-%m-%d"),
            timeout=10,
        )

        frames = sb.get_data_frames()
        if not frames:
            raise ValueError("ScoreboardV3 returned no tables")

        df = frames[0]

        logger.debug(f"Columns returned for {target_date}: {list(df.columns)}")

        # --------------------------------------------------------
        # Schema drift guard — never fatal
        # --------------------------------------------------------
        if "gameId" not in df.columns:
            logger.warning(
                f"ScoreboardV3: 'gameId' missing for {target_date}. "
                "Skipping date safely."
            )
            return pd.DataFrame()

        valid_ids = (
            df.get("gameId").astype(str).str.strip().replace("None", pd.NA).dropna()
        )

        if valid_ids.empty:
            logger.warning(
                f"ScoreboardV3: No valid gameId values for {target_date}. "
                "Skipping date safely."
            )
            return pd.DataFrame()

        schema = detect_schema(df)
        logger.info(f"📐 ScoreboardV3 schema detected for {target_date}: {schema}")

        df["fetch_date"] = target_date
        df["schema_version"] = schema
        return df

    df_raw = _retry(_fetch)

    if df_raw is None or df_raw.empty:
        logger.debug(f"No games returned for {target_date}")
        return pd.DataFrame()

    logger.debug(f"Fetched {len(df_raw)} rows for {target_date}")
    return df_raw
