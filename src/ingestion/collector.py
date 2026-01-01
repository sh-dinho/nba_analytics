from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Scoreboard Collector
# File: src/ingestion/collector.py
# Author: Sadiq
#
# Description:
#     Fetches raw NBA scoreboard data for a given date using the
#     NBA API (ScoreboardV3). Handles:
#       - retry logic
#       - malformed JSON handling
#       - schema detection (modern / legacy)
#       - safe DataFrame construction
#       - logging and error handling
#
#     Output:
#         A raw DataFrame with whatever columns the API returns.
#         Normalization is handled by scoreboard_normalizer.py.
# ============================================================

from datetime import date
from typing import Optional

import pandas as pd
import requests
from loguru import logger
import time

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

NBA_SCOREBOARD_URL = (
    "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_{}.json"
)

NBA_SCOREBOARD_URL_LEGACY = (
    "https://data.nba.net/prod/v1/{}/scoreboard.json"
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _format_date(d: date) -> str:
    """Return YYYYMMDD string."""
    return d.strftime("%Y%m%d")


def _safe_request(url: str, retries: int = 5, timeout: int = 10) -> Optional[dict]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()

            # 429 = rate limit, 403 = temporary ban
            if resp.status_code in [403, 429]:
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")

    return None


# ------------------------------------------------------------
# Main public function
# ------------------------------------------------------------

def fetch_scoreboard_for_date(day: date) -> pd.DataFrame:
    """
    Fetch raw scoreboard data for a given date.

    Returns:
        pd.DataFrame with raw scoreboard rows.
        If no games or API failure → empty DataFrame.
    """
    logger.info(f"[Collector] Fetching ScoreboardV3 for {day}...")

    day_str = _format_date(day)
    url = NBA_SCOREBOARD_URL.format(day_str)

    data = _safe_request(url)
    if data is None:
        logger.error(f"[Collector] ScoreboardV3 failed for {day}. Trying legacy endpoint...")
        return _fetch_legacy_scoreboard(day)

    if not isinstance(data, dict) or "scoreboard" not in data:
        logger.error(f"[Collector] Malformed ScoreboardV3 JSON for {day}.")
        return _fetch_legacy_scoreboard(day)

    try:
        games = data.get("scoreboard", {}).get("games", [])
        if not games:
            logger.warning(f"[Collector] No games found for {day}.")
            return pd.DataFrame()

        df = pd.json_normalize(games)
        df["schema_version"] = "scoreboard_v3"

        logger.success(f"[Collector] Retrieved {len(df)} games for {day}.")
        logger.debug(f"[Collector] ScoreboardV3 columns: {df.columns.tolist()}")
        return df

    except Exception as e:
        logger.error(f"[Collector] Failed to parse ScoreboardV3 for {day}: {e}")
        return _fetch_legacy_scoreboard(day)


# ------------------------------------------------------------
# Legacy fallback
# ------------------------------------------------------------

def _fetch_legacy_scoreboard(day: date) -> pd.DataFrame:
    """Fallback to legacy NBA API endpoint."""
    day_str = day.strftime("%Y%m%d")
    url = NBA_SCOREBOARD_URL_LEGACY.format(day_str)

    logger.warning(f"[Collector] Using legacy scoreboard endpoint for {day}.")

    data = _safe_request(url)
    if data is None:
        logger.error(f"[Collector] Legacy scoreboard also failed for {day}.")
        return pd.DataFrame()

    try:
        games = data.get("games", [])
        if not games:
            logger.warning(f"[Collector] No legacy games found for {day}.")
            return pd.DataFrame()

        df = pd.json_normalize(games)
        df["schema_version"] = "scoreboard_legacy"

        logger.success(f"[Collector] Retrieved {len(df)} legacy games for {day}.")
        logger.debug(f"[Collector] Legacy scoreboard columns: {df.columns.tolist()}")
        return df

    except Exception as e:
        logger.error(f"[Collector] Failed to parse legacy scoreboard for {day}: {e}")
        return pd.DataFrame()