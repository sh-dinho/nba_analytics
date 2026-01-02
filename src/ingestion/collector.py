from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Scoreboard Collector
# File: src/ingestion/collector.py
# Author: Sadiq
# ============================================================

from datetime import date
from typing import Optional
import pandas as pd
import requests
from loguru import logger
import time

NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_{}.json"
NBA_SCOREBOARD_URL_LEGACY = "https://data.nba.net/prod/v1/{}/scoreboard.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "*/*",
    "Connection": "keep-alive",
}


def _format_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def _safe_request(url: str, retries: int = 5, timeout: int = 10) -> Optional[dict]:
    """
    Robust GET request with retry logic, SSL fallback for legacy endpoints,
    and safe JSON parsing.
    """
    is_legacy = "data.nba.net" in url

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
                verify=not is_legacy,  # disable SSL only for legacy
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (403, 429):
                wait = (attempt * 2) + 5
                logger.warning(
                    f"[Collector] Blocked ({resp.status_code}) for {url}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            logger.warning(
                f"[Collector] Unexpected status {resp.status_code} for {url}"
            )

        except requests.exceptions.SSLError as e:
            logger.error(f"[Collector] SSL error for {url}: {e}")
            if not is_legacy:
                return None  # only bypass SSL for legacy

        except Exception as e:
            logger.warning(f"[Collector] Attempt {attempt} failed for {url}: {e}")

    logger.error(f"[Collector] Exhausted retries for {url}")
    return None


def fetch_scoreboard_for_date(day: date) -> pd.DataFrame:
    """
    Fetch ScoreboardV3, fallback to legacy if needed.
    """
    logger.info(f"[Collector] Fetching ScoreboardV3 for {day}...")
    day_str = _format_date(day)
    url = NBA_SCOREBOARD_URL.format(day_str)

    data = _safe_request(url)
    if data is None:
        return _fetch_legacy_scoreboard(day)

    try:
        games = data.get("scoreboard", {}).get("games", [])
        if not games:
            logger.warning(f"[Collector] No games found for {day} (V3).")
            return pd.DataFrame({"schema_version": ["scoreboard_v3"]})

        df = pd.json_normalize(games)
        df["schema_version"] = "scoreboard_v3"
        return df

    except Exception as e:
        logger.error(f"[Collector] Failed to parse V3 scoreboard: {e}")
        return _fetch_legacy_scoreboard(day)


def _fetch_legacy_scoreboard(day: date) -> pd.DataFrame:
    """
    Fetch legacy scoreboard as fallback.
    """
    day_str = _format_date(day)
    url = NBA_SCOREBOARD_URL_LEGACY.format(day_str)

    logger.warning(f"[Collector] Falling back to legacy scoreboard for {day}")

    data = _safe_request(url)
    if data is None:
        logger.error(f"[Collector] Legacy scoreboard also failed for {day}")
        return pd.DataFrame({"schema_version": ["scoreboard_legacy"]})

    try:
        games = data.get("games", [])
        if not games:
            return pd.DataFrame({"schema_version": ["scoreboard_legacy"]})

        df = pd.json_normalize(games)
        df["schema_version"] = "scoreboard_legacy"
        return df

    except Exception as e:
        logger.error(f"[Collector] Failed to parse legacy scoreboard: {e}")
        return pd.DataFrame({"schema_version": ["scoreboard_legacy"]})
