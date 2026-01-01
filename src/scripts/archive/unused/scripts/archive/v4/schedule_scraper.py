from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: External Schedule Scraper (ESPN)
# File: src/ingestion/schedule_scraper.py
#
# Description:
#     Scrapes the NBA schedule (regular season + playoffs)
#     from ESPN and stores it as season_schedule.parquet.
#
#     - Covers Oct 1 → Jun 30 of a season
#     - Normalizes team names (via ESPN normalizer)
#     - Produces synthetic, stable game_id
#     - Parallel fetching with retries and backoff
#     - Optional HTML debug dumps
# ============================================================

from datetime import datetime, date
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Iterable, List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.config.paths import SEASON_SCHEDULE_PATH
from src.ingestion.espn_normalizer import normalize_espn_schedule

ESPN_SCHEDULE_URL = "https://www.espn.com/nba/schedule/_/date/{date}"

# Parallel + retry configuration
MAX_WORKERS = 16
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0

# HTML debug dump dir (optional)
HTML_DEBUG_DIR = Path("data/debug/espn_schedule_html")


# ------------------------------------------------------------
# Low-level HTTP + parsing
# ------------------------------------------------------------


def _fetch_html_with_retries(date_str: str) -> str:
    """
    Fetch raw HTML for a given date with retries and basic backoff.
    date_str format: YYYYMMDD
    """
    url = ESPN_SCHEDULE_URL.format(date=date_str)
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"[ScheduleScraper] Fetching ESPN schedule (attempt {attempt}/{MAX_RETRIES}): {url}"
            )
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt >= MAX_RETRIES:
                logger.error(
                    f"[ScheduleScraper] Failed to fetch {url} after {MAX_RETRIES} attempts: {e}"
                )
                raise
            sleep_for = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"[ScheduleScraper] Error fetching {url}: {e} "
                f"(retrying in {sleep_for:.1f}s)..."
            )
            time.sleep(sleep_for)

    raise RuntimeError(f"[ScheduleScraper] Exhausted retries for {url}")


def _dump_html_debug(date_str: str, html: str) -> None:
    """
    Optionally dump raw HTML to disk for debugging.
    Controlled via env var: NBA_SCHEDULE_DUMP_HTML=1
    """
    if os.getenv("NBA_SCHEDULE_DUMP_HTML", "0") != "1":
        return

    HTML_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HTML_DEBUG_DIR / f"espn_schedule_{date_str}.html"
    try:
        out_path.write_text(html, encoding="utf-8")
        logger.debug(f"[ScheduleScraper] HTML debug dump → {out_path}")
    except Exception as e:
        logger.warning(f"[ScheduleScraper] Failed to dump HTML for {date_str}: {e}")


def _parse_day_html(html: str, date_str: str) -> pd.DataFrame:
    """
    Parse HTML for a single day of schedule.
    Extracts:
        - away_team
        - home_team
        - game_time (local time string as shown on ESPN, if present)

    Returns a DataFrame with columns:
        - date (YYYYMMDD as string)
        - away_team
        - home_team
        - game_time
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    if not tables:
        logger.info(
            f"[ScheduleScraper] No tables found for {date_str} (possibly no games)."
        )
        return pd.DataFrame()

    rows = []
    meaningful_rows = 0

    for table in tables:
        trs = table.find_all("tr")
        if len(trs) <= 1:
            continue  # header-only or empty

        for tr in trs[1:]:
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            away = tds[0].get_text(strip=True)
            home = tds[1].get_text(strip=True)
            if not away or not home:
                continue

            game_time: Optional[str] = None
            if len(tds) >= 3:
                game_time_raw = tds[2].get_text(strip=True)
                game_time = game_time_raw or None

            rows.append(
                {
                    "date": date_str,
                    "away_team": away,
                    "home_team": home,
                    "game_time": game_time,
                }
            )
            meaningful_rows += 1

    if tables and meaningful_rows == 0:
        logger.warning(
            f"[ScheduleScraper] Tables found but no parsable rows for {date_str}. "
            f"ESPN markup may have changed."
        )

    return pd.DataFrame(rows)


def _fetch_day(date_str: str) -> pd.DataFrame:
    """
    Fetch schedule for a single day from ESPN.
    date_str format: YYYYMMDD
    """
    html = _fetch_html_with_retries(date_str)
    _dump_html_debug(date_str, html)
    return _parse_day_html(html, date_str)


# ------------------------------------------------------------
# Date utilities
# ------------------------------------------------------------


def _infer_season_window(season_start_year: int) -> tuple[datetime, datetime]:
    """
    For a given season_start_year (e.g. 2024 for 2024-25),
    scrape from Oct 1 (start) to Jun 30 (end).
    """
    start = datetime(season_start_year, 10, 1)
    end = datetime(season_start_year + 1, 6, 30)
    return start, end


def _date_range(start: datetime, end: datetime) -> list[date]:
    """
    Inclusive date range [start, end].
    """
    cur = start
    dates: list[date] = []
    while cur <= end:
        dates.append(cur.date())
        cur = cur + pd.Timedelta(days=1)
    return dates


# ------------------------------------------------------------
# Main public API
# ------------------------------------------------------------


def build_season_schedule(
    season_start_year: int, max_workers: int = MAX_WORKERS
) -> pd.DataFrame:
    """
    Scrape the full NBA schedule (regular + playoffs) from ESPN,
    normalize team names, and persist to SEASON_SCHEDULE_PATH as Parquet.

    This is a SCHEDULE artifact, separate from the ScoreboardV3 ingestion pipeline.
    """
    logger.info(
        f"[ScheduleScraper] Scraping ESPN schedule for season "
        f"{season_start_year}-{season_start_year + 1}"
    )

    start, end = _infer_season_window(season_start_year)
    all_dates = _date_range(start, end)
    logger.info(
        f"[ScheduleScraper] Date window: {start.date()} → {end.date()} "
        f"({len(all_dates)} days)"
    )

    all_rows: List[pd.DataFrame] = []

    def _task(d: date) -> tuple[str, Optional[pd.DataFrame], Optional[Exception]]:
        date_str = d.strftime("%Y%m%d")
        try:
            df_day = _fetch_day(date_str)
            return date_str, df_day, None
        except Exception as exc:
            return date_str, None, exc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(_task, d): d for d in all_dates}

        for future in as_completed(future_to_date):
            d = future_to_date[future]
            date_str = d.strftime("%Y%m%d")
            try:
                fetched_date_str, df_day, exc = future.result()
                assert fetched_date_str == date_str
                if exc is not None:
                    logger.error(f"[ScheduleScraper] Failed to fetch {date_str}: {exc}")
                    continue

                if df_day is not None and not df_day.empty:
                    all_rows.append(df_day)
                    logger.debug(
                        f"[ScheduleScraper] Parsed {len(df_day)} games for {date_str}"
                    )
                else:
                    logger.debug(f"[ScheduleScraper] No games found for {date_str}")
            except Exception as e:
                logger.error(f"[ScheduleScraper] Unexpected error on {date_str}: {e}")

    if not all_rows:
        raise RuntimeError("[ScheduleScraper] No schedule data scraped from ESPN.")

    raw_df = pd.concat(all_rows, ignore_index=True)

    # Convert date string (YYYYMMDD) to datetime for normalizer
    raw_df["date"] = pd.to_datetime(raw_df["date"], format="%Y%m%d")

    # Normalize via ESPN normalizer
    normalized = normalize_espn_schedule(raw_df)

    if normalized.empty:
        raise RuntimeError(
            "[ScheduleScraper] All rows dropped after ESPN normalization. "
            "Check team name mappings / HTML parsing."
        )

    # Persist to Parquet
    SEASON_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(SEASON_SCHEDULE_PATH, index=False)

    logger.success(
        f"[ScheduleScraper] ESPN season schedule saved → {SEASON_SCHEDULE_PATH} "
        f"(rows={len(normalized)})"
    )

    return normalized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape NBA season schedule from ESPN and store as Parquet."
    )
    parser.add_argument(
        "--season-start-year",
        type=int,
        required=False,
        default=2024,
        help="Starting year of the NBA season (e.g. 2024 for 2024-25).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of worker threads for parallel fetching.",
    )

    args = parser.parse_args()
    build_season_schedule(
        season_start_year=args.season_start_year,
        max_workers=args.max_workers,
    )
