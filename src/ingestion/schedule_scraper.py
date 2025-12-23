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
#     - Normalizes team names to NBA tricodes
#     - Produces synthetic, stable game_id
# ============================================================

from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.config.paths import SEASON_SCHEDULE_PATH
from src.utils.team_names import normalize_schedule

ESPN_SCHEDULE_URL = "https://www.espn.com/nba/schedule/_/date/{date}"


def _fetch_day(date_str: str) -> pd.DataFrame:
    """
    Fetch schedule for a single day from ESPN.
    date_str format: YYYYMMDD
    """
    url = ESPN_SCHEDULE_URL.format(date=date_str)
    logger.info(f"[ScheduleScraper] Fetching ESPN schedule: {url}")

    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    rows = []
    for table in tables:
        # First row is header usually
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            away = tds[0].get_text(strip=True)
            home = tds[1].get_text(strip=True)

            if not away or not home:
                continue

            rows.append(
                {
                    "date": date_str,
                    "away_team": away,
                    "home_team": home,
                }
            )

    return pd.DataFrame(rows)


def _infer_season_window(season_start_year: int) -> tuple[datetime, datetime]:
    """
    For a given season_start_year (e.g. 2024 for 2024-25),
    scrape from Oct 1 (start) to Jun 30 (end).
    """
    start = datetime(season_start_year, 10, 1)
    end = datetime(season_start_year + 1, 6, 30)
    return start, end


def build_season_schedule(season_start_year: int) -> pd.DataFrame:
    """
    Scrape the full NBA schedule (regular + playoffs) from ESPN,
    normalize team names to tricodes, and persist to
    SEASON_SCHEDULE_PATH as Parquet.
    """
    logger.info(
        f"[ScheduleScraper] Scraping ESPN schedule for season "
        f"{season_start_year}-{season_start_year + 1}"
    )

    start, end = _infer_season_window(season_start_year)

    all_rows = []
    cur = start
    while cur <= end:
        date_str = cur.strftime("%Y%m%d")
        try:
            df_day = _fetch_day(date_str)
            if not df_day.empty:
                all_rows.append(df_day)
        except Exception as e:
            logger.error(f"[ScheduleScraper] Failed to fetch {date_str}: {e}")

        cur = cur + pd.Timedelta(days=1)

    if not all_rows:
        raise RuntimeError("[ScheduleScraper] No schedule data scraped from ESPN.")

    df = pd.concat(all_rows, ignore_index=True)

    # Convert date string to datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Normalize team names → tricodes
    df = normalize_schedule(df)

    if df.empty:
        raise RuntimeError(
            "[ScheduleScraper] All rows dropped after team normalization. "
            "Check TEAM_NAME_MAP."
        )

    # Create synthetic, stable game_id
    df["game_id"] = df.apply(
        lambda r: f"{r['date'].strftime('%Y%m%d')}_{r['away_team']}_{r['home_team']}",
        axis=1,
    )

    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    SEASON_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SEASON_SCHEDULE_PATH, index=False)

    logger.success(
        f"[ScheduleScraper] Season schedule saved → {SEASON_SCHEDULE_PATH} "
        f"(rows={len(df)})"
    )

    return df


if __name__ == "__main__":
    # Example: for 2024-25 season
    build_season_schedule(season_start_year=2024)
