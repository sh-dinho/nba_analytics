from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5
# Unified Multi‑Source Game Loader
# Author: Sadiq
#
# Priority:
#   1. NBA Stats API (fast, structured)
#   2. Basketball Reference fallback (stable, public)
#
# Features:
#   • Full browser headers for NBA API
#   • Retry logic with exponential backoff
#   • Session pooling
#   • NBA API: avoids future seasons
#   • BRef: old + new formats (monthly + playoffs)
#   • Team cleaning + normalization
#   • Deduplication + sorting + season annotation
#
# Output:
#   data/raw/games_master.csv
# ============================================================

import time
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.utils.team_names import normalize_team


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clean_team(name: str) -> str:
    """Remove seeds, symbols, and whitespace."""
    if not isinstance(name, str):
        return name
    name = re.sub(r"\(.*?\)", "", name)
    name = name.replace("*", "").replace("†", "")
    return name.strip()


# ============================================================
# 🏀 NBA API SECTION (PRIMARY SOURCE)
# ============================================================

NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.nba.com",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://www.nba.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}

NBA_API_URL = "https://stats.nba.com/stats/leaguegamelog"


def nba_fetch_with_retry(session: requests.Session, params: dict, max_retries: int = 5):
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(
                NBA_API_URL,
                params=params,
                headers=NBA_HEADERS,
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning(f"⚠️ NBA API attempt {attempt}: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ NBA API attempt {attempt}: {e}")
        time.sleep(delay)
        delay *= 2
    logger.error("❌ NBA API failed after retries")
    return None


def nba_fetch_season(session: requests.Session, season: str, season_type: str) -> pd.DataFrame:
    logger.info(f"📥 NBA API: {season_type} {season}")

    params = {
        "Season": season,
        "SeasonType": season_type,
        "PlayerOrTeam": "T",
        "Counter": "0",
        "Sorter": "DATE",
        "Direction": "ASC",
        "LeagueID": "00",
        "MeasureType": "Base",
        "PerMode": "Totals",
    }

    data = nba_fetch_with_retry(session, params)
    if not data:
        return pd.DataFrame()

    rows = data["resultSets"][0]["rowSet"]
    headers = data["resultSets"][0]["headers"]
    return pd.DataFrame(rows, columns=headers)


def convert_season_format(year: int) -> str:
    """Convert 2016 → '2016-17'."""
    return f"{year}-{str(year + 1)[-2:]}"


def load_from_nba_api(start_year: int, end_year: int | None) -> pd.DataFrame | None:
    """
    Load games from NBA Stats API.

    end_year is interpreted as the last *start year* (inclusive) to fetch.
    If None, it is clamped to last plausible completed season.
    """
    # If today is 2026 → most recent safe start year = 2024 (for 2024-25).
    # To avoid future / not-yet-published seasons, clamp one year back.
    if end_year is None:
        end_year = pd.Timestamp.today().year - 1

    session = requests.Session()
    frames: list[pd.DataFrame] = []

    # inclusive range over start years
    for year in range(start_year, end_year + 1):
        season = convert_season_format(year)

        df_reg = nba_fetch_season(session, season, "Regular Season")
        df_po = nba_fetch_season(session, season, "Playoffs")

        if not df_reg.empty:
            frames.append(df_reg)
        if not df_po.empty:
            frames.append(df_po)

    if not frames:
        logger.warning("⚠️ NBA API returned no data for all requested seasons")
        return None

    df = pd.concat(frames, ignore_index=True)

    df = df.rename(
        columns={
            "GAME_DATE": "date",
            "MATCHUP": "matchup",
            "PTS": "team_score",
            "TEAM_NAME": "team",
            "TEAM_ABBREVIATION": "team_abbr",
            "GAME_ID": "game_id",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "matchup"])

    # Home/away detection from MATCHUP
    df["is_home"] = df["matchup"].str.contains("vs").astype(int)
    df["opponent"] = df["matchup"].str.extract(r" (?:vs\.|@) (.*)")

    df["team"] = df["team"].apply(clean_team).apply(normalize_team)
    df["opponent"] = df["opponent"].apply(clean_team).apply(normalize_team)

    df = df.dropna(subset=["team", "opponent"])

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    if home.empty or away.empty:
        logger.warning("⚠️ NBA API: home or away subset is empty")
        return None

    merged = pd.merge(
        home,
        away,
        on="game_id",
        suffixes=("_home", "_away"),
        how="inner",
    )

    if merged.empty:
        logger.warning("⚠️ NBA API: no complete home/away pairs after merge")
        return None

    final = pd.DataFrame(
        {
            "date": merged["date_home"],
            "home_team": merged["team_home"],
            "away_team": merged["team_away"],
            "home_score": merged["team_score_home"],
            "away_score": merged["team_score_away"],
            "game_id": merged["game_id"],
        }
    )

    logger.info(f"✅ NBA API produced {len(final)} games")
    return final


# ============================================================
# 🏀 BASKETBALL REFERENCE SECTION (FALLBACK)
# ============================================================

BREF_OLD_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_games.html"
BREF_MONTHS = [
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
]
BREF_NEW_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
BREF_PLAYOFFS_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_games-playoffs.html"


def bref_fetch_html_with_retry(url: str) -> str | None:
    for attempt in range(1, 4):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            logger.warning(f"⚠️ BRef attempt {attempt} failed for {url}: {e}")
        time.sleep(1)
    logger.warning(f"⚠️ BRef failed after retries: {url}")
    return None


def parse_bref_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "games"})
    if table is None:
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]
    df = df.dropna(subset=["Date"])

    df = df.rename(
        columns={
            "Visitor/Neutral": "away_team",
            "Home/Neutral": "home_team",
            "PTS": "away_score",
            "PTS.1": "home_score",
        }
    )

    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df[["date", "home_team", "away_team", "home_score", "away_score"]]


def load_from_bref(start_year: int, end_year: int | None) -> pd.DataFrame | None:
    if end_year is None:
        end_year = pd.Timestamp.today().year + 1

    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        logger.info(f"📥 BRef: fetching season {year}")

        # Old format (pre‑2018)
        if year < 2018:
            html = bref_fetch_html_with_retry(BREF_OLD_URL.format(year=year))
            if html:
                df = parse_bref_table(html)
                if not df.empty:
                    frames.append(df)
            continue

        # New format (2018+): monthly pages
        for month in BREF_MONTHS:
            url = BREF_NEW_URL.format(year=year, month=month)
            html = bref_fetch_html_with_retry(url)
            if not html:
                continue
            df = parse_bref_table(html)
            if not df.empty:
                frames.append(df)

        # Playoffs
        playoffs_html = bref_fetch_html_with_retry(BREF_PLAYOFFS_URL.format(year=year))
        if playoffs_html:
            df = parse_bref_table(playoffs_html)
            if not df.empty:
                frames.append(df)

    if not frames:
        logger.warning("⚠️ BRef returned no data for all requested seasons")
        return None

    df = pd.concat(frames, ignore_index=True)

    df["home_team"] = df["home_team"].apply(clean_team).apply(normalize_team)
    df["away_team"] = df["away_team"].apply(clean_team).apply(normalize_team)

    df = df.dropna(subset=["home_team", "away_team"])

    logger.info(f"✅ BRef produced {len(df)} games")
    return df


# ============================================================
# 🏀 MULTI‑SOURCE LOADER
# ============================================================

def load_games_multi_source(start_year: int = 2016, end_year: int | None = None) -> Path:
    logger.info("🔍 Multi‑source loader starting (NBA API → BRef fallback)")

    # 1. Try NBA API
    df_api = load_from_nba_api(start_year, end_year)

    if df_api is not None and not df_api.empty:
        df = df_api
        logger.success(f"✅ Using NBA API data ({len(df)} games)")
    else:
        logger.warning("⚠️ NBA API unavailable or empty — falling back to Basketball Reference")
        df_bref = load_from_bref(start_year, end_year)
        if df_bref is None or df_bref.empty:
            raise RuntimeError("❌ Both NBA API and Basketball Reference failed to produce data")
        df = df_bref
        logger.success(f"📚 Using BRef fallback ({len(df)} games)")

    # Final cleaning
    df = df.dropna(subset=["home_team", "away_team"])
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)

    df["season"] = df["date"].dt.year.where(
        df["date"].dt.month >= 10,
        df["date"].dt.year - 1,
    )

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_path = raw_dir / "games_master.csv"
    df.to_csv(out_path, index=False)

    logger.success(f"🎉 Multi‑source load complete → {out_path}")
    logger.success(f"📊 Total games: {len(df)}")

    return out_path


# ============================================================
# 🏀 ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    load_games_multi_source()
