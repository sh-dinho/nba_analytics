"""
Historical NBA Playoff Fetcher

- Fetches ALL playoff games from 2020 → last completed season
- Merges them into existing schedule.csv + schedule.parquet
- Safe: does NOT overwrite existing games
- Only adds missing playoff games

Usage:
    python fetch_historical_playoffs.py
"""

from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd
from pathlib import Path
import time

CSV_PATH = Path("../../../data/raw/schedule.csv")
PARQUET_PATH = Path("../../../data/parquet/schedule.parquet")

START_YEAR = 2020
MAX_RETRIES = 5
RETRY_DELAY = 5


def fetch_playoffs(season: str) -> pd.DataFrame:
    """Fetch playoff games for a given season."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching Playoffs for {season} (attempt {attempt})...")
            log = LeagueGameLog(season=season, season_type_all_star="Playoffs")
            df = log.get_data_frames()[0]
            if df.empty:
                print(f"No playoff data for {season}.")
                return pd.DataFrame()
            df["season_type"] = "Playoffs"
            return df
        except Exception as e:
            print(f"Error fetching playoffs {season}: {e}")
            time.sleep(RETRY_DELAY)

    print(f"Failed to fetch playoffs for {season}.")
    return pd.DataFrame()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize playoff data to match your pipeline schema."""
    if df.empty:
        return df

    df = df.rename(
        columns={
            "GAME_DATE": "date",
            "MATCHUP": "matchup",
            "PTS": "points",
            "TEAM_NAME": "team",
            "GAME_ID": "game_id",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    def parse_matchup(row):
        m = row["matchup"]
        if " vs. " in m:
            home, away = m.split(" vs. ")
        elif " @ " in m:
            away, home = m.split(" @ ")
        else:
            home = row["team"]
            away = None
        return pd.Series([home, away])

    df[["home_team", "away_team"]] = df.apply(parse_matchup, axis=1)

    df["home_score"] = df.apply(
        lambda r: r["points"] if r["team"] == r["home_team"] else None, axis=1
    )
    df["away_score"] = df.apply(
        lambda r: r["points"] if r["team"] == r["away_team"] else None, axis=1
    )

    df = (
        df.groupby("game_id")
        .agg(
            date=("date", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_score=("home_score", "max"),
            away_score=("away_score", "max"),
            season_type=("season_type", "first"),
        )
        .reset_index()
    )

    return df


def load_existing():
    """Load existing schedule data."""
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        print("Loaded existing Parquet.")
        return df

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print("Loaded existing CSV.")
        return df

    raise FileNotFoundError("No existing schedule data found.")


def save(df: pd.DataFrame):
    """Save updated dataset."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(CSV_PATH, index=False)
    df.to_parquet(PARQUET_PATH, index=False)

    print("Saved updated dataset (CSV + Parquet).")


def fetch_historical_playoffs():
    existing = load_existing()
    existing_ids = set(existing["game_id"].astype(str))

    current_year = pd.Timestamp.today().year
    current_season = (
        f"{current_year}-{str(current_year + 1)[-2:]}"
        if pd.Timestamp.today().month >= 10
        else f"{current_year - 1}-{str(current_year)[-2:]}"
    )

    new_frames = []

    for year in range(START_YEAR, current_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        if season == current_season:
            print(f"Skipping {season} (current season playoffs not started).")
            continue

        df = fetch_playoffs(season)
        df = normalize(df)

        if df.empty:
            continue

        df = df[~df["game_id"].astype(str).isin(existing_ids)]
        if not df.empty:
            print(f"Adding {len(df)} playoff games for {season}.")
            new_frames.append(df)

    if not new_frames:
        print("No new playoff games found.")
        return

    updated = pd.concat([existing] + new_frames, ignore_index=True)
    save(updated)
    print("Historical playoff backfill complete.")


if __name__ == "__main__":
    fetch_historical_playoffs()
