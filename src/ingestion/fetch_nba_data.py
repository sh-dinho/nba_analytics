"""
NBA Game Data Fetcher (Full & Daily Incremental)
Fetches real NBA games (2020 → today) using nba_api.
Stores results in data/raw/schedule.csv
"""

from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd
from pathlib import Path
import time

OUTPUT_PATH = Path("data/raw/schedule.csv")
RETRY_DELAY = 5
MAX_RETRIES = 5
START_YEAR = 2020


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch NBA season with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log = LeagueGameLog(season=season, season_type_all_star="Regular Season")
            df = log.get_data_frames()[0]
            return df
        except Exception as e:
            print(f"Error fetching season {season}, attempt {attempt}: {e}")
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Failed to fetch season {season} after {MAX_RETRIES} attempts")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize nba_api schema → pipeline schema."""
    df = df.rename(
        columns={
            "GAME_DATE": "date",
            "MATCHUP": "matchup",
            "PTS": "points",
            "TEAM_NAME": "team",
            "GAME_ID": "game_id",
        }
    )

    def parse_matchup(row):
        matchup = row["matchup"]
        if " vs. " in matchup:
            home, away = matchup.split(" vs. ")
        elif " @ " in matchup:
            away, home = matchup.split(" @ ")
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
        )
        .reset_index()
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df


def get_current_season() -> str:
    today = pd.Timestamp.today()
    year = today.year
    if today.month >= 10:
        return f"{year}-{str(year+1)[-2:]}"
    else:
        return f"{year-1}-{str(year)[-2:]}"


def fetch_all_seasons(start_year=START_YEAR) -> pd.DataFrame:
    seasons = []
    current_year = pd.Timestamp.today().year

    for year in range(start_year, current_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"Fetching season {season}...")
        df = fetch_season(season)
        df = normalize(df)
        seasons.append(df)

    return pd.concat(seasons, ignore_index=True)


def incremental_update():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        existing_ids = set(existing["game_id"])
        last_date = existing["date"].max()
        print(f"Existing data found. Last game date: {last_date.date()}")

        season = get_current_season()
        print(f"Fetching new games for season {season}...")
        new_data = normalize(fetch_season(season))
        new_data = new_data[
            (new_data["date"] > last_date) & (~new_data["game_id"].isin(existing_ids))
        ]

        if not new_data.empty:
            updated = pd.concat([existing, new_data], ignore_index=True)
            updated.to_csv(OUTPUT_PATH, index=False)
            print(f"Added {len(new_data)} new games.")
        else:
            print("No new games found today.")
    else:
        print("No existing schedule.csv found. Fetching all seasons...")
        all_data = fetch_all_seasons()
        all_data.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved full dataset → {OUTPUT_PATH}")


if __name__ == "__main__":
    incremental_update()
