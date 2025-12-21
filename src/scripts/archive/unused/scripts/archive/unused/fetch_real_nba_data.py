"""
NBA Game Data Fetcher (Full & Daily Incremental)
Fetches real NBA games (2020 → today) using nba_api.
Includes Regular Season + Playoffs with automatic season-phase detection.
Stores results in:
    - data/raw/schedule.csv
    - data/parquet/schedule.parquet

Usage:
    python fetch_real_nba_data.py
"""

from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd
from pathlib import Path
import time

OUTPUT_PATH = Path("../../data/raw/schedule.csv")
PARQUET_PATH = Path("../../data/parquet/schedule.parquet")
RETRY_DELAY = 5
MAX_RETRIES = 5
START_YEAR = 2020


# ---------------------------------------------------------
#  AUTOMATIC SEASON PHASE DETECTION
# ---------------------------------------------------------


def get_season_phase():
    """
    Automatically determine whether the NBA is in:
    - Regular Season (Oct → mid-Apr)
    - Playoffs (mid-Apr → June)
    - Offseason (July → Sept)
    """
    today = pd.Timestamp.today()
    month = today.month

    if month >= 10 or month <= 4:
        return "regular"
    elif 4 < month <= 6:
        return "playoffs"
    else:
        return "offseason"


def allowed_season_types():
    """Return which season types should be fetched based on the calendar."""
    phase = get_season_phase()

    if phase == "regular":
        return ["Regular Season"]
    elif phase == "playoffs":
        return ["Regular Season", "Playoffs"]
    else:
        # Offseason: only fetch Regular Season (Playoffs are over)
        return ["Regular Season"]


# ---------------------------------------------------------
#  FETCHING
# ---------------------------------------------------------


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch NBA season with automatic season-phase filtering."""
    season_types = allowed_season_types()
    all_frames = []

    for season_type in season_types:
        df = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"Fetching {season_type} for {season} (attempt {attempt})...")
                log = LeagueGameLog(season=season, season_type_all_star=season_type)
                df = log.get_data_frames()[0]

                if df.empty:
                    print(f"No data for {season_type} {season} — skipping.")
                    df = None
                else:
                    df["season_type"] = season_type

                break

            except Exception as e:
                print(f"Error fetching {season_type} {season}, attempt {attempt}: {e}")
                time.sleep(RETRY_DELAY)

        if df is not None:
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------
#  NORMALIZATION
# ---------------------------------------------------------


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize nba_api schema → pipeline schema (one row per game)."""
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
        matchup = row["matchup"]
        if isinstance(matchup, str):
            if " vs. " in matchup:
                home, away = matchup.split(" vs. ")
            elif " @ " in matchup:
                away, home = matchup.split(" @ ")
            else:
                home = row["team"]
                away = None
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

    agg_dict = {
        "date": ("date", "first"),
        "home_team": ("home_team", "first"),
        "away_team": ("away_team", "first"),
        "home_score": ("home_score", "max"),
        "away_score": ("away_score", "max"),
    }

    if "season_type" in df.columns:
        agg_dict["season_type"] = ("season_type", "first")

    df = df.groupby("game_id").agg(**agg_dict).reset_index()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["home_team", "away_team"])

    return df


# ---------------------------------------------------------
#  SEASON UTILITIES
# ---------------------------------------------------------


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
        print(f"Fetching full season {season}...")
        df = fetch_season(season)
        df = normalize(df)
        if not df.empty:
            seasons.append(df)

    return pd.concat(seasons, ignore_index=True)


# ---------------------------------------------------------
#  OUTPUT
# ---------------------------------------------------------


def save_outputs(df: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    df.to_parquet(PARQUET_PATH, index=False)

    print(f"Saved CSV → {OUTPUT_PATH}")
    print(f"Saved Parquet → {PARQUET_PATH}")


def parquet_is_empty():
    """Return True if parquet file is missing or empty."""
    if not PARQUET_PATH.exists():
        return True
    try:
        df = pd.read_parquet(PARQUET_PATH)
        return df.empty
    except Exception:
        return True


# ---------------------------------------------------------
#  INCREMENTAL UPDATE
# ---------------------------------------------------------


def incremental_update():
    if parquet_is_empty():
        print("Parquet missing or empty → fetching all seasons...")
        all_data = fetch_all_seasons()
        save_outputs(all_data)
        return

    existing = pd.read_parquet(PARQUET_PATH)
    existing_ids = set(existing["game_id"])
    last_date = existing["date"].max()

    print(f"Existing data found. Last game date: {last_date.date()}")

    season = get_current_season()
    print(f"Fetching new games for season {season}...")

    new_raw = fetch_season(season)
    new_data = normalize(new_raw)

    new_data = new_data[
        (new_data["date"] > last_date) & (~new_data["game_id"].isin(existing_ids))
    ]

    if new_data.empty:
        print("No new games found.")
        return

    updated = pd.concat([existing, new_data], ignore_index=True)
    save_outputs(updated)
    print(f"Added {len(new_data)} new games.")


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    incremental_update()
