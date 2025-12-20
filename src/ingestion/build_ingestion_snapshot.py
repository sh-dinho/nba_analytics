"""
Build a canonical ingestion snapshot that merges:
- Completed games (with scores)
- Scheduled games (no scores yet)

Output:
    data/ingestion/ingestion_snapshot.parquet
"""

from pathlib import Path
import pandas as pd


def load_completed_games() -> pd.DataFrame:
    """
    Load completed games from your existing raw/processed file.

    Expected columns at minimum:
        - game_id
        - date
        - home_team
        - away_team
        - home_score
        - away_score
    """
    path = Path("data/raw/completed_games.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Completed games file not found: {path}")

    df = pd.read_parquet(path)

    required_cols = {
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Completed games missing columns: {missing}")

    df["status"] = "final"
    return df


def load_scheduled_games() -> pd.DataFrame:
    """
    Load scheduled games from the season schedule.

    Expected input file:
        data/raw/season_schedule.csv

    Expected columns at minimum:
        - game_id
        - game_date
        - home_team
        - away_team
    """
    path = Path("data/raw/season_schedule.csv")
    if not path.exists():
        raise FileNotFoundError(f"Season schedule file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"game_id", "game_date", "home_team", "away_team"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Season schedule missing columns: {missing}")

    # Standardize column name to 'date' to match completed games
    df = df.rename(columns={"game_date": "date"})

    # Scheduled games have no scores yet
    df["home_score"] = None
    df["away_score"] = None
    df["status"] = "scheduled"

    return df


def build_ingestion_snapshot() -> pd.DataFrame:
    """
    Merge completed + scheduled games into a canonical snapshot.
    Completed games override scheduled ones if both exist.
    """
    completed = load_completed_games()
    scheduled = load_scheduled_games()

    # Only keep scheduled games that are NOT already completed
    completed_ids = set(completed["game_id"].unique())
    scheduled = scheduled[~scheduled["game_id"].isin(completed_ids)]

    df = pd.concat([completed, scheduled], ignore_index=True)

    # Sort by date, then game_id for stability
    df = df.sort_values(["date", "game_id"])

    out = Path("data/ingestion/ingestion_snapshot.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Saved merged ingestion snapshot → {out}")
    return df


if __name__ == "__main__":
    build_ingestion_snapshot()
