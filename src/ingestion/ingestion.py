"""
NBA Ingestion Pipeline (Enhanced)
"""

import pandas as pd
from pathlib import Path
from scripts.archieve.normalize_schedule import normalize_schedule
from scripts.archieve.fetch_today_games import fetch_today_games
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

RAW_CSV = Path("data/raw/schedule.csv")
RAW_PARQUET = Path("data/parquet/schedule.parquet")
SNAPSHOT_PATH = Path("data/ingestion/ingestion_snapshot.parquet")


# ---------------------------------------------------------
# LOAD RAW DATA
# ---------------------------------------------------------


def load_raw_data(input_path: str = None) -> pd.DataFrame:
    """
    Load raw schedule data from CSV or Parquet.
    Priority:
        1. Explicit input_path
        2. Parquet (canonical)
        3. CSV (fallback)
    """
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        logging.info(f"Loading raw data from explicit path: {path}")
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    if RAW_PARQUET.exists():
        logging.info(f"Loading raw data from Parquet: {RAW_PARQUET}")
        return pd.read_parquet(RAW_PARQUET)

    if RAW_CSV.exists():
        logging.info(f"Loading raw data from CSV: {RAW_CSV}")
        return pd.read_csv(RAW_CSV)

    raise FileNotFoundError("No raw schedule file found (CSV or Parquet).")


# ---------------------------------------------------------
# VALIDATION (FIXED)
# ---------------------------------------------------------


def validate_raw(df: pd.DataFrame):
    """
    Validate RAW schedule data BEFORE normalization.
    This checks for nba_api raw schema, not normalized schema.
    """
    required_cols = {"GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS"}
    missing = required_cols - set(df.columns)

    if missing:
        logging.warning(f"Raw data missing expected columns: {missing}")

    # Duplicate GAME_ID rows are expected (one per team)
    # But duplicate GAME_ID + TEAM_NAME is suspicious
    dupes = df[df.duplicated(subset=["GAME_ID", "TEAM_NAME"], keep=False)]
    if not dupes.empty:
        logging.warning(
            f"Found {len(dupes)} suspicious duplicate rows (GAME_ID + TEAM_NAME)."
        )


def log_season_summary(df: pd.DataFrame):
    """Log counts per season for debugging."""
    if "date" not in df.columns:
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = df["date"].apply(
        lambda d: (
            f"{d.year}-{str(d.year+1)[-2:]}"
            if d.month >= 10
            else f"{d.year-1}-{str(d.year)[-2:]}"
        )
    )

    counts = df.groupby("season")["game_id"].nunique().sort_index()

    logging.info("Season summary (raw data):")
    for season, count in counts.items():
        logging.info(f"  {season}: {count} games")


# ---------------------------------------------------------
# MAIN INGESTION PIPELINE
# ---------------------------------------------------------


def run_ingestion(input_path: str = None):
    """
    Run full ingestion pipeline: load → validate → normalize → snapshot.
    """
    logging.info("Starting NBA ingestion pipeline...")

    df_raw = load_raw_data(input_path)
    logging.info(f"Loaded raw schedule: {len(df_raw)} rows")

    validate_raw(df_raw)
    log_season_summary(df_raw)

    df_clean = normalize_schedule(df_raw)
    logging.info(f"Normalized schedule: {len(df_clean)} rows")

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(SNAPSHOT_PATH, index=False)
    logging.info(f"Saved canonical ingestion snapshot → {SNAPSHOT_PATH}")

    return df_clean, str(SNAPSHOT_PATH)


# ---------------------------------------------------------
# INGEST TODAY'S GAMES
# ---------------------------------------------------------


def ingest_today_games():
    """
    Fetch today's NBA games and merge them into the canonical dataset.
    """
    logging.info("Fetching today's NBA games...")

    today_df = fetch_today_games()

    if today_df.empty:
        logging.info("No NBA games today. Nothing to ingest.")
        return None

    # Load existing canonical dataset
    if RAW_PARQUET.exists():
        base = pd.read_parquet(RAW_PARQUET)
    elif RAW_CSV.exists():
        base = pd.read_csv(RAW_CSV)
    else:
        raise FileNotFoundError("No base schedule dataset found.")

    base["game_id"] = base["game_id"].astype(str)
    today_df["game_id"] = today_df["game_id"].astype(str)

    # Filter out duplicates
    new_games = today_df[~today_df["game_id"].isin(base["game_id"])]

    if new_games.empty:
        logging.info("Today's games already ingested.")
        return None

    updated = pd.concat([base, new_games], ignore_index=True)

    # Save updated dataset
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    updated.to_csv(RAW_CSV, index=False)
    updated.to_parquet(RAW_PARQUET, index=False)

    logging.info(f"Ingested {len(new_games)} new games from today.")
    return new_games
