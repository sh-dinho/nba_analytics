"""
NBA Ingestion Pipeline
"""

import pandas as pd
from pathlib import Path
from src.ingestion.normalize_schedule import normalize_schedule
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def run_ingestion(input_path: str = None):
    """
    Run full ingestion pipeline: load CSV → normalize → save snapshot.

    Args:
        input_path (str): Optional path to raw CSV.

    Returns:
        Tuple[pd.DataFrame, str]: Normalized DataFrame, path to snapshot parquet.
    """
    base_path = Path(input_path) if input_path else Path("data/raw/schedule.csv")
    if not base_path.exists():
        raise FileNotFoundError(f"Ingestion failed, file not found: {base_path}")

    logging.info(f"Starting ingestion with base schedule: {base_path}")
    df_raw = pd.read_csv(base_path)
    logging.info(f"Loaded raw schedule: {len(df_raw)} rows")

    # Normalize
    df_clean = normalize_schedule(df_raw)
    logging.info(f"Normalized schedule: {len(df_clean)} rows")

    # Save canonical ingestion snapshot
    snapshot_path = Path("data/ingestion/ingestion_snapshot.parquet")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(snapshot_path, index=False)
    logging.info(f"Saved canonical ingestion snapshot → {snapshot_path}")

    return df_clean, str(snapshot_path)
