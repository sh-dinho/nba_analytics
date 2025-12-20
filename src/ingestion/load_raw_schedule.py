"""
Load raw schedule data for NBA Analytics v3.

This is the ONLY place that reads data/raw/schedule.csv.
Everything else should use the normalized ingestion snapshot.
"""

from pathlib import Path
import pandas as pd


def load_raw_schedule(input_path: str) -> pd.DataFrame:
    """
    Load the raw schedule CSV.

    Expected file:
        data/raw/schedule.csv

    This function DOES NOT enforce schema; that is done
    in normalize_schedule().
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw schedule not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Raw schedule is empty: {path}")

    return df
