"""
Feature Store Module
Handles loading, creating, and managing feature snapshots.
"""

import pandas as pd
from pathlib import Path

# Path to store parquet files
FEATURE_PATH = Path("data/parquet")
RAW_PATH = Path("data/raw")


class FeatureStore:
    def __init__(self):
        # Initialize paths or other attributes
        pass

    def ensure_parquet(self):
        """
        Checks if the canonical CSV (schedule.csv) exists, and converts it to Parquet format if not already done.
        """
        raw_csv_path = RAW_PATH / "schedule.csv"
        parquet_path = FEATURE_PATH / "schedule.parquet"

        # Check if CSV exists and Parquet file does not exist
        if raw_csv_path.exists() and not parquet_path.exists():
            print(f"Converting {raw_csv_path} to Parquet format...")
            df = pd.read_csv(raw_csv_path)
            parquet_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists
            df.to_parquet(parquet_path)
            print(f"Conversion complete. Parquet saved at {parquet_path}")
        else:
            print(
                f"Parquet file already exists at {parquet_path}, skipping conversion."
            )

    def load_latest_snapshot(self) -> pd.DataFrame:
        """
        Load the latest feature snapshot (Parquet format).
        """
        snapshots = sorted(FEATURE_PATH.glob("*.parquet"), reverse=True)
        if snapshots:
            return pd.read_parquet(snapshots[0])
        else:
            raise FileNotFoundError("No feature snapshots found.")
