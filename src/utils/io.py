# ============================================================
# File: src/utils/io.py
# Purpose: Load/Save data utility functions
# ============================================================

import pandas as pd
import os

def load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV or Parquet file into DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format: must be CSV or Parquet")

def save_dataframe(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV or Parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported file format: must be CSV or Parquet")
