import pandas as pd
from pathlib import Path


def load_master_schedule(file_path: Path) -> pd.DataFrame:
    if file_path.exists():
        return pd.read_parquet(file_path)
    return pd.DataFrame()


def save_master_schedule(df: pd.DataFrame, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)
