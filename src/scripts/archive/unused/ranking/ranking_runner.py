"""
Ranking runner for NBA Analytics v3.
Takes predictions → produces ranking snapshot.
"""

from pathlib import Path
from loguru import logger
import pandas as pd


def run_ranking(df_preds: pd.DataFrame, output_dir="data/rankings"):
    logger.info("Building ranking snapshot...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example: sort by probability descending
    df_ranked = df_preds.sort_values("probability", ascending=False)

    snapshot_path = output_dir / "rankings_latest.parquet"
    df_ranked.to_parquet(snapshot_path, index=False)

    logger.info(f"Ranking snapshot saved → {snapshot_path}")
    return snapshot_path
