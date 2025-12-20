"""
Ingestion Runner — NBA Analytics v3
"""

import logging
from src.ingestion.ingestion import run_ingestion

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def run_ingestion_runner(input_path=None, return_df=False):
    """
    Runs the ingestion pipeline.

    Args:
        input_path (str): Optional CSV path for raw schedule data.
        return_df (bool): If True, returns DataFrame along with snapshot path.

    Returns:
        Tuple[pd.DataFrame, str] if return_df=True, else path to ingestion snapshot
    """
    logging.info("Running ingestion pipeline...")
    df_snapshot, snapshot_path = run_ingestion(input_path=input_path)

    logging.info(f"Ingestion complete → {snapshot_path}")
    logging.info(f"Ingested rows: {len(df_snapshot)}")

    if return_df:
        return df_snapshot, snapshot_path
    return snapshot_path
