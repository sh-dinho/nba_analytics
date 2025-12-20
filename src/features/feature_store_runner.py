"""
Feature Store Runner
Wrapper for building feature store snapshots
"""

import logging
from src.features.feature_store import create_snapshot

logging.basicConfig(level=logging.INFO)


def run_feature_store(canonical_ingestion_path: str) -> str:
    """
    Build feature store snapshot from canonical ingestion.

    Args:
        canonical_ingestion_path (str): Path to ingestion snapshot

    Returns:
        str: Path to saved feature snapshot
    """
    logging.info("Starting Feature Store snapshot build…")
    snapshot_path = create_snapshot(canonical_ingestion_path)
    logging.info(f"Feature snapshot complete → {snapshot_path}")
    return snapshot_path
