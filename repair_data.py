import pandas as pd
from src.ingestion.collector import NBADataCollector
from src.ingestion.transform import wide_to_long
from src.config.paths import SCHEDULE_SNAPSHOT, LONG_SNAPSHOT
from loguru import logger


def total_repair():
    collector = NBADataCollector()

    # 1. Re-fetch all history since 2022 (with actual scores!)
    logger.info("Fetching fresh history (2022-2025)...")
    # Change this line in repair_data.py
    seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]
    fresh_wide = collector.fetch_history(seasons)

    if fresh_wide.empty:
        logger.error("Failed to fetch history. Check your internet/NBA API.")
        return

    # 2. Convert to Long Format
    logger.info("Converting to ML Long Format...")
    fresh_long = wide_to_long(fresh_wide)

    # 3. Save clean snapshots
    SCHEDULE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    fresh_wide.to_parquet(SCHEDULE_SNAPSHOT, index=False)
    fresh_long.to_parquet(LONG_SNAPSHOT, index=False)

    logger.success(f"REPAIR COMPLETE: {len(fresh_wide)} games recovered with scores.")


if __name__ == "__main__":
    total_repair()
