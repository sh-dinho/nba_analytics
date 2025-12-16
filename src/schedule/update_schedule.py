import pandas as pd
import logging
from pathlib import Path

HISTORY_DIR = Path("data/history")


def update_incremental_schedule(master_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Load all historical Parquet files, deduplicate, and merge with master_schedule.
    Only new games are added.
    """
    all_files = sorted(HISTORY_DIR.glob("*.parquet"))
    if not all_files:
        logging.warning(f"No historical files found in {HISTORY_DIR}")
        return master_schedule

    logging.info(f"Found {len(all_files)} historical files. Merging...")

    df_list = [pd.read_parquet(f) for f in all_files]
    historical_df = pd.concat(df_list, ignore_index=True)

    if "game_id" in historical_df.columns:
        historical_df = historical_df.drop_duplicates(subset=["game_id"])

    if not master_schedule.empty:
        # Add only new games
        if "game_id" in master_schedule.columns:
            historical_df = historical_df[
                ~historical_df["game_id"].isin(master_schedule["game_id"])
            ]

    combined_schedule = pd.concat([master_schedule, historical_df], ignore_index=True)

    if "game_id" in combined_schedule.columns:
        combined_schedule = combined_schedule.drop_duplicates(subset=["game_id"])

    # Ensure date column is datetime
    if "game_date" in combined_schedule.columns:
        combined_schedule["game_date"] = pd.to_datetime(combined_schedule["game_date"])
        combined_schedule = combined_schedule.sort_values("game_date")

    logging.info(f"Incremental schedule updated (rows={len(combined_schedule)})")
    return combined_schedule
