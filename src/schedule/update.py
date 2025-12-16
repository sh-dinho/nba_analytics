import logging
import pandas as pd
from datetime import datetime


def update_incremental_schedule(master_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Update master schedule with today's games.
    Returns updated master schedule.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    if "date" not in master_schedule.columns:
        logging.warning("No date column in master schedule")
        return master_schedule

    today_games = master_schedule[master_schedule["date"] == today_str]
    if today_games.empty:
        logging.info("No new games today.")
        return master_schedule

    master_schedule = pd.concat([master_schedule, today_games]).drop_duplicates(
        subset="gameId"
    )
    logging.info(f"Added {len(today_games)} new games for today")
    return master_schedule
