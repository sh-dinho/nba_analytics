# ============================================================
# File: src/schedule/compare.py
# Purpose: Compare today schedule vs baseline
# ============================================================

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compare_schedules(
    historical_schedule: pd.DataFrame, today_schedule: pd.DataFrame, config
) -> pd.DataFrame:
    """
    Compare today's schedule against baseline historical schedule.
    """

    baseline_file = config.paths["history"] / "historical_schedule.parquet"

    try:
        baseline_schedule = pd.read_parquet(baseline_file)
    except FileNotFoundError:
        logger.warning(f"Baseline schedule not found at {baseline_file}")
        baseline_schedule = pd.DataFrame()

    if today_schedule.empty:
        logger.warning("Today schedule is empty; nothing to compare.")
        return pd.DataFrame()

    if not baseline_schedule.empty:
        merged = today_schedule.merge(
            baseline_schedule,
            on=["GAME_ID"],
            how="left",
            indicator=True,
        )
        new_games = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
        logger.info(f"Found {len(new_games)} new games vs baseline.")
    else:
        new_games = today_schedule
        logger.info("No baseline found; all games considered new.")

    return new_games
