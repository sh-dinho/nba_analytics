# ============================================================
# File: src/scripts/compare_schedule.py
# Purpose: Compare baseline season schedule (CSV) vs API feeds (today + season snapshot)
# ============================================================

import logging
import os
import pandas as pd
from src.schemas import TODAY_SCHEDULE_COLUMNS, ENRICHED_SCHEDULE_COLUMNS, normalize_df
from src.api.nba_api_client import fetch_today_games

logger = logging.getLogger("scripts.compare_schedule")
logging.basicConfig(level=logging.INFO)

BASELINE_PATH = "data/reference/season_schedule.csv"
TODAY_PATH = "data/results/today_schedule.parquet"
SEASON_PATH = "data/cache/schedule.csv"
LOG_PATH = "data/results/reschedule_log.csv"


def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        logger.error("Baseline schedule not found at %s", BASELINE_PATH)
        return pd.DataFrame()
    df = pd.read_csv(BASELINE_PATH)
    logger.info("Loaded baseline schedule (%d rows)", len(df))
    return df


def compare(baseline: pd.DataFrame, live: pd.DataFrame, label: str):
    if baseline.empty or live.empty:
        logger.warning("Skipping comparison for %s (empty data)", label)
        return pd.DataFrame()

    # Normalize baseline
    baseline_cols = ["GAME_ID", "GAME_DATE_EST", "HOME_TEAM", "AWAY_TEAM"]
    for col in baseline_cols:
        if col not in baseline.columns:
            baseline[col] = None

    # Normalize live
    if label == "today":
        live = normalize_df(live, TODAY_SCHEDULE_COLUMNS)
        live = live.rename(
            columns={
                "HOME_TEAM_ABBREVIATION": "HOME_TEAM",
                "AWAY_TEAM_ABBREVIATION": "AWAY_TEAM",
            }
        )
    else:
        live = normalize_df(live, ENRICHED_SCHEDULE_COLUMNS)
        live = live.rename(
            columns={
                "HOME_TEAM_ID": "HOME_TEAM",
                "VISITOR_TEAM_ID": "AWAY_TEAM",
                "GAME_DATE": "GAME_DATE_EST",
            }
        )

    # Merge and detect differences
    merged = baseline.merge(
        live, on="GAME_ID", how="inner", suffixes=("_baseline", "_live")
    )
    changes = merged[
        (merged["GAME_DATE_EST_baseline"] != merged["GAME_DATE_EST_live"])
        | (merged["HOME_TEAM_baseline"] != merged["HOME_TEAM_live"])
        | (merged["AWAY_TEAM_baseline"] != merged["AWAY_TEAM_live"])
    ]

    if not changes.empty:
        logger.warning("Detected %d rescheduled games in %s feed.", len(changes), label)
    else:
        logger.info("No rescheduled games detected in %s feed.", label)

    return changes


def main():
    baseline = load_baseline()
    if baseline.empty:
        return

    all_changes = []

    # --- Compare today feed ---
    if os.path.exists(TODAY_PATH):
        today = pd.read_parquet(TODAY_PATH)
    else:
        today = fetch_today_games()
    changes_today = compare(baseline, today, "today")
    if not changes_today.empty:
        all_changes.append(changes_today)

    # --- Compare season snapshot feed ---
    if os.path.exists(SEASON_PATH):
        season = pd.read_csv(SEASON_PATH)
        changes_season = compare(baseline, season, "season")
        if not changes_season.empty:
            all_changes.append(changes_season)

    # --- Save log ---
    if all_changes:
        combined = pd.concat(all_changes, ignore_index=True)
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        combined.to_csv(LOG_PATH, index=False)
        logger.info("Reschedule log written to %s", LOG_PATH)
    else:
        logger.info("No reschedules detected in any feed.")


if __name__ == "__main__":
    main()
