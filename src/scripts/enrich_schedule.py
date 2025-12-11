# ============================================================
# File: src/scripts/enrich_schedule.py
# Purpose: Fetch and enrich NBA schedule with WL outcomes
# ============================================================

import logging
import os
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamefinder

logger = logging.getLogger("scripts.enrich_schedule")
logging.basicConfig(level=logging.INFO)


def fetch_game_results(season: int, retries: int = 3, delay: int = 10) -> pd.DataFrame:
    """
    Fetch game results for a given season with retry logic.
    Returns a DataFrame or empty DataFrame if API fails.
    """
    for attempt in range(retries):
        try:
            logger.info(
                "Fetching season games with WL outcomes for %s (attempt %d)",
                season,
                attempt + 1,
            )
            gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
            df = gamefinder.get_data_frames()[0]
            return df
        except requests.exceptions.ReadTimeout:
            logger.warning(
                "Timeout fetching season %s games. Retrying in %s seconds...",
                season,
                delay,
            )
            time.sleep(delay)
        except Exception as e:
            logger.error("Unexpected error fetching season %s games: %s", season, e)
            return pd.DataFrame()

    logger.error("Failed to fetch season %s games after %d retries", season, retries)
    return pd.DataFrame()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season", type=int, required=True, help="Season year (e.g., 2025)"
    )
    args = parser.parse_args()

    season = args.season
    out_path = "data/cache/schedule.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- Try to fetch results ---
    results = fetch_game_results(season)

    if results.empty:
        logger.warning("No schedule data retrieved for season %s", season)

        # --- Fallback: use cached file if available ---
        if os.path.exists(out_path):
            logger.info("Using cached schedule file at %s", out_path)
            return
        else:
            # --- Write empty file so downstream steps succeed ---
            empty_cols = [
                "GAME_ID",
                "GAME_DATE_EST",
                "HOME_TEAM_ID",
                "VISITOR_TEAM_ID",
                "WL",
            ]
            pd.DataFrame(columns=empty_cols).to_csv(out_path, index=False)
            logger.warning("Empty schedule file written to %s", out_path)
    else:
        results.to_csv(out_path, index=False)
        logger.info("Schedule saved to %s (rows: %d)", out_path, len(results))


if __name__ == "__main__":
    main()
