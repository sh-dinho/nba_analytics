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
from src.schemas import ENRICHED_SCHEDULE_COLUMNS

logger = logging.getLogger("scripts.enrich_schedule")
logging.basicConfig(level=logging.INFO)


def fetch_game_results(season: int, retries: int = 3, delay: int = 10) -> pd.DataFrame:
    """Fetch game results for a given season with retry logic. Returns a DataFrame or empty DataFrame."""
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
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as e:
            logger.warning(
                "Network error fetching season %s games: %s. Retrying in %s seconds...",
                season,
                e,
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
    parquet_path = "data/cache/historical_schedule.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Try to fetch results
    results = fetch_game_results(season)

    if results.empty:
        logger.warning("No schedule data retrieved for season %s", season)

        # Fallback: use cached file if available
        if os.path.exists(out_path):
            logger.info("Using cached schedule file at %s", out_path)
            return
        else:
            # Write empty files with unified schema
            pd.DataFrame(columns=ENRICHED_SCHEDULE_COLUMNS).to_csv(
                out_path, index=False
            )
            pd.DataFrame(columns=ENRICHED_SCHEDULE_COLUMNS).to_parquet(
                parquet_path, index=False
            )
            logger.warning(
                "Empty schedule file written to %s and %s", out_path, parquet_path
            )
    else:
        # Map and reduce to enriched columns if present
        cols_map = {
            "GAME_ID": "GAME_ID",
            "GAME_DATE": "GAME_DATE",
            "GAME_DATE_EST": "GAME_DATE",
            "HOME_TEAM_ID": "HOME_TEAM_ID",
            "VISITOR_TEAM_ID": "VISITOR_TEAM_ID",
            "PTS": "PTS",
            "PTS_OPP": "PTS_OPP",
            "WL": "WL",
            "SEASON_ID": "SEASON",
        }
        df = pd.DataFrame()
        for src, dst in cols_map.items():
            if src in results.columns:
                df[dst] = results[src]
        # Derive WIN if WL exists
        if "WL" in df.columns and "WIN" not in df.columns:
            df["WIN"] = df["WL"].apply(lambda x: 1 if str(x).upper() == "W" else 0)
        # Ensure all expected columns exist
        for col in ENRICHED_SCHEDULE_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df[ENRICHED_SCHEDULE_COLUMNS]
        df.to_csv(out_path, index=False)
        df.to_parquet(parquet_path, index=False)
        logger.info(
            "Schedule saved to %s and %s (rows: %d)", out_path, parquet_path, len(df)
        )


if __name__ == "__main__":
    main()
