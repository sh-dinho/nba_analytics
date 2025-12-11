#!/usr/bin/env python
# ============================================================
# File: src/scripts/enrich_schedule.py
# Purpose: Add WL outcomes + boxscore stats to schedule
# ============================================================

import os
import pandas as pd
import datetime
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from src.utils.logging_config import configure_logging

OUT_FILE = "data/cache/historical_schedule_with_results.parquet"


def fetch_game_results(season: int) -> pd.DataFrame:
    """Fetch all games for a season with WL outcomes."""
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = gamefinder.get_data_frames()[0]
    return df


def fetch_boxscore_points(game_id: str) -> pd.DataFrame:
    """Fetch team-level points for a given game."""
    box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    df = box.get_data_frames()[1]  # team stats table
    return df[["GAME_ID", "TEAM_ID", "PTS"]]


def main(season=2025):
    logger = configure_logging(name="scripts.enrich_schedule")
    logger.info("Fetching season games with WL outcomes for %s", season)

    results = fetch_game_results(season)
    if results.empty:
        logger.error("No results data found.")
        return

    # --- Skip future games ---
    today = datetime.date.today()
    results["GAME_DATE"] = pd.to_datetime(results["GAME_DATE"], errors="coerce")
    past_games = results[results["GAME_DATE"] < pd.Timestamp(today)]
    logger.info(
        "Found %d past games (skipping %d future games)",
        len(past_games),
        len(results) - len(past_games),
    )

    # --- Collect boxscores ---
    all_boxscores = []
    for gid in past_games["GAME_ID"].unique():
        try:
            df_box = fetch_boxscore_points(gid)
            all_boxscores.append(df_box)
        except Exception as e:
            logger.warning("Boxscore fetch failed for %s: %s", gid, e)

    if all_boxscores:
        boxscores = pd.concat(all_boxscores, ignore_index=True)
        # Merge points scored
        enriched = past_games.merge(boxscores, on=["GAME_ID", "TEAM_ID"], how="left")
        # Compute opponent points
        enriched["PTS_OPP"] = enriched.groupby("GAME_ID")["PTS"].transform(
            lambda x: x[::-1].values
        )
    else:
        enriched = past_games.copy()
        enriched["PTS"] = None
        enriched["PTS_OPP"] = None

    # Keep relevant columns
    enriched = enriched[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "MATCHUP",
            "WL",
            "PTS",
            "PTS_OPP",
        ]
    ]
    enriched = enriched.rename(columns={"TEAM_ABBREVIATION": "TEAM_NAME"})

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    enriched.to_parquet(OUT_FILE, index=False)
    logger.info("Enriched schedule saved to %s", OUT_FILE)


if __name__ == "__main__":
    main()
