# ============================================================
# File: src/ingestion/pipeline.py
# ============================================================

import pandas as pd
from datetime import date, timedelta
from loguru import logger
from src.ingestion.collector import NBACollector
from src.config.paths import SCHEDULE_SNAPSHOT, LONG_SNAPSHOT


class IngestionPipeline:
    def __init__(self):
        self.collector = NBACollector()

    def run_today_ingestion(self):
        """Fetches yesterday and today, ensuring data types are correct."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        all_new_games = []
        for d in [yesterday, today]:
            # Force the collector to use the '00' LeagueID
            df = self.collector.fetch_scoreboard(d)
            if not df.empty:
                all_new_games.append(df)

        if not all_new_games:
            logger.warning("No games found for yesterday or today.")
            return

        combined_df = pd.concat(all_new_games).drop_duplicates(subset=["game_id"])

        # FIX 1: Force column to datetime, then to date objects (not strings!)
        combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.date

        self._update_historical_data(combined_df)
        self._update_schedule(combined_df)

    def _update_schedule(self, new_df: pd.DataFrame):
        """Saves the schedule snapshot specifically for the prediction model."""
        # FIX 2: Parquet preserves 'object' types; we must ensure they are dates
        # Some engines convert dates to strings; we'll force them back on load in the model
        new_df.to_parquet(SCHEDULE_SNAPSHOT, index=False, engine="pyarrow")
        logger.success(f"Verified {len(new_df)} games saved to schedule.")

    def _update_long_history(self, results_df: pd.DataFrame):
        """Converts scoreboard format to 'Long' format and merges with history."""
        long_rows = []
        for _, row in results_df.iterrows():
            # Home Team Entry
            long_rows.append(
                {
                    "game_id": row["game_id"],
                    "date": row["date"],
                    "team": row["home_team"],
                    "opponent": row["away_team"],
                    "is_home": 1,
                    "points_for": row["home_score"],
                    "points_against": row["away_score"],
                    "won": 1 if row["home_score"] > row["away_score"] else 0,
                }
            )
            # Away Team Entry
            long_rows.append(
                {
                    "game_id": row["game_id"],
                    "date": row["date"],
                    "team": row["away_team"],
                    "opponent": row["home_team"],
                    "is_home": 0,
                    "points_for": row["away_score"],
                    "points_against": row["home_score"],
                    "won": 1 if row["away_score"] > row["home_score"] else 0,
                }
            )

        new_long_df = pd.DataFrame(long_rows)
        new_long_df["date"] = pd.to_datetime(new_long_df["date"]).dt.date

        if LONG_SNAPSHOT.exists():
            existing_df = pd.read_parquet(LONG_SNAPSHOT)
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
            combined = pd.concat([existing_df, new_long_df]).drop_duplicates(
                subset=["game_id", "team"]
            )
        else:
            combined = new_long_df

        combined.to_parquet(LONG_SNAPSHOT, index=False)
        logger.info(f"Historical training data updated → {LONG_SNAPSHOT}")
