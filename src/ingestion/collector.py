# ============================================================
# File: src/ingestion/collector.py
# Purpose: Optimized NBA data collector
# ============================================================

import time
import random
import pandas as pd
from datetime import date
from loguru import logger
from typing import Dict

try:
    from nba_api.stats.endpoints import ScoreboardV3
    from nba_api.stats.static import teams as static_teams

    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False


class NBACollector:
    def __init__(self):
        self.team_map = self._build_team_map()

    def _build_team_map(self) -> Dict[int, str]:
        if not NBA_API_AVAILABLE:
            return {}
        return {t["id"]: t["full_name"] for t in static_teams.get_teams()}

    def _safe_call(self, func, label: str):
        """Standard retry logic with exponential backoff."""
        for i in range(5):
            try:
                time.sleep(random.uniform(0.5, 1.0))  # Polite delay
                return func()
            except Exception as e:
                wait = (2**i) + random.random()
                logger.warning(f"{label} failed. Retrying in {wait:.1f}s... Error: {e}")
                time.sleep(wait)
        return None

    def fetch_scoreboard(self, target_date: date) -> pd.DataFrame:
        """Fetches ScoreboardV3 with case-insensitive column mapping."""
        date_str = target_date.strftime("%Y-%m-%d")
        logger.info(f"📅 Fetching ScoreboardV3: {date_str}")

        sb = self._safe_call(
            lambda: ScoreboardV3(game_date=date_str), f"Scoreboard {date_str}"
        )
        if sb is None:
            return pd.DataFrame()

        all_dfs = sb.get_data_frames()

        # Helper to find a column regardless of casing (e.g., 'gameId' vs 'GAME_ID')
        def get_col(df, target):
            for col in df.columns:
                if col.lower() == target.lower():
                    return df[col]
            return None

        # Search for the table that has game identifiers
        df = next(
            (
                d
                for d in all_dfs
                if any(c.lower() == "gameid" for c in d.columns)
                and any(c.lower() == "hometeamid" for c in d.columns)
            ),
            pd.DataFrame(),
        )

        if df.empty:
            logger.warning(f"No valid game header found in V3 datasets for {date_str}")
            return pd.DataFrame()

        try:
            # Build the dataframe using the helper to catch any casing
            processed = pd.DataFrame()
            processed["game_id"] = get_col(df, "gameId")
            processed["date"] = pd.to_datetime(get_col(df, "gameEt")).dt.date
            processed["home_id"] = get_col(df, "homeTeamId")
            processed["away_id"] = get_col(df, "awayTeamId")
            processed["home_score"] = get_col(df, "homeTeamScore").fillna(0)
            processed["away_score"] = get_col(df, "awayTeamScore").fillna(0)
            processed["status"] = get_col(df, "gameStatusText")

            # Map to names
            processed["home_team"] = processed["home_id"].map(self.team_map)
            processed["away_team"] = processed["away_id"].map(self.team_map)

            # Remove games that didn't map (usually All-Star or non-NBA teams)
            final_df = processed.dropna(subset=["home_team", "away_team"])

            if not final_df.empty:
                logger.info(f"Successfully collected {len(final_df)} games.")

            return final_df

        except Exception as e:
            logger.error(f"Indestructible mapping failed: {e}")
            return pd.DataFrame()
