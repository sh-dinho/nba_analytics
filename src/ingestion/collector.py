# ============================================================
# File: src/ingestion/collector.py
# Purpose: Robust NBA data collector (history + live schedule)
# ============================================================

import time
import random
import pandas as pd
from datetime import date
from loguru import logger
from typing import List

try:
    from nba_api.stats.endpoints import (
        LeagueGameLog,
        ScoreboardV2,
        BoxScoreSummaryV2,
    )
    from nba_api.stats.static import teams as static_teams

    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    logger.error("nba_api not found. Install via: pip install nba_api")


class NBADataCollector:
    """
    Produces a unified WIDE-format schema:

        game_id
        date
        home_team
        away_team
        home_score
        away_score
        status
        season

    Feature builder will later convert this to LONG format.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.team_map = self._build_team_map()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _build_team_map(self):
        if not NBA_API_AVAILABLE:
            return {}
        return {t["id"]: t["full_name"] for t in static_teams.get_teams()}

    def _safe_call(self, func, label: str):
        """Retry logic with exponential backoff."""
        for i in range(5):
            try:
                time.sleep(random.uniform(0.4, 1.0))
                return func()
            except Exception as e:
                wait = (2**i) + random.random()
                logger.warning(
                    f"{label} failed (attempt {i+1}). Retrying in {wait:.1f}s... Error: {e}"
                )
                time.sleep(wait)
        logger.error(f"{label} failed permanently.")
        return None

    # ---------------------------------------------------------
    # Historical ingestion
    # ---------------------------------------------------------

    def fetch_all_history(self) -> pd.DataFrame:
        """Fetches all seasons from 2000 → current year."""
        current_year = date.today().year
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2000, current_year + 1)]
        return self.fetch_history(seasons)

    def fetch_history(self, seasons: List[str]) -> pd.DataFrame:
        """Fetch multiple seasons of LeagueGameLog."""
        all_rows = []

        for season in seasons:
            logger.info(f"📥 Pulling History: {season}")

            df = self._safe_call(
                lambda: LeagueGameLog(
                    season=season, season_type_all_star="Regular Season"
                ).get_data_frames()[0],
                f"History {season}",
            )

            if df is not None and not df.empty:
                all_rows.append(self._normalize_history(df, season))

        if not all_rows:
            return pd.DataFrame()

        return pd.concat(all_rows, ignore_index=True)

    def _normalize_history(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """
        Convert LeagueGameLog into WIDE format.
        LeagueGameLog gives one row per TEAM per GAME.
        We pivot it into one row per GAME.
        """

        df = df.rename(
            columns={
                "GAME_ID": "game_id",
                "GAME_DATE": "date",
                "TEAM_ID": "team_id",
                "MATCHUP": "matchup",
                "PTS": "points_for",
                "WL": "result",
            }
        )

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Split into home/away rows
        home = df[df["matchup"].str.contains("vs.")].copy()
        away = df[df["matchup"].str.contains("@")].copy()

        # Extract opponent team_id
        home["opponent_id"] = away["team_id"].values
        away["opponent_id"] = home["team_id"].values

        # Merge scores
        merged = home.merge(
            away[["game_id", "points_for"]],
            on="game_id",
            suffixes=("_home", "_away"),
        )

        merged["home_team"] = merged["team_id"]
        merged["away_team"] = merged["opponent_id"]

        merged["home_score"] = merged["points_for_home"]
        merged["away_score"] = merged["points_for_away"]

        merged["status"] = "final"
        merged["season"] = season

        # Map team IDs → names
        merged["home_team"] = merged["home_team"].map(self.team_map)
        merged["away_team"] = merged["away_team"].map(self.team_map)

        return merged[
            [
                "game_id",
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "status",
                "season",
            ]
        ]

    # ---------------------------------------------------------
    # Today’s games (with live scores)
    # ---------------------------------------------------------

    def fetch_today(self) -> pd.DataFrame:
        return self.fetch_scoreboard(date.today())

    def fetch_scoreboard(self, target_date: date) -> pd.DataFrame:
        """Fetch schedule + live scores for a given date."""
        date_str = target_date.strftime("%Y-%m-%d")
        logger.info(f"📅 Pulling Scoreboard: {date_str}")

        sb = self._safe_call(
            lambda: ScoreboardV2(game_date=date_str),
            f"Scoreboard {date_str}",
        )

        if sb is None:
            return pd.DataFrame()

        header = sb.game_header.get_data_frame()
        if header.empty:
            return pd.DataFrame()

        # Normalize
        df = header.rename(
            columns={
                "GAME_ID": "game_id",
                "GAME_DATE_EST": "date",
                "HOME_TEAM_ID": "home_id",
                "VISITOR_TEAM_ID": "away_id",
                "GAME_STATUS_TEXT": "status",
            }
        )

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Map team names
        df["home_team"] = df["home_id"].map(self.team_map)
        df["away_team"] = df["away_id"].map(self.team_map)

        # Fetch live scores
        df["home_score"] = None
        df["away_score"] = None

        for idx, row in df.iterrows():
            box = self._safe_call(
                lambda: BoxScoreSummaryV2(game_id=row["game_id"]),
                f"BoxScore {row['game_id']}",
            )
            if box is None:
                continue

            line = box.game_summary.get_data_frame()
            if not line.empty:
                df.at[idx, "home_score"] = line["PTS_HOME"].iloc[0]
                df.at[idx, "away_score"] = line["PTS_AWAY"].iloc[0]

        df["season"] = f"{target_date.year}-{str(target_date.year+1)[-2:]}"

        return df[
            [
                "game_id",
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "status",
                "season",
            ]
        ]
