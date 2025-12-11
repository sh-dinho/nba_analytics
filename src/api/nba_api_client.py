# ============================================================
# File: src/api/nba_api_client.py
# Purpose: Fetch NBA data (schedule + boxscores)
# ============================================================

import logging
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
import pandas as pd

logger = logging.getLogger("api.nba_api_client")


def fetch_season_games(season: int) -> pd.DataFrame:
    """Fetch all games for a season (schedule only)."""
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = gamefinder.get_data_frames()[0]
    return df


def fetch_boxscores(game_ids: list[str]) -> pd.DataFrame:
    """Fetch boxscores for a list of game IDs, including WL outcome."""
    all_boxscores = []
    for gid in game_ids:
        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
            df_box = box.get_data_frames()[0]
            all_boxscores.append(df_box)
        except Exception as e:
            logger.warning("Failed to fetch boxscore for %s: %s", gid, e)
    if not all_boxscores:
        return pd.DataFrame()
    return pd.concat(all_boxscores, ignore_index=True)
