# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Automated canonicalization of NBA team names.
# ============================================================

import pandas as pd
from functools import lru_cache
from loguru import logger


@lru_cache(maxsize=1)
def load_team_reference() -> pd.DataFrame:
    try:
        url = "https://stats.nba.com/stats/commonteamyears?LeagueID=00"
        df = pd.read_json(url)
        return df
    except Exception as e:
        logger.error(f"Failed to load team reference: {e}")
        return pd.DataFrame()


def canonicalize_team_name(name: str) -> str:
    ref = load_team_reference()
    if ref.empty:
        return name

    name_norm = name.lower().strip()

    for _, row in ref.iterrows():
        if name_norm in {
            str(row.get("TEAM_CITY", "")).lower().strip(),
            str(row.get("TEAM_NAME", "")).lower().strip(),
            str(row.get("TEAM_NICKNAME", "")).lower().strip(),
        }:
            return f"{row.get('TEAM_CITY')} {row.get('TEAM_NAME')}"

    return name
