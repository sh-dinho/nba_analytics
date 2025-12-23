from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Team Name Normalization
# File: src/utils/team_names.py
#
# Description:
#     Normalize team names from various sources (ESPN, NBA
#     Stats full names, odd variants) into a single canonical
#     format: official NBA tricodes (e.g., BOS, LAL, MIA).
#
#     Public API:
#       - normalize_team(name: str) -> str | None
#       - normalize_schedule(df) -> df
# ============================================================

from typing import Optional

import pandas as pd
from loguru import logger

# Official NBA tricodes
NBA_TRICODES = {
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
}

# Map ESPN / full names / variants → tricode
TEAM_NAME_MAP = {
    # Atlanta
    "Atlanta Hawks": "ATL",
    "ATL": "ATL",
    # Boston
    "Boston Celtics": "BOS",
    "BOS": "BOS",
    # Brooklyn
    "Brooklyn Nets": "BKN",
    "Brooklyn": "BKN",
    "BKN": "BKN",
    # Charlotte
    "Charlotte Hornets": "CHA",
    "Charlotte": "CHA",
    "CHA": "CHA",
    # Chicago
    "Chicago Bulls": "CHI",
    "Chicago": "CHI",
    "CHI": "CHI",
    # Cleveland
    "Cleveland Cavaliers": "CLE",
    "Cleveland": "CLE",
    "CLE": "CLE",
    # Dallas
    "Dallas Mavericks": "DAL",
    "Dallas": "DAL",
    "DAL": "DAL",
    # Denver
    "Denver Nuggets": "DEN",
    "Denver": "DEN",
    "DEN": "DEN",
    # Detroit
    "Detroit Pistons": "DET",
    "Detroit": "DET",
    "DET": "DET",
    # Golden State
    "Golden State Warriors": "GSW",
    "Golden State": "GSW",
    "GS Warriors": "GSW",
    "GSW": "GSW",
    # Houston
    "Houston Rockets": "HOU",
    "Houston": "HOU",
    "HOU": "HOU",
    # Indiana
    "Indiana Pacers": "IND",
    "Indiana": "IND",
    "IND": "IND",
    # LA Clippers
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
    "LAC": "LAC",
    # LA Lakers
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "LAL": "LAL",
    # Memphis
    "Memphis Grizzlies": "MEM",
    "Memphis": "MEM",
    "MEM": "MEM",
    # Miami
    "Miami Heat": "MIA",
    "Miami": "MIA",
    "MIA": "MIA",
    # Milwaukee
    "Milwaukee Bucks": "MIL",
    "Milwaukee": "MIL",
    "MIL": "MIL",
    # Minnesota
    "Minnesota Timberwolves": "MIN",
    "Minnesota": "MIN",
    "MIN": "MIN",
    # New Orleans
    "New Orleans Pelicans": "NOP",
    "New Orleans": "NOP",
    "NO Pelicans": "NOP",
    "NOP": "NOP",
    # New York
    "New York Knicks": "NYK",
    "NY Knicks": "NYK",
    "New York": "NYK",
    "NYK": "NYK",
    # Oklahoma City
    "Oklahoma City Thunder": "OKC",
    "Oklahoma City": "OKC",
    "OKC Thunder": "OKC",
    "OKC": "OKC",
    # Orlando
    "Orlando Magic": "ORL",
    "Orlando": "ORL",
    "ORL": "ORL",
    # Philadelphia
    "Philadelphia 76ers": "PHI",
    "Philadelphia": "PHI",
    "PHI": "PHI",
    # Phoenix
    "Phoenix Suns": "PHX",
    "Phoenix": "PHX",
    "PHX Suns": "PHX",
    "PHX": "PHX",
    # Portland
    "Portland Trail Blazers": "POR",
    "Portland": "POR",
    "POR": "POR",
    # Sacramento
    "Sacramento Kings": "SAC",
    "Sacramento": "SAC",
    "SAC": "SAC",
    # San Antonio
    "San Antonio Spurs": "SAS",
    "San Antonio": "SAS",
    "SA Spurs": "SAS",
    "SAS": "SAS",
    # Toronto
    "Toronto Raptors": "TOR",
    "Toronto": "TOR",
    "TOR": "TOR",
    # Utah
    "Utah Jazz": "UTA",
    "Utah": "UTA",
    "UTA": "UTA",
    # Washington
    "Washington Wizards": "WAS",
    "Washington": "WAS",
    "WSH Wizards": "WAS",
    "WAS": "WAS",
}


def normalize_team(name: str) -> Optional[str]:
    """
    Normalize a raw team name from any known source
    into an official NBA tricode.
    Returns None if the name is unknown.
    """
    if not isinstance(name, str):
        return None

    name = name.strip()

    # Direct mapping
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]

    # Already a valid tricode?
    upper = name.upper()
    if upper in NBA_TRICODES:
        return upper

    logger.warning(f"[TeamNames] Unknown team name variant: '{name}'")
    return None


def normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize 'home_team' and 'away_team' columns in a schedule DataFrame
    to official tricodes. Drops rows where normalization fails.
    """
    df = df.copy()

    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(normalize_team)
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(normalize_team)

    before = len(df)
    df = df.dropna(subset=["home_team", "away_team"]).reset_index(drop=True)
    after = len(df)

    if after < before:
        logger.warning(
            f"[TeamNames] Dropped {before - after} rows due to unknown team names "
            f"during schedule normalization."
        )

    return df
