from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Team Name Normalization
# File: src/ingestion/normalizer/team_names.py
# Author: Sadiq
#
# Description:
#     Maps raw team names from external sources (ESPN, NBA API,
#     legacy feeds) into canonical team tricodes used across
#     ingestion and modeling.
# ============================================================

from typing import Dict


TEAM_NAME_TO_TRICODE: Dict[str, str] = {
    # Atlantic
    "Boston Celtics": "BOS", "Boston": "BOS", "Celtics": "BOS",
    "Brooklyn Nets": "BKN", "Brooklyn": "BKN", "Nets": "BKN",
    "New Jersey Nets": "BKN",
    "New York Knicks": "NYK", "New York": "NYK", "Knicks": "NYK",
    "Philadelphia 76ers": "PHI", "Philadelphia": "PHI", "Sixers": "PHI", "76ers": "PHI",
    "Toronto Raptors": "TOR", "Toronto": "TOR", "Raptors": "TOR",

    # Central
    "Chicago Bulls": "CHI", "Chicago": "CHI", "Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Cleveland": "CLE", "Cavs": "CLE",
    "Detroit Pistons": "DET", "Detroit": "DET", "Pistons": "DET",
    "Indiana Pacers": "IND", "Indiana": "IND", "Pacers": "IND",
    "Milwaukee Bucks": "MIL", "Milwaukee": "MIL", "Bucks": "MIL",

    # Southeast
    "Atlanta Hawks": "ATL", "Atlanta": "ATL", "Hawks": "ATL",
    "Charlotte Hornets": "CHA", "Charlotte": "CHA", "Hornets": "CHA",
    "Miami Heat": "MIA", "Miami": "MIA", "Heat": "MIA",
    "Orlando Magic": "ORL", "Orlando": "ORL", "Magic": "ORL",
    "Washington Wizards": "WAS", "Washington": "WAS", "Wizards": "WAS",
    "Washington Bullets": "WAS",

    # Northwest
    "Denver Nuggets": "DEN", "Denver": "DEN", "Nuggets": "DEN",
    "Minnesota Timberwolves": "MIN", "Minnesota": "MIN", "Timberwolves": "MIN", "Wolves": "MIN",
    "Oklahoma City Thunder": "OKC", "Oklahoma City": "OKC", "Thunder": "OKC",
    "Seattle SuperSonics": "OKC",
    "Portland Trail Blazers": "POR", "Portland": "POR", "Blazers": "POR",
    "Utah Jazz": "UTA", "Utah": "UTA", "Jazz": "UTA",

    # Pacific
    "Golden State Warriors": "GSW", "Golden State": "GSW", "Warriors": "GSW",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "LA Lakers": "LAL", "Lakers": "LAL",
    "Phoenix Suns": "PHX", "Phoenix": "PHX", "Suns": "PHX",
    "Sacramento Kings": "SAC", "Sacramento": "SAC", "Kings": "SAC",

    # Southwest
    "Dallas Mavericks": "DAL", "Dallas": "DAL", "Mavericks": "DAL", "Mavs": "DAL",
    "Houston Rockets": "HOU", "Houston": "HOU", "Rockets": "HOU",
    "Memphis Grizzlies": "MEM", "Memphis": "MEM", "Grizzlies": "MEM",
    "New Orleans Pelicans": "NOP", "New Orleans": "NOP", "Pelicans": "NOP",
    "New Orleans Hornets": "NOP",
    "San Antonio Spurs": "SAS", "San Antonio": "SAS", "Spurs": "SAS",
}


def to_tricode(raw: str) -> str:
    """
    Map raw team name to canonical tricode.
    Falls back to the input if no mapping is found.
    """
    if raw is None:
        return raw

    raw_norm = str(raw).strip()

    # Exact match
    if raw_norm in TEAM_NAME_TO_TRICODE:
        return TEAM_NAME_TO_TRICODE[raw_norm]

    # Title-case fallback
    raw_title = raw_norm.title()
    if raw_title in TEAM_NAME_TO_TRICODE:
        return TEAM_NAME_TO_TRICODE[raw_title]

    # Unknown → return normalized input
    return raw_norm