from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Team Name Normalization (Canonical)
# File: src/utils/team_names.py
# Author: Sadiq
#
# Description:
#     Normalize team names from various sources (ESPN, NBA Stats,
#     betting feeds, odd variants) into a single canonical format:
#     official NBA tricodes (e.g., BOS, LAL).
#
#     Public API:
#       - normalize_team(name: str) -> str | None
#       - normalize_schedule(df) -> df
#       - validate_team_names(names) -> list[str]
#
#     Enhancements:
#       • case-insensitive matching
#       • whitespace-insensitive matching
#       • accent-insensitive matching
#       • optional strict mode for schedule normalization
# ============================================================

from typing import Optional, Iterable
import pandas as pd
from loguru import logger
import unicodedata


# ------------------------------------------------------------
# Official NBA tricodes (canonical)
# ------------------------------------------------------------
NBA_TRICODES = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
    "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS",
    "TOR", "UTA", "WAS",
}


# ------------------------------------------------------------
# Mapping: ESPN / NBA Stats / betting feeds → tricode
# (canonical, case-insensitive)
# ------------------------------------------------------------
TEAM_NAME_MAP = {
    # Atlanta
    "atlanta hawks": "ATL", "atlanta": "ATL", "atl": "ATL",

    # Boston
    "boston celtics": "BOS", "boston": "BOS", "bos": "BOS",

    # Brooklyn
    "brooklyn nets": "BKN", "brooklyn": "BKN", "bkn": "BKN",

    # Charlotte
    "charlotte hornets": "CHA", "charlotte": "CHA", "cha": "CHA",

    # Chicago
    "chicago bulls": "CHI", "chicago": "CHI", "chi": "CHI",

    # Cleveland
    "cleveland cavaliers": "CLE", "cleveland": "CLE", "cle": "CLE",

    # Dallas
    "dallas mavericks": "DAL", "dallas": "DAL", "dal": "DAL",

    # Denver
    "denver nuggets": "DEN", "denver": "DEN", "den": "DEN",

    # Detroit
    "detroit pistons": "DET", "detroit": "DET", "det": "DET",

    # Golden State
    "golden state warriors": "GSW", "golden state": "GSW",
    "gs warriors": "GSW", "gsw": "GSW",

    # Houston
    "houston rockets": "HOU", "houston": "HOU", "hou": "HOU",

    # Indiana
    "indiana pacers": "IND", "indiana": "IND", "ind": "IND",

    # LA Clippers
    "los angeles clippers": "LAC", "la clippers": "LAC", "lac": "LAC",

    # LA Lakers
    "los angeles lakers": "LAL", "la lakers": "LAL", "lal": "LAL",

    # Memphis
    "memphis grizzlies": "MEM", "memphis": "MEM", "mem": "MEM",

    # Miami
    "miami heat": "MIA", "miami": "MIA", "mia": "MIA",

    # Milwaukee
    "milwaukee bucks": "MIL", "milwaukee": "MIL", "mil": "MIL",

    # Minnesota
    "minnesota timberwolves": "MIN", "minnesota": "MIN", "min": "MIN",

    # New Orleans
    "new orleans pelicans": "NOP", "new orleans": "NOP",
    "no pelicans": "NOP", "nop": "NOP",

    # New York
    "new york knicks": "NYK", "ny knicks": "NYK",
    "new york": "NYK", "nyk": "NYK",

    # Oklahoma City
    "oklahoma city thunder": "OKC", "oklahoma city": "OKC",
    "okc thunder": "OKC", "okc": "OKC",

    # Orlando
    "orlando magic": "ORL", "orlando": "ORL", "orl": "ORL",

    # Philadelphia
    "philadelphia 76ers": "PHI", "philadelphia": "PHI", "phi": "PHI",

    # Phoenix
    "phoenix suns": "PHX", "phoenix": "PHX",
    "phx suns": "PHX", "phx": "PHX",

    # Portland
    "portland trail blazers": "POR", "portland": "POR", "por": "POR",

    # Sacramento
    "sacramento kings": "SAC", "sacramento": "SAC", "sac": "SAC",

    # San Antonio
    "san antonio spurs": "SAS", "san antonio": "SAS",
    "sa spurs": "SAS", "sas": "SAS",

    # Toronto
    "toronto raptors": "TOR", "toronto": "TOR", "tor": "TOR",

    # Utah
    "utah jazz": "UTA", "utah": "UTA", "uta": "UTA",

    # Washington
    "washington wizards": "WAS", "washington": "WAS",
    "wsh wizards": "WAS", "was": "WAS",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _clean(name: str) -> str:
    """Lowercase, strip, remove accents."""
    name = name.strip().lower()
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def normalize_team(name: str) -> Optional[str]:
    """
    Normalize a raw team name from any known source into an NBA tricode.
    Returns None if the name is unknown.
    """
    if not isinstance(name, str):
        return None

    cleaned = _clean(name)

    # Direct mapping
    if cleaned in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[cleaned]

    # Already a tricode?
    upper = cleaned.upper()
    if upper in NBA_TRICODES:
        return upper

    logger.warning(f"[TeamNames] Unknown team name variant: '{name}'")
    return None


def normalize_schedule(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Normalize 'home_team' and 'away_team' columns in a schedule DataFrame.
    Drops rows where normalization fails unless strict=True.
    """
    df = df.copy()

    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team)

    before = len(df)

    if strict:
        if df.isna().any().any():
            unknowns = df[df.isna().any(axis=1)]
            raise ValueError(f"Unknown team names in strict mode:\n{unknowns}")
        return df

    df = df.dropna(subset=["home_team", "away_team"]).reset_index(drop=True)
    after = len(df)

    if after < before:
        logger.warning(f"[TeamNames] Dropped {before - after} rows due to unknown team names.")

    return df


def validate_team_names(names: Iterable[str]) -> list[str]:
    """Return a list of unknown team names for QA."""
    return [name for name in names if normalize_team(name) is None]