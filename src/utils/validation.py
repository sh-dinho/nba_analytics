# ============================================================
# File: src/utils/validation.py
# Purpose: Validation helpers for game IDs, seasons, and file extensions
# Project: nba_analysis
# Version: 1.1 (adds dependencies section + clearer error handling)
#
# Dependencies:
# - datetime (standard library)
# - pathlib (standard library)
# - typing (standard library)
# - pandas
# ============================================================

from datetime import datetime
from pathlib import Path
from typing import Iterable
import pandas as pd


def validate_game_ids(game_ids: str | Iterable[str]) -> list[str]:
    """
    Validate NBA game IDs ensuring they are 10-digit numeric strings.

    Args:
        game_ids (str | Iterable[str]): Comma-separated string or iterable of game IDs.

    Returns:
        list[str]: List of validated game IDs.

    Raises:
        ValueError: If any game ID is not a 10-digit numeric string.
    """
    if not game_ids:
        return []

    if isinstance(game_ids, str):
        ids = [gid.strip() for gid in game_ids.split(",") if gid.strip()]
    else:
        ids = [str(gid).strip() for gid in game_ids if str(gid).strip()]

    for gid in ids:
        if not gid.isdigit() or len(gid) != 10:
            raise ValueError(f"Invalid game_id format: {gid}. Expected a 10-digit numeric string.")

    return ids


def validate_season(season: str) -> str:
    """
    Validate and format a season year into NBA season format (e.g., '2022' -> '2021-22').

    Args:
        season (str): Season year as a string.

    Returns:
        str: Season formatted as 'YYYY-YY'.

    Raises:
        ValueError: If season is missing, not an integer, or outside the valid range.
    """
    if not season:
        raise ValueError("Season year must be provided.")

    try:
        year = int(season)
    except ValueError:
        raise ValueError(f"Invalid season: {season}. Must be an integer year.")

    if year < 1996 or year > 2030:
        raise ValueError("Season must be between 1996 and 2030.")

    return f"{year-1}-{str(year)[-2:]}"


def validate_file_extension(path: str | Path, allowed: list[str]) -> Path:
    """
    Validate that a file path has an allowed extension.

    Args:
        path (str | Path): File path to validate.
        allowed (list[str]): List of allowed extensions (e.g., ['.csv', '.parquet']).

    Returns:
        Path: Validated Path object.

    Raises:
        ValueError: If the file extension is not in the allowed list.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext not in [a.lower() for a in allowed]:
        raise ValueError(f"Invalid file extension: {ext}. Allowed extensions are: {', '.join(allowed)}")

    return p
