# ============================================================
# File: src/utils/validation.py
# Purpose: Validation helpers for game IDs, seasons, and file extensions
# Project: nba_analysis
# Version: 1.3 (adds combined helper validate_inputs)
# ============================================================

from datetime import datetime
from pathlib import Path
from typing import Iterable
import pandas as pd
import logging

logger = logging.getLogger("validation_utils")


# --- Existing helpers (unchanged) ---
def validate_game_ids(game_ids: str | Iterable[str]) -> list[str]:
    if not game_ids:
        return []
    if isinstance(game_ids, str):
        ids = [gid.strip() for gid in game_ids.split(",") if gid.strip()]
    elif isinstance(game_ids, Iterable):
        ids = [str(gid).strip() for gid in game_ids if str(gid).strip()]
    else:
        raise TypeError("game_ids must be a string or iterable of strings")
    for gid in ids:
        if not gid.isdigit() or len(gid) != 10:
            raise ValueError(
                f"Invalid game_id format: {gid}. Expected a 10-digit numeric string."
            )
    logger.info("Validated %d game IDs", len(ids))
    return ids


def validate_season(season: str) -> str:
    if not season:
        raise ValueError("Season year must be provided.")
    if "-" in season:
        return season
    try:
        year = int(season)
    except ValueError:
        raise ValueError(f"Invalid season: {season}. Must be an integer year.")
    if year < 1996 or year > 2030:
        raise ValueError(
            f"Invalid season year: {season}. Must be between 1996 and 2030."
        )
    formatted = f"{year-1}-{str(year)[-2:]}"
    logger.info("Validated season %s -> %s", season, formatted)
    return formatted


def validate_file_extension(path: str | Path, allowed: list[str]) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path object")
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in [a.lower() for a in allowed]:
        raise ValueError(
            f"Invalid file extension: {ext}. Allowed extensions are: {', '.join(allowed)}"
        )
    logger.info("Validated file extension %s for path %s", ext, p)
    return p


# --- New combined helper ---
def validate_inputs(
    game_ids: str | Iterable[str] = None,
    season: str = None,
    path: str | Path = None,
    allowed_extensions: list[str] = None,
) -> dict:
    """
    Combined helper: validate game IDs, season, and file extension in one call.

    Args:
        game_ids (str | Iterable[str], optional): Game IDs to validate.
        season (str, optional): Season year or formatted string.
        path (str | Path, optional): File path to validate.
        allowed_extensions (list[str], optional): Allowed file extensions.

    Returns:
        dict: Dictionary with validated values.
    """
    results = {}
    if game_ids is not None:
        results["game_ids"] = validate_game_ids(game_ids)
    if season is not None:
        results["season"] = validate_season(season)
    if path is not None and allowed_extensions is not None:
        results["path"] = validate_file_extension(path, allowed_extensions)
    logger.info("Validated inputs: %s", results)
    return results
