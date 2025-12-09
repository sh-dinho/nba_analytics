# ============================================================
# File: src/utils/validation.py
# Purpose: Input validation helpers for NBA AI project
# Project: nba_analysis
# ============================================================

from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
from datetime import datetime


def validate_game_ids(game_ids: str | Iterable[str]) -> list[str]:
    """Validate and normalize a list of NBA game IDs."""
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
    """Validate a season year and return NBA API season string."""
    if not season:
        raise ValueError("Season year must be provided.")
    try:
        year = int(season)
    except ValueError:
        raise ValueError(f"Invalid season: {season}. Must be an integer year.")
    if year < 1996 or year > 2030:
        raise ValueError("Season must be between 1996 and 2030.")
    return f"{year-1}-{str(year)[-2:]}"


def validate_output_dir(out_dir: str | Path) -> Path:
    """Ensure the output directory exists and is writable."""
    p = Path(out_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise ValueError(f"Output directory {p} is not writable: {e}") from e
    return p


def validate_file_extension(path: str | Path, allowed: list[str]) -> Path:
    """Validate that a file has one of the allowed extensions."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in [a.lower() for a in allowed]:
        raise ValueError(f"Invalid file extension: {ext}. Allowed extensions are: {', '.join(allowed)}")
    return p


def validate_dataframe_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Validate that a DataFrame contains all required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def validate_numeric_range(value: float, min_val: float, max_val: float, name: str = "value") -> float:
    """Validate that a numeric value falls within a given range [min_val, max_val]."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
    if value < min_val or value > max_val:
        raise ValueError(f"{name}={value} out of range [{min_val}, {max_val}]")
    return value


def validate_date(date_str: str, fmt: str = "%Y-%m-%d",
                  min_date: Optional[str] = None,
                  max_date: Optional[str] = None,
                  name: str = "date") -> datetime:
    """Validate that a date string matches the given format and optionally falls within a range."""
    try:
        dt = datetime.strptime(date_str, fmt)
    except ValueError:
        raise ValueError(f"Invalid {name}: '{date_str}'. Expected format {fmt}.")
    if min_date:
        min_dt = datetime.strptime(min_date, fmt)
        if dt < min_dt:
            raise ValueError(f"{name}={date_str} is before minimum allowed {min_date}.")
    if max_date:
        max_dt = datetime.strptime(max_date, fmt)
        if dt > max_dt:
            raise ValueError(f"{name}={date_str} is after maximum allowed {max_date}.")
    return dt


def validate_team_code(team_code: str) -> str:
    """Validate that a team code is a valid NBA team abbreviation."""
    valid_codes = {
        "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET",
        "GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN",
        "NOP","NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS",
        "TOR","UTA","WAS"
    }
    code = team_code.strip().upper()
    if code not in valid_codes:
        raise ValueError(f"Invalid team code: {team_code}. Must be one of {', '.join(sorted(valid_codes))}.")
    return code


def validate_player_id(player_id: str) -> str:
    """Validate that a player ID is numeric and has a valid length (7–10 digits)."""
    pid = player_id.strip()
    if not pid.isdigit() or not (7 <= len(pid) <= 10):
        raise ValueError(f"Invalid player_id format: {player_id}. Expected a 7–10 digit numeric string.")
    return pid


def validate_stat_category(stat: str) -> str:
    """Validate that a stat category is one of the recognized NBA stats."""
    valid_stats = {
        "points","rebounds","assists","steals","blocks",
        "turnovers","minutes","fgm","fga","fg_pct",
        "three_pm","three_pa","three_pct","ftm","fta","ft_pct",
        "plus_minus","efficiency"
    }
    stat_norm = stat.strip().lower()
    if stat_norm not in valid_stats:
        raise ValueError(f"Invalid stat category: {stat}. Must be one of {', '.join(sorted(valid_stats))}.")
    return stat_norm


def validate_lineup(player_ids: Iterable[str]) -> list[str]:
    """Validate that a list of player IDs forms a valid lineup (exactly 5 unique valid IDs)."""
    ids = [validate_player_id(pid) for pid in player_ids]
    if len(ids) != 5:
        raise ValueError(f"Lineup must contain exactly 5 players, got {len(ids)}.")
    if len(set(ids)) != 5:
        raise ValueError("Lineup contains duplicate player IDs.")
    return ids


def validate_schedule(schedule: Iterable[dict]) -> list[dict]:
    """Validate a schedule of games with date, home, and away team codes."""
    validated = []
    for game in schedule:
        if "date" not in game or "home" not in game or "away" not in game:
            raise ValueError(f"Game missing required fields: {game}")
        dt = validate_date(game["date"])
        home = validate_team_code(game["home"])
        away = validate_team_code(game["away"])
        if home == away:
            raise ValueError(f"Invalid game: home and away teams cannot be the same ({home}).")
        validated.append({"date": dt.strftime("%Y-%m-%d"), "home": home, "away": away})
    return validated


def validate_config(config: dict, required_keys: list[str]) -> dict:
    """
    Validate a configuration dictionary.
    - Ensures required keys exist.
    - Validates values using appropriate helpers when possible.
    Returns the normalized config if valid.
    """
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Example validations
    if "season" in config:
        config["season"] = validate_season(str(config["season"]))
    if "output_dir" in config:
        config["output_dir"] = str(validate_output_dir(config["output_dir"]))
    if "stat" in config:
        config["stat"] = validate_stat_category(config["stat"])
    if "lineup" in config:
        config["lineup"] = validate_lineup(config["lineup"])

    return config
