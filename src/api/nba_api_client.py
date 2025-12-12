# ============================================================
# File: src/api/nba_api_client.py
# Purpose: Fetch NBA data (season schedule, boxscores, today’s games, next game day)
# ============================================================

import logging
import datetime
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    teamdetails,
    scoreboardv2,
)
from src.schemas import TODAY_SCHEDULE_COLUMNS, normalize_today_schedule

logger = logging.getLogger("api.nba_api_client")


def get_team_abbreviation(team_id: int) -> str:
    """Map TEAM_ID to team abbreviation (e.g., BOS, LAL)."""
    try:
        details = teamdetails.TeamDetails(team_id=team_id)
        df = details.get_data_frames()[0]
        return df.loc[0, "ABBREVIATION"]
    except Exception as e:
        logger.warning("Failed to fetch team abbreviation for %s: %s", team_id, e)
        return str(team_id)


def fetch_season_games(season: int, retries: int = 3, delay: int = 10) -> pd.DataFrame:
    """Fetch all games for a season with WL outcomes and team abbreviations."""
    for attempt in range(retries):
        try:
            logger.info("Fetching season %s games (attempt %d)...", season, attempt + 1)
            gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
            df = gamefinder.get_data_frames()[0]

            if "TEAM_ID" in df.columns:
                df["TEAM_ABBREVIATION"] = df["TEAM_ID"].apply(get_team_abbreviation)

            return df
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as e:
            logger.warning(
                "Network error fetching season %s: %s. Retrying in %s seconds...",
                season,
                e,
                delay,
            )
            time.sleep(delay)
        except Exception as e:
            logger.error("Unexpected error fetching season %s: %s", season, e)
            return pd.DataFrame()
    logger.error("Failed to fetch season %s games after %d retries", season, retries)
    return pd.DataFrame()


def fetch_boxscores(game_ids: list[str]) -> pd.DataFrame:
    """Fetch boxscores for a list of game IDs, including points scored and allowed."""
    all_boxscores = []
    for gid in game_ids:
        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
            df_box = box.get_data_frames()[1]  # team stats table
            df_box = df_box[["GAME_ID", "TEAM_ID", "PTS"]]
            all_boxscores.append(df_box)
        except Exception as e:
            logger.warning("Failed to fetch boxscore for %s: %s", gid, e)

    if not all_boxscores:
        logger.warning("No boxscores fetched for provided game IDs.")
        return pd.DataFrame()

    boxscores = pd.concat(all_boxscores, ignore_index=True)

    # Opponent points: reverse values within each GAME_ID group
    boxscores["PTS_OPP"] = boxscores.groupby("GAME_ID")["PTS"].transform(
        lambda x: x[::-1].values
    )

    # Add team abbreviation
    boxscores["TEAM_ABBREVIATION"] = boxscores["TEAM_ID"].apply(get_team_abbreviation)
    return boxscores


def fetch_today_games(date: str | None = None) -> pd.DataFrame:
    """
    Fetch today's NBA games. If no games today, look ahead up to 7 days and return the next game day.
    Returns normalized DataFrame with TODAY_SCHEDULE_COLUMNS, using team abbreviations.
    """
    if date is None:
        date = datetime.date.today().strftime("%Y-%m-%d")

    required_cols = ["GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]

    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date)
        games = sb.get_data_frames()[0]
        games = games[[c for c in required_cols if c in games.columns]]

        def decorate(df: pd.DataFrame, current_date: datetime.date) -> pd.DataFrame:
            if df.empty:
                return df
            team_map = {}
            for tid in pd.concat([df["HOME_TEAM_ID"], df["VISITOR_TEAM_ID"]]).unique():
                team_map[tid] = get_team_abbreviation(int(tid))
            df["HOME_TEAM_ABBREVIATION"] = df["HOME_TEAM_ID"].map(team_map)
            df["AWAY_TEAM_ABBREVIATION"] = df["VISITOR_TEAM_ID"].map(team_map)
            # NBA Cup window (configurable per year)
            cup_windows = {
                2025: (datetime.date(2025, 12, 9), datetime.date(2025, 12, 16)),
            }
            cup_start, cup_end = cup_windows.get(current_date.year, (None, None))
            if cup_start and cup_end and cup_start <= current_date <= cup_end:
                df["GAME_TYPE"] = "NBA Cup"
            else:
                df["GAME_TYPE"] = "Regular Season"
            return df[
                [
                    "GAME_ID",
                    "GAME_DATE_EST",
                    "HOME_TEAM_ABBREVIATION",
                    "AWAY_TEAM_ABBREVIATION",
                    "GAME_TYPE",
                ]
            ]

        if not games.empty:
            out = decorate(games, datetime.date.fromisoformat(date))
            return normalize_today_schedule(out)

        # Fallback: look ahead up to a week
        logger.info("No games today. Searching for next scheduled game day...")
        next_date = datetime.date.fromisoformat(date)
        for _ in range(7):
            next_date += datetime.timedelta(days=1)
            try:
                sb_next = scoreboardv2.ScoreboardV2(
                    game_date=next_date.strftime("%Y-%m-%d")
                )
                games_next = sb_next.get_data_frames()[0]
                games_next = games_next[
                    [c for c in required_cols if c in games_next.columns]
                ]
                if not games_next.empty:
                    out = decorate(games_next, next_date)
                    logger.info(
                        "Next NBA game day is %s (%s)",
                        next_date.strftime("%Y-%m-%d"),
                        out["GAME_TYPE"].iloc[0],
                    )
                    return normalize_today_schedule(out)
            except Exception as e:
                logger.warning("Error checking next game day %s: %s", next_date, e)
                continue

        logger.warning("No NBA games found in the next 7 days.")
        return pd.DataFrame()

    except Exception as e:
        logger.error("Failed to fetch today’s games: %s", e)
        return pd.DataFrame()
