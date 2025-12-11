# ============================================================
# File: src/api/nba_api_client.py
# Purpose: Fetch NBA data (season schedule, boxscores, today’s games, next game day)
# ============================================================

import logging
import datetime
import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    teamdetails,
    scoreboardv2,
)

logger = logging.getLogger("api.nba_api_client")


# --- Helper: TEAM_ID → Abbreviation ---
def get_team_abbreviation(team_id: int) -> str:
    """Map TEAM_ID to team abbreviation (e.g., BOS, LAL)."""
    try:
        details = teamdetails.TeamDetails(team_id=team_id)
        df = details.get_data_frames()[0]
        return df.loc[0, "ABBREVIATION"]
    except Exception as e:
        logger.warning("Failed to fetch team abbreviation for %s: %s", team_id, e)
        return str(team_id)


# --- Season Games ---
def fetch_season_games(season: int) -> pd.DataFrame:
    """Fetch all games for a season with WL outcomes and team abbreviations."""
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = gamefinder.get_data_frames()[0]

    if "TEAM_ID" in df.columns:
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].apply(get_team_abbreviation)

    return df


# --- Boxscores ---
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
        return pd.DataFrame()

    boxscores = pd.concat(all_boxscores, ignore_index=True)
    boxscores["PTS_OPP"] = boxscores.groupby("GAME_ID")["PTS"].transform(
        lambda x: x[::-1].values
    )
    boxscores["TEAM_ABBREVIATION"] = boxscores["TEAM_ID"].apply(get_team_abbreviation)
    return boxscores


# --- Today’s Games with fallback ---
def fetch_today_games(date: str | None = None) -> pd.DataFrame:
    """
    Fetch today's NBA games with team abbreviations.
    If no games today, check NBA Cup schedule or next game day.
    Adds a column GAME_TYPE = 'Regular Season' or 'NBA Cup'.
    """
    if date is None:
        date = datetime.date.today().strftime("%Y-%m-%d")

    try:
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date)
        games = scoreboard.get_data_frames()[0]

        required_cols = ["GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
        games = games[[c for c in required_cols if c in games.columns]]

        if not games.empty:
            team_map = {}
            for tid in pd.concat(
                [games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]]
            ).unique():
                team_map[tid] = get_team_abbreviation(int(tid))
            games["HOME_TEAM_ABBREVIATION"] = games["HOME_TEAM_ID"].map(team_map)
            games["AWAY_TEAM_ABBREVIATION"] = games["VISITOR_TEAM_ID"].map(team_map)

            # Label as Regular Season by default
            games["GAME_TYPE"] = "Regular Season"
            return games[
                [
                    "GAME_ID",
                    "GAME_DATE_EST",
                    "HOME_TEAM_ABBREVIATION",
                    "AWAY_TEAM_ABBREVIATION",
                    "GAME_TYPE",
                ]
            ]

        # --- Fallback: look ahead to next game day ---
        logger.info("No games today. Searching for next scheduled game day...")
        next_date = datetime.date.today()
        for _ in range(7):  # look ahead up to a week
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
                    team_map = {}
                    for tid in pd.concat(
                        [games_next["HOME_TEAM_ID"], games_next["VISITOR_TEAM_ID"]]
                    ).unique():
                        team_map[tid] = get_team_abbreviation(int(tid))
                    games_next["HOME_TEAM_ABBREVIATION"] = games_next[
                        "HOME_TEAM_ID"
                    ].map(team_map)
                    games_next["AWAY_TEAM_ABBREVIATION"] = games_next[
                        "VISITOR_TEAM_ID"
                    ].map(team_map)

                    # Detect NBA Cup by date range (Dec 9–16, 2025 for this season)
                    cup_start = datetime.date(2025, 12, 9)
                    cup_end = datetime.date(2025, 12, 16)
                    if cup_start <= next_date <= cup_end:
                        games_next["GAME_TYPE"] = "NBA Cup"
                    else:
                        games_next["GAME_TYPE"] = "Regular Season"

                    logger.info(
                        "Next NBA game day is %s (%s)",
                        next_date.strftime("%Y-%m-%d"),
                        games_next["GAME_TYPE"].iloc[0],
                    )
                    return games_next[
                        [
                            "GAME_ID",
                            "GAME_DATE_EST",
                            "HOME_TEAM_ABBREVIATION",
                            "AWAY_TEAM_ABBREVIATION",
                            "GAME_TYPE",
                        ]
                    ]
            except Exception:
                continue

        logger.warning("No NBA games found in the next 7 days.")
        return pd.DataFrame()

    except Exception as e:
        logger.error("Failed to fetch today’s games: %s", e)
        return pd.DataFrame()
