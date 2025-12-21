"""
Fetch today's NBA games and return a clean DataFrame.
"""

from datetime import datetime
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder


def parse_matchup(matchup: str, team: str):
    m = matchup.replace(".", "").replace("(OT)", "").strip()

    if " vs " in m:
        home = team
        away = m.split(" vs ")[1].strip()
    elif " @ " in m:
        away = team
        home = m.split(" @ ")[1].strip()
    else:
        home = None
        away = None

    return home, away


def fetch_today_games_df() -> pd.DataFrame:
    today = datetime.today().strftime("%Y-%m-%d")
    today_ts = pd.to_datetime(today)

    finder = LeagueGameFinder()
    df = finder.get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df_today = df[df["GAME_DATE"] == today_ts]

    if df_today.empty:
        return pd.DataFrame(columns=["game_id", "date", "home_team", "away_team"])

    parsed = df_today.apply(
        lambda r: parse_matchup(r["MATCHUP"], r["TEAM_NAME"]), axis=1
    )
    df_today["home_team"] = parsed.apply(lambda x: x[0])
    df_today["away_team"] = parsed.apply(lambda x: x[1])

    df_games = (
        df_today[["GAME_ID", "GAME_DATE", "home_team", "away_team"]]
        .drop_duplicates(subset=["GAME_ID"])
        .rename(
            columns={
                "GAME_ID": "game_id",
                "GAME_DATE": "date",
            }
        )
    )

    return df_games
