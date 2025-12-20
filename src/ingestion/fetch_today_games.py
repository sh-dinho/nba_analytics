"""
Fetch today's NBA games using LeagueGameFinder (stable endpoint).
Parses MATCHUP to determine home/away teams.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder

OUTPUT_PATH = Path("data/raw/today_games.csv")


def parse_matchup(row):
    """
    Convert MATCHUP like:
        'LAL vs BOS' → home=LAL, away=BOS
        'GSW @ PHX'  → home=PHX, away=GSW
    """
    matchup = row["MATCHUP"]
    team = row["TEAM_NAME"]

    if " vs " in matchup:
        # TEAM_NAME is home
        home = team
        away = matchup.split(" vs ")[1]
    elif " @ " in matchup:
        # TEAM_NAME is away
        away = team
        home = matchup.split(" @ ")[1]
    else:
        home = None
        away = None

    return pd.Series({"home_team": home, "away_team": away})


def fetch_today_games():
    today = datetime.today().strftime("%Y-%m-%d")

    finder = LeagueGameFinder()
    df = finder.get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df_today = df[df["GAME_DATE"] == pd.to_datetime(today)]

    if df_today.empty:
        print("No NBA games today.")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_today.to_csv(OUTPUT_PATH, index=False)
        return OUTPUT_PATH

    # Parse home/away teams from MATCHUP
    teams = df_today.apply(parse_matchup, axis=1)

    df_out = pd.DataFrame(
        {
            "game_id": df_today["GAME_ID"],
            "date": today,
            "home_team": teams["home_team"],
            "away_team": teams["away_team"],
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved today's games → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    fetch_today_games()
