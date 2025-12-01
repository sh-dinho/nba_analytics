import pandas as pd

from nba_core.db_module import connect


def team_stats(team: str):
    with connect() as con:
        df = pd.read_sql("SELECT * FROM nba_games WHERE home_team = ? OR away_team = ?", con, params=[team, team])
    df["points_scored"] = df.apply(lambda r: r["home_score"] if r["home_team"] == team else r["away_score"], axis=1)
    df["points_allowed"] = df.apply(lambda r: r["away_score"] if r["home_team"] == team else r["home_score"], axis=1)
    return df[["date", "season", "points_scored", "points_allowed", "winner"]]