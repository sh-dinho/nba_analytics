import pandas as pd
import logging

from nba_analytics_core.db_module import connect, init_db


def import_games_from_csv(path: str):
    df = pd.read_csv(path)
    init_db()
    with connect() as con:
        cur = con.cursor()
        for _, row in df.iterrows():
            cur.execute("""
                INSERT OR REPLACE INTO nba_games
                (game_id, date, season, home_team, away_team, home_score, away_score, winner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["game_id"],
                row["date"],
                int(row["season"]),
                row["home_team"],
                row["away_team"],
                int(row["home_score"]) if not pd.isna(row["home_score"]) else None,
                int(row["away_score"]) if not pd.isna(row["away_score"]) else None,
                row["winner"]
            ))
        con.commit()
    logging.info(f"✔ Imported {len(df)} games from {path}")