import sqlite3
from nba_api import fetch_nba_games
import logging

DB_PATH = "db/nba_analytics.db"

def store_games(df):
    if df.empty:
        logging.error("No games to store.")
        return
    con = sqlite3.connect(DB_PATH)
    for _, row in df.iterrows():
        con.execute("""
        INSERT OR REPLACE INTO nba_games (GameID, Date, Visitor, Visitor_PTS, Home, Home_PTS)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["id"],
            row["date"],
            row["visitor_team"]["full_name"],
            row["visitor_team_score"],
            row["home_team"]["full_name"],
            row["home_team_score"]
        ))
    con.commit()
    con.close()
    logging.info("✅ Games stored successfully.")

if __name__ == "__main__":
    df = fetch_nba_games()
    store_games(df)
