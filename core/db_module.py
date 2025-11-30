import sqlite3


def connect():
    return sqlite3.connect("bets.db")


def init_db():
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nba_games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                season INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                winner TEXT
            )
        """)
        con.commit()