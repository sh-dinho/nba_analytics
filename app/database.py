import sqlite3
import pandas as pd
import logging

DB_PATH = 'nba_analytics.db'
logging.basicConfig(level=logging.INFO)

def store_games(df, replace=False):
    """Store NBA games data into SQLite database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            mode = 'replace' if replace else 'append'
            df.to_sql('nba_games', conn, if_exists=mode, index=False)
        logging.info(f"Stored {len(df)} NBA games into the database.")
    except Exception as e:
        logging.error(f"Error storing data: {e}")

def fetch_games():
    """Fetch NBA games from the database."""
    query = "SELECT * FROM nba_games"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        logging.error(f"Error fetching games: {e}")
        return pd.DataFrame()