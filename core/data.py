import sqlite3
import pandas as pd
import logging

from core.db_module import connect


def fetch_historical_games():
    with connect() as con:
        return pd.read_sql("SELECT * FROM nba_games", con)


def engineer_features(df):
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_points"] = df["home_score"].fillna(0) + df["away_score"].fillna(0)
    return df