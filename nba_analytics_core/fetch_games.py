import pandas as pd
import logging
from datetime import date

from nba_analytics_core.db_module import connect


def get_todays_games():
    today = date.today().strftime("%b %d, %Y")
    with connect() as con:
        df = pd.read_sql("SELECT * FROM nba_games WHERE date = ?", con, params=[today])
    return df