import pandas as pd
import sqlite3
import joblib
import logging
from utils.features import generate_features
from utils.betting import calculate_ev, kelly_stake
from utils.notify import send_daily_picks

DB_PATH = "./data/nba_games.db"
MODEL_PATH = "./data/models/xgb_model.pkl"
BANKROLL = 1000

logging.basicConfig(level=logging.INFO)
model = joblib.load(MODEL_PATH)

with sqlite3.connect(DB_PATH) as con:
    df_games = pd.read_sql("SELECT * FROM nba_games ORDER BY GAME_DATE DESC LIMIT 50", con)

if df_games.empty:
    logging.error("No games found")
else:
    X = generate_features(df_games)
    df_games["Probability"] = model.predict_proba(X)[:, 1]
    df_games["EV"] = df_games["Probability"].apply(lambda p: calculate_ev(p, odds=2.0))
    df_games["SuggestedStake"] = df_games["EV"].apply(lambda ev: kelly_stake(ev, BANKROLL))
    send_daily_picks(df_games)
