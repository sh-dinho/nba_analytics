import os
import sqlite3
import pandas as pd
import joblib
import logging
import yaml
from utils.notify import send_daily_picks
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = os.path.abspath(CONFIG["database"]["path"])
MODEL_PATH = os.path.join(CONFIG["model"]["models_dir"], CONFIG["model"]["filename"])
model = joblib.load(MODEL_PATH)

with sqlite3.connect(DB_PATH) as con:
    df_games = pd.read_sql("SELECT * FROM nba_games", con)

# Demo: simple EV calculation
df_games["Probability"] = model.predict_proba(df_games[["PTS","REB","AST"]])[:,1]
df_games["Odds"] = 2.0
df_games["EV"] = df_games["Probability"]*df_games["Odds"]-1
df_picks = df_games[df_games["EV"]>=CONFIG["betting"]["min_ev_threshold"]]

# Store picks
with sqlite3.connect(DB_PATH) as con:
    df_picks.to_sql("daily_picks", con, if_exists="replace", index=False)

send_daily_picks(df_picks)
