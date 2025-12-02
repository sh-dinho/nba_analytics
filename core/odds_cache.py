# core/odds_cache.py
import os
import pandas as pd
from core.paths import DATA_DIR

CACHE_FILE = os.path.join(DATA_DIR, "odds_cache.csv")

def load_odds():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame()

def save_odds(df):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
