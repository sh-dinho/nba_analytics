import streamlit as st
import pandas as pd
import sqlite3
import yaml
import requests
from datetime import datetime

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
API_URL = CONFIG["server"]["api_url"]

def fetch(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        return r.json()
    except:
        return {}

st.set_page_config(page_title="NBA Analytics", layout="wide")
st.title("🏀 NBA Analytics Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Picks","Model metrics","Bankroll","System status"])

with tab1:
    st.header("Daily Picks")
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 50", con)
    con.close()
    st.dataframe(df if not df.empty else "No picks yet.")

with tab2:
    st.header("Model Metrics")
    con = sqlite3.connect(DB_PATH)
    metrics = pd.read_sql("SELECT * FROM model_metrics ORDER BY Timestamp DESC LIMIT 20", con)
    con.close()
    st.dataframe(metrics if not metrics.empty else "No metrics yet.")

with tab3:
    st.header("Bankroll Tracker")
    con = sqlite3.connect(DB_PATH)
    tracker = pd.read_sql("SELECT * FROM bankroll_tracker ORDER BY Timestamp DESC LIMIT 50", con)
    con.close()
    st.dataframe(tracker if not tracker.empty else "No bankroll records yet.")

with tab4:
    st.header("System Status")
    health = fetch("/health")
    st.write(health)
    if st.button("Retrain Model"):
        resp = requests.post(f"{API_URL}/train/run")
        st.write(resp.json())
