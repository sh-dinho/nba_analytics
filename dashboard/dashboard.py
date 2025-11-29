import streamlit as st
import pandas as pd
import requests
import sqlite3
import yaml

CONFIG = yaml.safe_load(open("config.yaml"))
API_URL = CONFIG["server"]["api_url"]
DB_PATH = CONFIG["database"]["path"]

def fetch(path: str, method: str = "GET"):
    url = f"{API_URL}{path}"
    if method == "GET":
        r = requests.get(url, timeout=10)
    else:
        r = requests.post(url, timeout=30)
    return r.json()

st.set_page_config(page_title="NBA Analytics", layout="wide")
st.title("🏀 NBA Analytics Platform")

tab1, tab2, tab3, tab4 = st.tabs(["Picks", "Model metrics", "Bankroll", "System status"])

# ------------------- Daily Picks -------------------
with tab1:
    st.header("Daily picks")
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 50", con)
    con.close()
    if df.empty:
        st.info("No picks yet.")
    else:
        st.dataframe(df)

# ------------------- Model Metrics -------------------
with tab2:
    st.header("Model metrics")
    con = sqlite3.connect(DB_PATH)
    metrics = pd.read_sql("SELECT * FROM model_metrics ORDER BY Timestamp DESC LIMIT 20", con)
    con.close()
    if metrics.empty:
        st.info("No metrics yet.")
    else:
        st.dataframe(metrics)
        st.metric("Latest AUC", f"{metrics.iloc[0]['AUC']:.3f}")
        st.metric("Latest Accuracy", f"{metrics.iloc[0]['Accuracy']:.3f}")

# ------------------- Bankroll -------------------
with tab3:
    st.header("Bankroll tracker")
    con = sqlite3.connect(DB_PATH)
    tracker = pd.read_sql("SELECT * FROM bankroll_tracker ORDER BY Timestamp DESC LIMIT 50", con)
    con.close()
    if tracker.empty:
        st.info("No bankroll records yet.")
    else:
        st.dataframe(tracker)
        st.metric("Current Bankroll", f"${tracker.iloc[0]['CurrentBankroll']:.2f}")
        st.metric("ROI", f"{tracker.iloc[0]['ROI']:.2%}")

# ------------------- System Status -------------------
with tab4:
    st.header("System status")
    try:
        health = fetch("/health")
        st.metric("API Health", health.get("status", "unknown"))
        st.write(f"Last checked: {health.get('time')}")
    except Exception:
        st.error("API not reachable.")

    if st.button("🔄 Retrain model now"):
        try:
            result = fetch("/train/run", method="POST")
            if result.get("status") == "ok":
                st.success("✔ Model retraining started successfully.")
            else:
                st.error(f"❌ Retrain failed: {result.get('message')}")
        except Exception as e:
            st.error(f"❌ Error triggering retrain: {e}")

    try:
        retrain = fetch("/metrics/retrain")
        if "history" in retrain and retrain["history"]:
            df = pd.DataFrame(retrain["history"])
            st.subheader("📅 Retrain history")
            st.dataframe(df.head(10))
            last_retrain = df.iloc[0]["Timestamp"]
            last_dt = pd.to_datetime(last_retrain)
            days_since = (pd.Timestamp.now() - last_dt).days
            overdue_days = days_since - CONFIG["model"]["retrain_days"]
            if overdue_days > 0:
                st.error(f"⚠ Retraining overdue by {overdue_days} days")
            else:
                st.success(f"Model retrained {days_since} days ago")
        else:
            st.info("No retrain history found.")
    except Exception:
        st.error("Failed to fetch retrain history.")
