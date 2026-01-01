from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Pipeline Health Dashboard
# File: src/app/pages/15_Pipeline_Health.py
# Purpose: Display ingestion and prediction freshness,
#          model registry status, and trigger pipeline runs.
# ============================================================

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

from src.app.utils.pipeline_status import (
    get_pipeline_last_run,
    get_ingestion_last_run,
)
from src.app.utils.pipeline_trigger import trigger_full_pipeline
from src.config.paths import DATA_DIR
from src.model.registry import list_models

st.set_page_config(page_title="Pipeline Health", page_icon="🩺", layout="wide")

render_header()
set_active_page("Pipeline Health")
render_navbar()

st.title("🩺 Pipeline Health Dashboard")


def freshness_badge(timestamp: str) -> str:
    try:
        dt = datetime.strptime(timestamp.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        age = datetime.utcnow() - dt

        if age < timedelta(hours=6):
            color = "#2ecc71"
        elif age < timedelta(hours=24):
            color = "#f1c40f"
        else:
            color = "#e74c3c"

        return f"<span style='background:{color};padding:4px 8px;border-radius:6px;color:black;'>{timestamp}</span>"
    except Exception:
        return "<span style='background:#e74c3c;padding:4px 8px;border-radius:6px;color:black;'>Invalid</span>"


st.markdown("## 🔄 Pipeline Status")

pipeline_ts = get_pipeline_last_run()
ingestion_ts = get_ingestion_last_run()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Last Pipeline Run:**")
    st.markdown(freshness_badge(pipeline_ts), unsafe_allow_html=True)
with col2:
    st.markdown("**Last Ingestion Run:**")
    st.markdown(freshness_badge(ingestion_ts), unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 🚀 Run Pipeline Now")

backfill_days = st.number_input(
    "Backfill days (0 = today only)", value=0, min_value=0, max_value=30
)

if st.button("Run Full Pipeline"):
    with st.spinner("Running pipeline..."):
        result = trigger_full_pipeline(backfill_days=int(backfill_days))
    st.success(result)

st.markdown("---")
st.markdown("## 🧠 Model Registry Health")

model_types = ["moneyline", "totals", "spread_regression", "spread_classification"]
rows = []
for mt in model_types:
    for m in list_models(mt):
        rows.append({"model_type": mt, **m})

if not rows:
    st.warning("No models found in registry.")
else:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("## 💵 Odds Health")

odds_files = list(DATA_DIR.glob("odds/odds_*.csv")) + list(
    DATA_DIR.glob("odds/odds_*.parquet")
)

if not odds_files:
    st.warning("No odds files found.")
else:
    latest = max(odds_files, key=lambda p: p.stat().st_mtime)
    ts = datetime.utcfromtimestamp(latest.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    st.markdown(f"**Latest odds file:** `{latest.name}`")
    st.markdown(f"**Last updated:** {freshness_badge(ts)}", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 🏀 Missing Games Check")

games_path = DATA_DIR / "games.parquet"
if not games_path.exists():
    st.warning("games.parquet not found.")
else:
    games = pd.read_parquet(games_path)
    missing = games[games["home_team"].isna() | games["away_team"].isna()]
    if missing.empty:
        st.success("No missing games.")
    else:
        st.error("Missing games detected:")
        st.dataframe(missing, use_container_width=True)
