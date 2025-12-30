from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Page: Pipeline Health Dashboard
# File: src/app/pages/15_Pipeline_Health.py
#
# Description:
#     Shows ingestion + prediction freshness,
#     model registry status, missing games,
#     and includes a "Run Pipeline Now" button.
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

from src.config.paths import DATA_DIR, MODEL_REGISTRY_PATH
from src.model.registry import list_models


from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()


# ------------------------------------------------------------
# Helper: color-coded freshness
# ------------------------------------------------------------
def freshness_badge(timestamp: str) -> str:
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S UTC")
        age = datetime.utcnow() - dt

        if age < timedelta(hours=6):
            color = "#2ecc71"  # green
        elif age < timedelta(hours=24):
            color = "#f1c40f"  # yellow
        else:
            color = "#e74c3c"  # red

        return f"<span style='background:{color};padding:4px 8px;border-radius:6px;color:black;'>{timestamp}</span>"
    except:
        return f"<span style='background:#e74c3c;padding:4px 8px;border-radius:6px;color:black;'>Invalid</span>"


# ------------------------------------------------------------
# Section 1: Pipeline timestamps
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Section 2: Trigger pipeline
# ------------------------------------------------------------
st.markdown("---")
st.markdown("## 🚀 Run Pipeline Now")

if st.button("Run Full Pipeline"):
    with st.spinner("Running pipeline..."):
        result = trigger_full_pipeline()
    st.success(result)


# ------------------------------------------------------------
# Section 3: Model Registry Health
# ------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧠 Model Registry Health")

models = list_models()

if not models:
    st.warning("No models found in registry.")
else:
    df = pd.DataFrame(models)
    st.dataframe(df, use_container_width=True)


# ------------------------------------------------------------
# Section 4: Odds Health
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Section 5: Missing Games Check
# ------------------------------------------------------------
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
