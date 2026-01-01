from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Model Registry
# File: src/app/pages/05_Model_Registry.py
# Purpose: Inspect registered models, versions, metrics, and
#          promotion status.
# ============================================================

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.model.registry import list_models

st.set_page_config(page_title="Model Registry", page_icon="🧠", layout="wide")

render_header()
set_active_page("Model Registry")
render_navbar()

st.title("🧠 Model Registry")

model_types = ["moneyline", "totals", "spread_regression", "spread_classification"]

rows = []
for mt in model_types:
    for m in list_models(mt):
        rows.append({"model_type": mt, **m})

if not rows:
    st.info("No models registered.")
else:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
