import streamlit as st
import pandas as pd
from src.model.registry import list_models

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.title("🏆 Model Leaderboard")

model_types = ["moneyline", "totals", "spread_regression", "spread_classification"]

rows = []
for mt in model_types:
    models = list_models(mt)
    for m in models:
        row = {
            "model_type": mt,
            "version": m["version"],
            "is_production": m["is_production"],
        }
        row.update(m["metrics"])
        rows.append(row)

df = pd.DataFrame(rows)
st.dataframe(df.sort_values(["model_type", "is_production"], ascending=[True, False]))
