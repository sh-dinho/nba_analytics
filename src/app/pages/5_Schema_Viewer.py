import streamlit as st
import json
from pathlib import Path
from src.config.paths import MODEL_DIR
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.title("🧬 Schema Viewer")

model_type = st.selectbox(
    "Model type", ["moneyline", "totals", "spread_regression", "spread_classification"]
)

schema_files = list((MODEL_DIR / model_type).glob("*_schema.json"))
schema_files = sorted(schema_files)

schema_file = st.selectbox("Select schema file", schema_files)

if schema_file:
    schema = json.loads(schema_file.read_text())
    st.json(schema)
