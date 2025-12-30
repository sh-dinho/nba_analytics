import streamlit as st
from src.model.registry import list_models, promote_model, delete_model
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.title("📚 Model Registry Browser")

model_type = st.selectbox(
    "Select model type",
    ["moneyline", "totals", "spread_regression", "spread_classification"],
)

models = list_models(model_type)

for m in models:
    with st.expander(f"Model {m['version']} {'(PROD)' if m['is_production'] else ''}"):
        st.json(m)

        col1, col2 = st.columns(2)
        if col1.button(f"Promote {m['version']}", key=f"promote_{m['version']}"):
            promote_model(model_type, m["version"])
            st.success("Promoted!")

        if col2.button(f"Delete {m['version']}", key=f"delete_{m['version']}"):
            delete_model(model_type, m["version"])
            st.warning("Deleted!")
