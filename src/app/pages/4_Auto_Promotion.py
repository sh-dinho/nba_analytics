import streamlit as st
from src.model.auto_promotion import auto_promote_all
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.title("🚀 Auto-Promotion Panel")

if st.button("Run Auto-Promotion"):
    results = auto_promote_all()
    st.json(results)
