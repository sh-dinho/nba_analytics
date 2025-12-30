import streamlit as st
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.title("📈 Backtesting Dashboard")
st.info("Backtesting engine coming soon.")
