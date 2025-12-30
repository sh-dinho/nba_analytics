# ============================================================
# 🏀 NBA Analytics v4
# Entry: Main Workstation
# Author: Sadiq
# Version: 4.0.0
# Purpose: Unified entry point for the analytics dashboard.
# ============================================================
import streamlit as st
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

# Initialize critical session state keys
if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

render_header()
set_active_page("Home")
render_navbar()

st.title("Welcome to NBA Analytics v4")
st.info(
    "The system is now running on a unified math engine with thread-safe data persistence."
)
