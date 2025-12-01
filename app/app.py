import streamlit as st
from config import configure_logging

configure_logging()

st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")
st.title("🏀 NBA Analytics Dashboard")

st.sidebar.header("Navigation")
st.sidebar.write("Use the Pages menu on the left to navigate.")
st.sidebar.success("Pages: Results Dashboard, Backtesting")

st.write("Welcome! This app shows live predictions, EV distributions, and historical backtesting performance.")
st.write("Use the sidebar to open pages for detailed charts and tables.")