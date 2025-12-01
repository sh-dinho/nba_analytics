import streamlit as st

st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")

st.title("🏀 NBA Analytics Dashboard")
st.caption("Navigate using the sidebar to explore predictions, summaries, and simulations.")

st.markdown("""
Welcome to the NBA Analytics Dashboard.  
Use the sidebar to switch between:
- 📅 Daily Predictions
- 📊 Weekly Summary
- 📈 Player Trends
- 🖥️ CLI Results
- 💰 Monte Carlo Bankroll Simulation
- 👤 Player-Level Monte Carlo
""")