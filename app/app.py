# ============================================================
# File: home.py
# Path: nba_dashboard/home.py
#
# Description:
#   Main landing page for the NBA Analytics Dashboard.
#   - Sets Streamlit page configuration.
#   - Provides a high-level introduction and sidebar navigation.
#   - Links to all main sections of the dashboard:
#       - Daily Predictions
#       - Weekly Summary
#       - Player Trends
#       - CLI Results
#       - Monte Carlo Bankroll Simulation
#       - Player-Level Monte Carlo
#
# Dependencies:
#   streamlit
#
# Author: Your Name
# Created: 2025-12-01
# Updated: 2025-12-01
# ============================================================

import streamlit as st

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="NBA Analytics Dashboard",
    layout="wide",
    page_icon="🏀"
)

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Navigation")
page_options = {
    "📅 Daily Predictions": "Daily predictions and game forecasts.",
    "📊 Weekly Summary": "Overview of weekly team performance and stats.",
    "📈 Player Trends": "Track player rolling trends and stats.",
    "🖥️ CLI Results": "View exported CLI results and simulations.",
    "💰 Monte Carlo Bankroll Simulation": "Simulate bankroll growth using Monte Carlo.",
    "👤 Player-Level Monte Carlo": "Run player-level Monte Carlo simulations."
}

selected_page = st.sidebar.radio(
    "Go to section:",
    list(page_options.keys()),
    index=0,
    help="Select a dashboard section to explore."
)

# Display description for selected section
st.sidebar.markdown(f"**Info:** {page_options[selected_page]}")

# ----------------------------
# Main page content
# ----------------------------
st.title("🏀 NBA Analytics Dashboard")
st.caption("Use the sidebar to navigate between sections.")

st.markdown("""
Welcome to the **NBA Analytics Dashboard**!  

This dashboard allows you to explore advanced NBA analytics, including:

- **Daily Predictions**: Pre-game win probabilities and odds analysis.
- **Weekly Summary**: Team performance, trends, and highlights.
- **Player Trends**: Rolling statistics and performance over time.
- **CLI Results**: Exported results from backend pipelines.
- **Monte Carlo Bankroll Simulation**: Simulate betting bankroll growth.
- **Player-Level Monte Carlo**: Simulate individual player outcomes.

Use the sidebar on the left to quickly switch between sections and explore insights.
""")
