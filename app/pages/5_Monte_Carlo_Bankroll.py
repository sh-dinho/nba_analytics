import streamlit as st
import pandas as pd
from scripts.simulate_bankroll import simulate_bankroll

st.title("💰 Monte Carlo Bankroll Simulation")

try:
    df_picks = pd.read_csv("results/picks.csv")
    sims = st.sidebar.slider("Number of simulations", 100, 2000, 500, 100)
    # Run simulation logic here
    ...
except Exception as e:
    st.error(f"Error running Monte Carlo simulation: {e}")