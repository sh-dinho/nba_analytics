# app/dashboard.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")

st.title("🏀 NBA Analytics Dashboard")

# ----------------------------
# Load picks
# ----------------------------
RESULTS_DIR = "results"
picks_file = st.sidebar.text_input("Picks CSV file", os.path.join(RESULTS_DIR, "picks.csv"))

if not os.path.exists(picks_file):
    st.warning(f"Picks file not found: {picks_file}")
else:
    df = pd.read_csv(picks_file)
    st.subheader("Picks Table")
    st.dataframe(df)

    # ----------------------------
    # Summary statistics
    # ----------------------------
    st.subheader("EV Summary")
    if "ev" in df.columns:
        st.bar_chart(df["ev"])

    st.subheader("Bankroll Trajectory")
    if "bankroll" in df.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df.index, df["bankroll"], marker="o", linestyle="-")
        ax.set_xlabel("Bet #")
        ax.set_ylabel("Bankroll ($)")
        ax.set_title("Bankroll over Bets")
        ax.grid(True)
        st.pyplot(fig)

    # Filter by threshold
    st.subheader("Strong Picks Filter")
    threshold = st.slider("EV threshold", float(df["ev"].min()), float(df["ev"].max()), 0.5)
    strong_picks = df[df["ev"] >= threshold]
    st.write(strong_picks)
