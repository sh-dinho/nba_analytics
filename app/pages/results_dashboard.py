import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Results Dashboard", layout="wide")
st.title("📊 Prediction Results Dashboard")

csv_file = st.sidebar.text_input("CSV file path", "results/picks.csv")

try:
    df = pd.read_csv(csv_file)

    st.subheader("Raw Predictions")
    st.dataframe(df, use_container_width=True)

    if "bankroll" in df.columns:
        st.subheader("Bankroll Growth Over Bets")
        fig, ax = plt.subplots()
        ax.plot(df["bankroll"], marker="o", color="green")
        ax.set_xlabel("Bet #")
        ax.set_ylabel("Bankroll ($)")
        st.pyplot(fig)

    if "ev" in df.columns:
        st.subheader("Expected Value Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["ev"], bins=10, color="skyblue", edgecolor="black")
        ax.set_xlabel("EV")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    if "pred_home_win_prob" in df.columns:
        st.subheader("Win Probability by Game")
        fig, ax = plt.subplots()
        ax.bar(df["home_team"], df["pred_home_win_prob"], color="orange")
        ax.set_ylabel("Win Probability")
        ax.set_xticklabels(df["home_team"], rotation=45, ha="right")
        st.pyplot(fig)

except FileNotFoundError:
    st.error(f"CSV file not found at {csv_file}. Run CLI with --export first.")