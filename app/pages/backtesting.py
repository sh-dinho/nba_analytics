import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Backtesting", layout="wide")
st.title("📈 Backtesting Performance")

csv_file = st.sidebar.text_input("Backtesting CSV path", "results/backtest.csv")

try:
    df = pd.read_csv(csv_file)

    st.subheader("Raw Backtesting Results")
    st.dataframe(df, use_container_width=True)

    if {"season", "roi"} <= set(df.columns):
        st.subheader("ROI Trend by Season")
        fig, ax = plt.subplots()
        ax.plot(df["season"], df["roi"] * 100, marker="o", color="blue")
        ax.set_xlabel("Season")
        ax.set_ylabel("ROI (%)")
        st.pyplot(fig)

    if {"season", "win_rate"} <= set(df.columns):
        st.subheader("Win Rate Trend by Season")
        fig, ax = plt.subplots()
        ax.bar(df["season"], df["win_rate"] * 100, color="orange")
        ax.set_xlabel("Season")
        ax.set_ylabel("Win Rate (%)")
        st.pyplot(fig)

    if "final_bankroll" in df.columns:
        st.subheader("Final Bankroll by Season")
        fig, ax = plt.subplots()
        ax.bar(df["season"], df["final_bankroll"], color="green")
        ax.set_xlabel("Season")
        ax.set_ylabel("Final Bankroll ($)")
        st.pyplot(fig)

except FileNotFoundError:
    st.error(f"CSV file not found at {csv_file}. Run backtesting script to generate results.")