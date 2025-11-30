import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_bet_history(df_bets: pd.DataFrame):
    """
    Plot bankroll history and ROI progression from the bet history DataFrame.
    Expects columns: Timestamp, CurrentBankroll, ROI
    """
    if df_bets is None or df_bets.empty:
        st.warning("No bet history available.")
        return

    # Ensure Timestamp is datetime
    df_bets["Timestamp"] = pd.to_datetime(df_bets["Timestamp"])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Bankroll progression
    ax[0].plot(df_bets["Timestamp"], df_bets["CurrentBankroll"], marker="o", linestyle="-", color="blue")
    ax[0].set_title("Bankroll Progression")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Bankroll")
    ax[0].grid(True)

    # ROI progression
    ax[1].plot(df_bets["Timestamp"], df_bets["ROI"], marker="o", linestyle="-", color="green")
    ax[1].set_title("ROI Progression")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("ROI")
    ax[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Show table
    st.write("### Bet History (latest records)")
    st.dataframe(df_bets.tail(20))