import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Player Trends", layout="wide")
st.title("📈 Weekly Player Trends")

try:
    df = pd.read_csv("results/player_trends.csv")

    st.subheader("Top Risers")
    risers = df[df["trend"] == "Rising"].sort_values("PTS_change", ascending=False).head(10)
    st.dataframe(risers[["PLAYER_NAME","TEAM_ABBREVIATION","PTS_curr","PTS_change"]], use_container_width=True)

    st.subheader("Top Fallers")
    fallers = df[df["trend"] == "Falling"].sort_values("PTS_change").head(10)
    st.dataframe(fallers[["PLAYER_NAME","TEAM_ABBREVIATION","PTS_curr","PTS_change"]], use_container_width=True)

    st.subheader("Trend Visualization")
    fig, ax = plt.subplots()
    ax.bar(df["PLAYER_NAME"], df["PTS_change"], color=df["trend"].map({"Rising":"green","Falling":"red","Stable":"gray"}))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Points per Game Change vs Last Week")
    ax.set_xticklabels(df["PLAYER_NAME"], rotation=45, ha="right")
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Trend data not found. Run weekly update to generate snapshots.")