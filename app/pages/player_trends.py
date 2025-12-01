import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Player Trends", layout="wide")
st.title("📈 Weekly Player Trends (Multi-Metric)")

try:
    df = pd.read_csv("results/player_trends.csv")

    st.subheader("Top Risers (Points)")
    risers = df[df["trend"] == "Rising"].sort_values("PTS_change", ascending=False).head(10)
    st.dataframe(risers[["PLAYER_NAME","TEAM_ABBREVIATION","PTS_curr","PTS_change"]], use_container_width=True)

    st.subheader("Top Fallers (Points)")
    fallers = df[df["trend"] == "Falling"].sort_values("PTS_change").head(10)
    st.dataframe(fallers[["PLAYER_NAME","TEAM_ABBREVIATION","PTS_curr","PTS_change"]], use_container_width=True)

    # Multi-metric visualization
    st.subheader("Metric Changes vs Last Week")
    metrics = ["PTS_change","REB_change","AST_change","TS_PCT_change"]
    for metric in metrics:
        fig, ax = plt.subplots()
        top = df.sort_values(metric, ascending=False).head(10)
        ax.bar(top["PLAYER_NAME"], top[metric], color="skyblue")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(metric.replace("_change"," Change"))
        ax.set_xticklabels(top["PLAYER_NAME"], rotation=45, ha="right")
        st.pyplot(fig)

except FileNotFoundError:
    st.error("Trend data not found. Run weekly update to generate snapshots.")