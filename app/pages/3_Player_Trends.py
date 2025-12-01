import streamlit as st
import pandas as pd

st.title("📈 Weekly Player Trends")

try:
    df_trends = pd.read_csv("results/player_trends.csv")
    st.dataframe(df_trends, use_container_width=True)
except Exception as e:
    st.error(f"Error loading player trends: {e}")