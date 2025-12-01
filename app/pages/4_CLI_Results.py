import streamlit as st
import pandas as pd

st.title("🖥️ CLI Results")

try:
    df_cli = pd.read_csv("results/picks.csv")
    st.dataframe(df_cli, use_container_width=True)
except Exception as e:
    st.error(f"Error loading CLI results: {e}")