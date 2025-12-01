import streamlit as st
import pandas as pd

st.title("📊 Weekly Summary")

try:
    df_summary = pd.read_csv("results/weekly_summary.csv")
    st.dataframe(df_summary, use_container_width=True)
except Exception as e:
    st.error(f"Error loading weekly summary: {e}")