import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

st.title("📊 Weekly Summary")
st.caption("Team performance trends aggregated by week.")

try:
    df_summary = pd.read_csv("results/weekly_summary.csv")

    if df_summary.empty:
        st.info("No weekly summary data available. Run the pipeline first.")
    else:
        # Show raw table
        st.subheader("Summary Table")
        st.dataframe(df_summary, use_container_width=True)

        # Metrics
        st.subheader("Key Metrics")
        num_teams = df_summary["team"].nunique()
        num_weeks = df_summary["week"].nunique()
        st.metric("Teams", num_teams)
        st.metric("Weeks Covered", num_weeks)

        # Average points per team chart
        if "PTS" in df_summary.columns:
            st.subheader("Average Points per Team by Week")
            chart_pts = alt.Chart(df_summary).mark_line(point=True).encode(
                x="week:O",
                y="PTS:Q",
                color="team:N",
                tooltip=["team","week","PTS"]
            ).interactive()
            st.altair_chart(chart_pts, use_container_width=True)

        # Win trends if available
        if "wins" in df_summary.columns:
            st.subheader("Weekly Wins per Team")
            chart_wins = alt.Chart(df_summary).mark_bar().encode(
                x="week:O",
                y="wins:Q",
                color="team:N",
                tooltip=["team","week","wins"]
            )
            st.altair_chart(chart_wins, use_container_width=True)

        # Export option
        export_name = f"weekly_summary_{datetime.now().date()}.csv"
        st.download_button(
            label="Download Weekly Summary as CSV",
            data=df_summary.to_csv(index=False),
            file_name=export_name,
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error loading weekly summary: {e}")