# ============================================================
# File: 4_CLI_Results.py
# Path: <project_root>/pages/4_CLI_Results.py
#
# Description:
#   Streamlit page displaying recommended picks generated
#   from the CLI pipeline. Includes table view, summary
#   metrics, EV distribution charts, bankroll simulation
#   preview, and CSV export.
# ============================================================

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from scripts.simulate_bankroll import simulate_bankroll

st.title("🖥️ CLI Results")
st.caption("Recommended picks generated from the CLI pipeline.")

try:
    df_cli = pd.read_csv("results/picks.csv")

    if df_cli.empty:
        st.info("No picks available. Run the CLI pipeline first.")
    else:
        # Show raw table
        st.subheader("Raw Picks Data")
        st.dataframe(df_cli, use_container_width=True)

        # Metrics
        st.subheader("Summary Metrics")
        st.metric("Total Picks", len(df_cli))
        if {"ev_home","ev_away"}.issubset(df_cli.columns):
            num_positive_ev = ((df_cli["ev_home"] > 0) | (df_cli["ev_away"] > 0)).sum()
            st.metric("Positive EV Bets", int(num_positive_ev))

        # EV distribution chart
        if {"ev_home","ev_away"}.issubset(df_cli.columns):
            ev_data = df_cli.melt(
                id_vars=["home_team","away_team"],
                value_vars=["ev_home","ev_away"],
                var_name="side",
                value_name="ev"
            )
            chart_ev = alt.Chart(ev_data).mark_bar().encode(
                x=alt.X("ev", bin=alt.Bin(maxbins=30), title="Expected Value"),
                y="count()",
                color="side",
                tooltip=["home_team","away_team","ev"]
            )
            st.subheader("EV Distribution")
            st.altair_chart(chart_ev, use_container_width=True)

        # Sidebar control for bankroll simulation strategy
        st.sidebar.header("Simulation Settings")
        strategy = st.sidebar.selectbox("Select Betting Strategy", ["kelly", "flat"])

        # Bankroll simulation preview
        if {"home_win_prob","decimal_odds_home"}.issubset(df_cli.columns):
            st.subheader("💰 Bankroll Simulation Preview")
            sim_df = df_cli.rename(columns={"home_win_prob":"prob","decimal_odds_home":"decimal_odds"})
            sim_input = sim_df[["decimal_odds","prob","ev_home"]].dropna()
            if not sim_input.empty:
                trajectories, metrics = simulate_bankroll(sim_input, strategy=strategy)
                trajectory = trajectories[0]  # take first run for preview

                st.write(f"Final Bankroll (mean): ${metrics['final_bankroll_mean']:.2f}")
                st.write(f"ROI (mean): {metrics['roi_mean']*100:.2f}%")
                st.write(f"Win Rate (mean): {metrics['win_rate_mean']*100:.2f}%")

                trajectory_df = pd.DataFrame({"Bet #": range(len(trajectory)), "Bankroll": trajectory})
                chart_bankroll = alt.Chart(trajectory_df).mark_line(point=True).encode(
                    x="Bet #",
                    y="Bankroll",
                    tooltip=["Bet #","Bankroll"]
                ).interactive()
                st.altair_chart(chart_bankroll, use_container_width=True)

        # Export option
        export_name = f"cli_picks_{datetime.now().date()}.csv"
        st.download_button(
            label="Download Picks as CSV",
            data=df_cli.to_csv(index=False),
            file_name=export_name,
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error loading CLI results: {e}")
