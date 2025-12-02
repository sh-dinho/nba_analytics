import streamlit as st
import pandas as pd
import altair as alt
from scripts.simulate_bankroll import simulate_bankroll
from datetime import datetime

st.title("👤 Player-Level Monte Carlo")
st.caption("Run bankroll simulations for picks involving the selected player's team.")

try:
    df_trends = pd.read_csv("results/player_trends.csv")

    if df_trends.empty:
        st.info("No player trends data available.")
    else:
        # Player selector
        player_choice = st.selectbox("Select Player", sorted(df_trends["PLAYER_NAME"].unique()))
        team_abbr = df_trends.loc[df_trends["PLAYER_NAME"] == player_choice, "TEAM_ABBREVIATION"].head(1)

        if not team_abbr.empty:
            team = team_abbr.iloc[0]
            df_picks = pd.read_csv("results/picks.csv")

            # Filter picks for this player's team
            picks_for_player = df_picks[(df_picks["home_team"] == team) | (df_picks["away_team"] == team)]
            st.subheader(f"Picks involving {team} ({player_choice})")
            st.dataframe(picks_for_player, use_container_width=True)

            if picks_for_player.empty:
                st.warning("No picks available for this player's team.")
            else:
                # Sidebar controls
                sims = st.sidebar.slider("Number of simulations", 100, 5000, 1000, 100)
                strategy = st.sidebar.selectbox("Betting Strategy", ["kelly", "flat"])

                # Prepare input for simulation
                if {"home_win_prob","decimal_odds_home"}.issubset(picks_for_player.columns):
                    sim_df = picks_for_player.rename(columns={"home_win_prob":"prob","decimal_odds_home":"decimal_odds"})
                    sim_input = sim_df[["decimal_odds","prob","ev_home"]].dropna()

                    if sim_input.empty:
                        st.warning("No valid picks with odds and probabilities available for simulation.")
                    else:
                        # Run simulation
                        trajectories, metrics = simulate_bankroll(sim_input, strategy=strategy, sims=sims)

                        # Metrics summary
                        st.subheader("Simulation Results")
                        st.write(f"Final Bankroll (mean): ${metrics['final_bankroll_mean']:.2f}")
                        st.write(f"ROI (mean): {metrics['roi_mean']*100:.2f}%")
                        st.write(f"Win Rate (mean): {metrics['win_rate_mean']*100:.2f}%")
                        st.write(f"Simulations Run: {sims}")

                        # Bankroll trajectory preview (first run)
                        trajectory = trajectories[0]
                        trajectory_df = pd.DataFrame({"Bet #": range(len(trajectory)), "Bankroll": trajectory})
                        chart_bankroll = alt.Chart(trajectory_df).mark_line(point=True).encode(
                            x="Bet #",
                            y="Bankroll",
                            tooltip=["Bet #","Bankroll"]
                        ).interactive()
                        st.subheader("Sample Bankroll Trajectory")
                        st.altair_chart(chart_bankroll, use_container_width=True)

                        # Distribution of final bankrolls
                        final_bankrolls = [traj[-1] for traj in trajectories]
                        dist_df = pd.DataFrame({"Final Bankroll": final_bankrolls})
                        chart_dist = alt.Chart(dist_df).mark_bar().encode(
                            x=alt.X("Final Bankroll", bin=alt.Bin(maxbins=30)),
                            y="count()",
                            tooltip=["Final Bankroll"]
                        )
                        st.subheader("Distribution of Final Bankrolls")
                        st.altair_chart(chart_dist, use_container_width=True)

                        # Export results
                        export_name = f"player_mc_{player_choice}_{datetime.now().date()}.csv"
                        st.download_button(
                            label="Download Player-Level Simulation Results as CSV",
                            data=dist_df.to_csv(index=False),
                            file_name=export_name,
                            mime="text/csv"
                        )
        else:
            st.info("No team mapping found for selected player.")

except Exception as e:
    st.error(f"Error loading player-level Monte Carlo: {e}")