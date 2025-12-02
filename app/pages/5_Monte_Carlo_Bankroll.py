import streamlit as st
import pandas as pd
import altair as alt
from scripts.simulate_bankroll import simulate_bankroll

st.title("💰 Monte Carlo Bankroll Simulation")
st.caption("Run multiple simulations to evaluate bankroll growth under different betting strategies.")

try:
    df_picks = pd.read_csv("results/picks.csv")

    if df_picks.empty:
        st.info("No picks available. Run the pipeline first.")
    else:
        # Sidebar controls
        sims = st.sidebar.slider("Number of simulations", 100, 5000, 1000, 100)
        strategy = st.sidebar.selectbox("Betting Strategy", ["kelly", "flat"])
        
        # Prepare input for simulation
        if {"home_win_prob","decimal_odds_home"}.issubset(df_picks.columns):
            sim_df = df_picks.rename(columns={"home_win_prob":"prob","decimal_odds_home":"decimal_odds"})
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

except Exception as e:
    st.error(f"Error running Monte Carlo simulation: {e}")